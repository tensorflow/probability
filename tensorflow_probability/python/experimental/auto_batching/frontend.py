# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""AutoGraph-based auto-batching frontend."""

import contextlib
import functools
import gast
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import allocation_strategy
from tensorflow_probability.python.experimental.auto_batching import dsl
from tensorflow_probability.python.experimental.auto_batching import gast_util
from tensorflow_probability.python.experimental.auto_batching import instructions
from tensorflow_probability.python.experimental.auto_batching import lowering
from tensorflow_probability.python.experimental.auto_batching import stack_optimization as stack
from tensorflow_probability.python.experimental.auto_batching import stackless as st
from tensorflow_probability.python.experimental.auto_batching import tf_backend
from tensorflow_probability.python.experimental.auto_batching import type_inference as ab_type_inference
from tensorflow_probability.python.experimental.auto_batching import virtual_machine as vm

# TODO(mdan): Move common converters under pyct; should have no other deps.
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.common_transformers import anf
from tensorflow.python.autograph.pyct.static_analysis import activity


# For backward compatibility:
# - `compiler` will be renamed to `loader` in TF 2.2+.
# - `naming` will be moved to `core` in TF 2.2+.
# - 'converter.standard_analysis` will no longer be needed in TF 2.3+.
try:
  from tensorflow.python.autograph.pyct import naming  # pylint:disable=g-import-not-at-top
except ImportError:
  from tensorflow.python.autograph.core import naming  # pylint:disable=g-import-not-at-top
try:
  from tensorflow.python.autograph.pyct import loader  # pylint:disable=g-import-not-at-top
except ImportError:
  from tensorflow.python.autograph.pyct import compiler  # pylint:disable=g-import-not-at-top

  loader = compiler
  loader.load_ast = compiler.ast_to_object

converter.standard_analysis = getattr(
    converter, 'standard_analysis', (lambda node, *_, **__: node))


TF_BACKEND = tf_backend.TensorFlowBackend()


def _parse_and_analyze(f, autobatch_functions):
  """Performs preliminary analyses and transformations.

  The goal is to massage the source program into a form on which
  the `_AutoBatchingTransformer` below will be successful.

  Args:
    f: Function to analyze
    autobatch_functions: List of Python `str` names of autobatched functions.
      Arguments to these functions will be canonicalized to variable references,
      but others will not.

  Returns:
    node: A Python AST node representing the function, suitable for
      passing to `_AutoBatchingTransformer.visit`
    entity_info: An AutoGraph `EntityInfo` object, with some information
      about `f`.  Required for initializing `_AutoBatchingTransformer`.
  """
  # TODO(mdan): Replace all this boilerplate with FunctionTranspiler.
  namespace = {}

  # Get the AST of the function
  future_features = inspect_utils.getfutureimports(f)
  node, _ = parser.parse_entity(f, future_features=future_features)

  # Boilerplate for AutoGraph transforms
  if hasattr(converter, 'EntityContext'):
    # TF 2.2-
    entity_info = transformer.EntityInfo(
        source_code='',
        source_file=None,
        future_features=future_features,
        namespace=namespace)
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=None)
    ctx = converter.EntityContext(
        namer=naming.Namer(namespace),
        entity_info=entity_info,
        program_ctx=program_ctx)
  else:
    # TF 2.3+
    entity_info = transformer.EntityInfo(
        name=f.__name__,
        source_code='',
        source_file=None,
        future_features=future_features,
        namespace=namespace)
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=None)
    ctx = transformer.Context(
        info=entity_info,
        namer=naming.Namer(namespace),
        user_context=program_ctx)

  # Canonicalize away break statements
  node = converter.standard_analysis(node, ctx)
  node = break_statements.transform(node, ctx)

  # Canonicalize away continue statements
  node = converter.standard_analysis(node, ctx)
  node = continue_statements.transform(node, ctx)

  # Force single returns
  node = converter.standard_analysis(node, ctx)
  node = return_statements.transform(node, ctx, default_to_null_return=False)

  # Transform into ANF
  # Replacing if tests and autobatched function call arguments because
  # that's where divergence can happen.
  # Replacing all function calls because the downstream transformation
  # expects calls to lead directly to assignments.
  def maybe_replace_function_argument(parent, field_name, child):
    del field_name, child
    if not anno.hasanno(parent.func, anno.Basic.QN):
      return False
    func_name = anno.getanno(parent.func, anno.Basic.QN)
    if str(func_name) in autobatch_functions:
      return True
    return False

  anf_config = [
      (anf.ASTEdgePattern(gast.If, 'test', anf.ANY), anf.REPLACE),
      (anf.ASTEdgePattern(anf.ANY, anf.ANY, gast.Call), anf.REPLACE),
      (anf.ASTEdgePattern(gast.Call, 'args', anf.ANY),
       maybe_replace_function_argument),
  ]
  node = anf.transform(node, ctx, config=anf_config)
  node = qual_names.resolve(node)
  node = activity.resolve(node, ctx)

  return node, ctx


class _AutoBatchingTransformer(transformer.Base):
  """A subclass of `pyct.transformer.Base` implementing auto-batching.

  Specifically, converts some Python source P into a Python function that will
  invoke methods on a `dsl.ProgramBuilder` to register an auto-batchable version
  of P.  This class is private: use through the `transform` wrapper.

  The input is expected to be in A-normal form, so that all intermediate values
  are explicitly assigned.  As such, this class need only intercept function
  definitions, assignment statements, and control constructs.  At the moment,
  only supporting if, function call, and return.  TODO(axch): Expand the subset.
  """

  def __init__(self, known_functions, enclosing_names, ctx):
    """Initializes an _AutoBatchingTransformer.

    Args:
      known_functions: List of Python strings.  These are the names of
        auto-batched functions.  Calls to the functions on this list (when
        recognized) are transformed into `ProgramBuilder.call` methods.  Other
        function calls become `ProgramBuilder.primop`.
      enclosing_names: List of Python variables.  These are names to draw from
        the transformee's enclosing scope (without turning them into autobatch
        variables).
      ctx: An AutoGraph `autograph.converter.EntityContext` object describing
        the node to be transformed, as returned by `_parse_and_analyze`.
        Inherits from `autograph.transformer.Context`, see
        `autograph.transformer.Base`.
    """
    self.known_functions = known_functions
    self.enclosing_names = enclosing_names
    super(_AutoBatchingTransformer, self).__init__(ctx)

  def visit_FunctionDef(self, node):
    """Intercepts function definitions.

    Converts function definitions to the corresponding `ProgramBuilder.function`
    construction.

    Args:
      node: An `ast.AST` node representing the function to convert.

    Returns:
      node: An updated node, representing the result.

    Raises:
      ValueError: If the input node does not adhere to the restrictions,
        e.g., failing to have a `return` statement at the end.
    """
    # Check input form
    return_node = node.body[-1]
    if not isinstance(return_node, gast.Return):
      msg = 'Last node in function body should be Return, not {}.'
      raise ValueError(msg.format(return_node))

    # Convert all args to _tfp_autobatching_context_.param()
    local_declarations = []
    for arg in node.args.args:
      # print('Creating param declaration for', arg, arg.id, type(arg.id))
      local_declarations.append(templates.replace(
          'target = _tfp_autobatching_context_.param(name=target_name)',
          target=arg.id,
          target_name=gast_util.Str(arg.id))[0])

    # Visit the content of the function
    node = self.generic_visit(node)

    # Prepend the declarations
    node.body = local_declarations + node.body

    # Convert the function into a
    # `with _tfp_autobatching_context_.define_function()` block.
    # Wrap the `with` block into a function so additional information (namely,
    # the auto-batching `ProgramBuilder` and the `instruction.Function`s that
    # may be called in the body) can be passed in through regular Python
    # variable references.
    callable_function_names = [
        gast_util.Name(n, ctx=gast.Store(), annotation=None)
        for n in self.known_functions]
    node = templates.replace(
        '''
        def func(_tfp_autobatching_context_,
                 _tfp_autobatching_available_functions_):
          names = _tfp_autobatching_available_functions_
          with _tfp_autobatching_context_.define_function(func):
            body
          return func''',
        func=node.name,
        names=gast.List(callable_function_names, ctx=gast.Store()),
        body=node.body)[0]

    return node

  def visit_Assign(self, node):
    """Intercepts assignment statements.

    The input is expected to be in A-normal form, so every assignment represents
    at most one interesting computation, which is the AST node serving as the
    assignment's right-hand side.

    Args:
      node: An `ast.AST` node representing the assignment to convert.

    Returns:
      node: An updated node, representing the result.

    Raises:
      ValueError: If the node is a call to a function whose name cannot be
        determined.
    """
    # If we're assigning a constant value, use the `ProgramBuilder.const`
    # shorthand.
    if gast_util.is_literal(node.value):
      # Emit `_tfp_autobatching_context_.const`
      node = templates.replace(
          'target = _tfp_autobatching_context_.const(value)',
          target=self._assignment_construct(node.targets[0]),
          value=node.value)
      return node

    # If we're calling into a function, and the VM knows about the function, use
    # the `ProgramBuilder.call` instruction; otherwise, we'll emit a
    # `ProgramBuilder.primop` instruction, with a lambda staging the computation
    # to be performed.
    elif isinstance(node.value, gast.Call):
      if not anno.hasanno(node.value.func, anno.Basic.QN):
        # TODO(axch): Make calls to unidentified functions fall back to
        # primop, or flag them?
        raise ValueError('Cannot resolve name of the called function')
      func_name = anno.getanno(node.value.func, anno.Basic.QN)
      used_vars = node.value.args
      if str(func_name) in self.known_functions:
        # Emit `_tfp_autobatching_context_.call`
        node = templates.replace(
            'target = _tfp_autobatching_context_.call(func_name, vars_in=args)',
            target=self._assignment_construct(node.targets[0]),
            func_name=func_name,
            args=self._to_reference_list(used_vars))
        return node
      else:
        return self._primop(node, [func_name])
    else:
      return self._primop(node, [])

  def _primop(self, node, non_autobatch_names):
    scope = anno.getanno(node, anno.Static.SCOPE)
    # We want the root names of any attribute accesses; the accesses will happen
    # inside the expression.
    used_vars = set().union(*[qn.support_set for qn in scope.read])
    # Exclude names from the blocklist.
    used_vars = list(used_vars - set(non_autobatch_names))
    # Exclude names that will be available from the enclosing scope.
    used_vars = [v for v in used_vars if str(v) not in self.enclosing_names]
    # TODO(aqj) It should not be necessary to convert these QN objects to
    # strings, but empirically made the difference.
    stringified = [str(name) for name in used_vars]
    node = templates.replace(
        ('target = _tfp_autobatching_context_.primop('
         'lambda args: expr, vars_in=refs)'),
        target=self._assignment_construct(node.targets[0]),
        args=stringified,
        refs=self._to_reference_list(used_vars),
        expr=node.value)
    return node

  def _assignment_construct(self, target):
    if isinstance(target, (gast.Tuple, gast.List)):
      return templates.replace_as_expression(
          '_tfp_autobatching_context_(pat).pattern',
          pat=self._assignment_construct_recur(target))
    return self._assignment_construct_recur(target)

  def _assignment_construct_recur(self, target):
    if isinstance(target, (gast.Tuple, gast.List)):
      subs = [self._assignment_construct_recur(t) for t in target.elts]
      if isinstance(target, gast.Tuple):
        # Context is not Store anymore, because this section is constructing the
        # pattern object
        return gast.Tuple(subs, ctx=gast.Load())
      else:
        # Context is not Store anymore, because this section is constructing the
        # pattern object
        return gast.List(subs, ctx=gast.Load())
    return templates.replace_as_expression(
        '_tfp_autobatching_context_.var.name', name=target)

  def _to_reference(self, node):
    if isinstance(node, (gast.Name, qual_names.QN)):
      return templates.replace_as_expression(
          '_tfp_autobatching_context_.var.name', name=node)
    elif gast_util.is_literal(node):
      raise ValueError('TODO(axch): Support literals, not just variables')
    else:
      msg = 'Expected trivial node, got {}.  Is the input in A-normal form?'
      raise ValueError(msg.format(node))

  def _to_reference_list(self, names):
    return gast.List(
        [self._to_reference(name) for name in names], ctx=gast.Load())

  def visit_If(self, node):
    """Intercepts if statements.

    Converts each `if` to up to two separate `with` statements,
    `ProgramBuilder.if_(condition_variable)` and `ProgramBuilder.else_()`.  If
    the incoming `if` had one arm, returns the transformed AST node; if it had
    two, returns two nodes in a list.

    Args:
      node: An `ast.AST` node representing the `if` statement to convert.

    Returns:
      then_node: A node representing the `with`-guarded consequent branch.
      else_node: A node representing the `with`-guarded alternate branch,
        if present.
    """
    # Transform a branch
    # NOTE: this is a little hackery to make sure that prepending works
    # properly. Wrapping a list of statements in a Module ensures
    # that the AST-visiting machinery won't choke on, e.g., a list.
    then = self.generic_visit(gast_util.Module(node.body)).body

    # Construct header (goes in the `with`s).
    then_header = templates.replace_as_expression(
        '_tfp_autobatching_context_.if_(cond)',
        cond=self._to_reference(node.test))

    # Construct `with` node.
    # TODO(axch): Test that this form actually works with multiline bodies.
    then_node = templates.replace(
        'with header: body', header=then_header, body=then)[0]

    if node.orelse:
      orelse = self.generic_visit(gast_util.Module(node.orelse)).body
      orelse_header = templates.replace_as_expression(
          '_tfp_autobatching_context_.else_()')
      orelse_node = templates.replace(
          'with header: body', header=orelse_header, body=orelse)[0]
      # Return both
      return [then_node, orelse_node]
    else:
      return then_node

  def visit_Return(self, node):
    """Intercepts return statements.

    Args:
      node: An `ast.AST` node representing the `return` statement to convert.

    Returns:
      node: A node representing the result.
    """
    node = templates.replace_as_expression(
        '_tfp_autobatching_context_.return_(value)',
        value=self._to_reference(node.value))
    return gast.Expr(node)


class Context(object):
  """Context object for auto-batching multiple Python functions together.

  Warning: This is alpha-grade software.  The exact subset of Python that can be
  successfully auto-batched is ill-specified and subject to change.
  Furthermore, the errors elicited by stepping outside this subset are likely to
  be confusing and misleading.  Use at your own risk.

  Usage:
  ```python
  ctx = frontend.Context()

  @ctx.batch(type_inference=lambda ...)
  def my_single_example_function_1(args):
    ...

  @ctx.batch(type_inference=lambda ...)
  def my_single_example_function_2(args):
    ...

  # etc
  ```

  Then calling any of the decorated functions will execute a batch computation.
  The decorated functions may call each other, including mutually recursively.
  See also the `batch` method.

  Limitations:
  - You must explicitly decorate every function to be auto-batched.
  - All calls to them must call them by name (no higher-order auto-batching).
  - Auto-batched functions must be defined with `def`, not `lambda`.
  """

  def __init__(self):
    """Initializes a `Context` object."""
    self._tagged_functions = []
    self._module = None
    self._in_dry_run = False
    # LRU-1 cache of compilation results, keyed by program, backend, and input
    # type signature.
    self._compile_cache = None
    # LRU-1 cache of lowering results, keyed by program, backend, and input
    # type signature.  These are separate because stackless mode requires
    # a non-lowered program, but lowering mutates (b/119122199).
    self._lowering_cache = None

  def _tag_function(self, function, type_inference):
    self._tagged_functions.append((function, type_inference))
    self._module = None

  @contextlib.contextmanager
  def _dry_run(self):
    old_dry_run = self._in_dry_run
    try:
      self._in_dry_run = True
      yield
    finally:
      self._in_dry_run = old_dry_run

  def batch(self, type_inference):
    """Decorates one function to auto-batch.

    The decorated function will run in batch.  It accepts all the same
    arguments, except:
    - All arguments must have an additional leading dimension for the batch.
      (By special dispensation, scalar inputs are promoted to shape [1], which
      then leads to broadcasting.)
    - All the arguments' sizes in the batch dimension must be the same, or 1.
      The latter are broadcast.
    - The returned value will also have a leading batch dimension, and
      will have the same size.
    - The batched function accepts an additional `bool` keyword argument
      `dry_run`.  If present and `True`, just calls the unbatched version,
      circumventing the auto-batching system.  This can be useful for
      debugging the program subject to auto-batching.
    - The batched function accepts an additional `bool` keyword argument
      `stackless`.  If present and `True`, invokes the stackless version of the
      auto-batching system.  This can be useful for avoiding stack maintenance
      overhead; but in general, it will recover less batching, and not work in
      graph-mode TensorFlow.
    - The batched function accepts an additional `int` keyword argument
      `max_stack_depth` specifying the maximum stack depth (default 15).
      Ignored in stackless execution.
    - The batched function accepts an additional keyword argument `backend`
      specifying the backend to use.  Must be an instance of
      `auto_batching.TensorFlowBackend` (default) or
      `auto_batching.NumpyBackend`.
    - The batched function accepts an additional keyword argument
      `block_code_cache`, a dict which allows the caching of basic block
      rewrites (i.e. `tf.function` + XLA) to live across calls to the
      autobatched function. The default value of `None` results in caching only
      within a given call to the batched function. Currently, stackless
      autobatching ignores the cache completely.

    Note: All functions called by the decorated function that need auto-batching
    must be decorated in the same `Context` before invoking any of them.

    Args:
      type_inference: A Python callable giving the type signature of the
        function being auto-batched.  The callable will be invoked with a single
        argument giving the list of `instructions.Type` objects describing the
        arguments at a particular call site, and must return a list of
        `instructions.Type` objects describing the values that call site will
        return.

    Returns:
      dec: A decorator that may be applied to a function with the given
        type signature to auto-batch it.

    Raises:
      ValueError: If the decorated function predictably cannot be auto-batched,
        e.g., name-clashing with another function already decorated in this
        `Context`.
    """
    return functools.partial(
        self.batch_uncurried, type_inference=type_inference)

  def batch_uncurried(self, function, type_inference):
    """A non-decorator version of `batch`, which see."""
    # TODO(axch): Friendly error message if `function` is not actually a
    # function.
    # TODO(axch): Friendly error message if `function` is a lambda and not a
    # `def`ed function.  Status quo will fail horribly on lambdas, because
    # the transformation requires the transformed functions be named.
    # Determine the name of the function being transformed
    name = function.__name__
    if name in self.function_names():
      msg = ('Cannot auto-batch the same-named function {} '
             'twice in the same `Context`.').format(name)
      raise ValueError(msg)
    self._tag_function(function, type_inference)
    def batched(*args, **kwargs):
      """The batched function."""
      # Accepting kwargs here because Python 2.7 gets confused otherwise.
      # See, e.g.,
      # https://stackoverflow.com/questions/14003939/python-function-args-and-kwargs-with-other-specified-keyword-arguments
      # The whole strategy of adding arguments to the decorated function has
      # the unfortunate consequence that pylint uses the argument signature of
      # the decoratee to issue warnings, with the result that
      # disable=unexpected-keyword-arg needs to be added to use sites of
      # decorated functions, for example in `frontend_test.py`.
      # TODO(axch) Figure out how to avoid pylint warnings in callers.
      # Options:
      # - Add `backend` and `max_stack_depth` arguments to the `Context`
      #   constructor, and use those as defaults here.
      #   - Likely insufficient, because dynamic selection of the stack size
      #     seems important.
      # - Carry the same information in a context manager, used like
      #     with ctx.config(backend=TF_BACKEND, max_stack_depth=15):
      #       some_decorated_function(normal, arguments)
      # - Just invoke batched functions from the `Context` object:
      #     ctx.run(some_decorated_function, input_1, max_stack_depth=15)
      # - Maybe even make magic methods for the above?
      #     ctx.some_decorated_function(input_1, max_stack_depth=15)
      max_stack_depth = kwargs.pop('max_stack_depth', 15)
      backend = kwargs.pop('backend', TF_BACKEND)
      dry_run = kwargs.pop('dry_run', False)
      stackless = kwargs.pop('stackless', False)
      block_code_cache = kwargs.pop('block_code_cache', {})
      if self._in_dry_run:
        return _run_at_batch_size_one(backend, function, args)
      if dry_run:
        with self._dry_run():
          return _run_at_batch_size_one(backend, function, args)
      if kwargs:
        msg = 'Auto-batched function given unexpected keyword arguments {}.'
        raise TypeError(msg.format(kwargs.keys()))
      sig = ab_type_inference.signature(self.program(main=name), args, backend)
      if stackless:
        compiled = self.program_compiled(main=name, sig=sig, backend=backend)
        return st.execute(compiled, backend, block_code_cache, *args)
      else:
        lowered = self.program_lowered(main=name, sig=sig, backend=backend)
        return vm.execute(lowered, args, max_stack_depth, backend,
                          block_code_cache=block_code_cache)
    return batched

  def function_names(self):
    return [f.__name__ for (f, _) in self._tagged_functions]

  def module(self):
    """Constructs an `instructions.Module` for this `Context`.

    Returns:
      module: An `instructions.Module` representing the batched computation
        defined by all the functions decorated with `batch` in this `Context` so
        far.
    """
    if self._module is not None:
      return self._module
    ab = dsl.ProgramBuilder()
    function_objects = []
    for function, type_inference in self._tagged_functions:
      declared = ab.declare_function(function.__name__, type_inference)
      function_objects.append(declared)
    for function, _ in self._tagged_functions:
      name = function.__name__
      node, ctx = _parse_and_analyze(function, self.function_names())
      node = _AutoBatchingTransformer(
          self.function_names(),
          [scoped_name for scoped_name, _ in _environment(function, [name])],
          ctx).visit(node)
      builder_module, _, _ = loader.load_ast(node)
      for scoped_name, val in _environment(function, [name]):
        builder_module.__dict__[scoped_name] = val
      builder = getattr(builder_module, name)
      builder(ab, function_objects)
    self._module = ab.module()
    return self._module

  def program(self, main):
    """Constructs an `instructions.Program` for this `Context`.

    This is a helper method, equivalent to `self.module().program(main)`.

    Args:
      main: Python string name of the function that should be the entry point.

    Returns:
      prog: An `instructions.Program` representing the batched computation
        defined by all the functions decorated with `batch` in this `Context` so
        far.  Suitable for downstream compilation with other passes in
        `auto_batching`.

    Raises:
      ValueError: If the intended `main` function was not decorated with
        `batch`.
    """
    return self.module().program(main)

  def program_compiled(self, main, sig=None, backend=None):
    """Constructs a compiled `instructions.Program` for this `Context`.

    This constructs the program with `self.program(main)`, and the performs type
    inference and optimization, to emit a result that can be executed by the
    stackless auto-batching VM.

    The point of having this as a method in its own right is that it caches the
    compilation on the types of the arguments.

    If either `sig` or `backend` are omitted or `None`, type inference is
    skipped.  The result is not executable, but it can be enlightening to
    inspect.

    Args:
      main: Python string name of the function that should be the entry point.
      sig: A `list` of (patterns of) `instructions.TensorType` aligned with
        the formal parameters to `main`.
      backend: Backend implementation.

    Returns:
      prog: An `instructions.Program` representing the batched computation
        defined by all the functions decorated with `batch` in this `Context` so
        far.  Suitable for execution or staging on real data by the
        auto-batching VM.
    """
    module = self.module()
    prog = module.program(main)
    if self._compile_cache is not None:
      key, result = self._compile_cache
      if key == (module, main, sig, backend):
        return result
      else:
        # Clear the module cache as well, because of b/119122199
        self._module = None
        module = self.module()
        prog = module.program(main)
    if sig is not None and backend is not None:
      typed = ab_type_inference.infer_types_from_signature(prog, sig, backend)
    else:
      typed = prog
    result = allocation_strategy.optimize(typed)
    self._compile_cache = ((module, main, sig, backend), result)
    return result

  def program_lowered(self, main, sig=None, backend=None):
    """Constructs a lowered `instructions.Program` for this `Context`.

    This constructs the program with `self.program(main)`, and the performs type
    inference, optimization, and lowering, to emit a result that can be executed
    (or staged) by the auto-batching VM.

    The point of having this as a method in its own right is that it caches the
    compilation on the types of the arguments.

    If either `sig` or `backend` are omitted or `None`, type inference is
    skipped.  The result is not executable, but it can be enlightening to
    inspect.

    Args:
      main: Python string name of the function that should be the entry point.
      sig: A `list` of (patterns of) `instructions.TensorType` aligned with
        the formal parameters to `main`.
      backend: Backend implementation.

    Returns:
      prog: An `instructions.Program` representing the batched computation
        defined by all the functions decorated with `batch` in this `Context` so
        far.  Suitable for execution or staging on real data by the
        auto-batching VM.
    """
    module = self.module()
    prog = module.program(main)
    if self._lowering_cache is not None:
      key, result = self._lowering_cache
      if key == (module, main, sig, backend):
        return result
      else:
        # Clear the module and compile caches as well, because of b/119122199
        self._module = None
        self._compile_cache = None
        module = self.module()
        prog = module.program(main)
    if sig is not None and backend is not None:
      typed = ab_type_inference.infer_types_from_signature(prog, sig, backend)
    else:
      typed = prog
    alloc = allocation_strategy.optimize(typed)
    lowered = lowering.lower_function_calls(alloc)
    result = stack.fuse_pop_push(lowered)
    self._lowering_cache = ((module, main, sig, backend), result)
    return result

  def lowered_for_args(self, name, args, backend):
    """Helper for calling program_lowered that computes the type signature."""
    sig = ab_type_inference.signature(self.program(main=name), args, backend)
    lowered = self.program_lowered(main=name, sig=sig, backend=backend)
    return lowered


def _environment(function, names_to_omit=()):
  """Yields the names and values visible from the function's scope."""
  str_names_to_omit = set(map(str, names_to_omit))
  for name, val in six.iteritems(six.get_function_globals(function)):
    if str(name) not in str_names_to_omit:
      yield name, val
  closure = six.get_function_closure(function)
  if closure is not None:
    freevars = six.get_function_code(function).co_freevars
    for name, cell in zip(freevars, closure):
      if str(name) not in str_names_to_omit:
        yield name, cell.cell_contents


def _run_at_batch_size_one(backend, function, args):
  """Executes the given function, checking in- and out-batch shape."""
  for arg in args:
    for type_ in instructions.pattern_traverse(
        ab_type_inference.type_of_pattern(arg, backend),
        leaf_type=instructions.TensorType):
      shape = type_.shape
      if len(shape) >= 1 and shape[0] != 1:
        msg = 'Expecting input batch dimension of size 1; got {} of shape {}.'
        raise ValueError(msg.format(arg, shape))
  result = function(*args)
  for item in instructions.pattern_traverse(result):
    for type_ in instructions.pattern_traverse(
        ab_type_inference.type_of_pattern(item, backend),
        leaf_type=instructions.TensorType):
      shape = type_.shape
      if len(shape) >= 1 and shape[0] != 1:
        msg = 'Expecting result batch dimension of size 1; got {} of shape {}.'
        raise ValueError(msg.format(item, shape))
  return result


def truthy(x):
  """Normalizes Tensor ranks for use in `if` conditions.

  This enables dry-runs of programs with control flow.  Usage: Program the
  conditions of `if` statements and `while` loops to have a batch dimension, and
  then wrap them with this function.  Example:
  ```python
  ctx = frontend.Context
  truthy = frontend.truthy

  @ctx.batch(type_inference=...)
  def my_abs(x):
    if truthy(x > 0):
      return x
    else:
      return -x

  my_abs([-5], dry_run=True)
  # returns [5] in Eager mode
  ```

  This is necessary because auto-batched programs still have a leading batch
  dimension (of size 1) even in dry-run mode, and a Tensor of shape [1] is not
  acceptable as the condition to an `if` or `while`.  However, the leading
  dimension is critical during batched execution; so conditions of ifs need to
  have rank 1 if running batched and rank 0 if running unbatched (i.e.,
  dry-run).  The `truthy` function arranges for this be happen (by detecting
  whether it is in dry-run mode or not).

  If you missed a spot where you should have used `truthy`, the error message
  will say `Non-scalar tensor <Tensor ...> cannot be converted to boolean.`

  Args:
    x: A Tensor.

  Returns:
    x: The Tensor `x` if we are in batch mode, or if the shape of `x` is
      anything other than `[1]`.  Otherwise returns the single scalar in `x` as
      a Tensor of scalar shape (which is acceptable in the conditions of `if`
      and `while` statements.
  """
  if ab_type_inference.is_inferring() or vm.is_staging() or st.is_running():
    return x
  elif isinstance(x, tf.Tensor):
    if x.shape == [1]:
      return x[0]
    else:
      return x
  else:
    return x
