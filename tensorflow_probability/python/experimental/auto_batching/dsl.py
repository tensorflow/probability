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
"""Python-embedded DSL frontend for authoring autobatchable IR programs.

This domain-specific language frontend serves two purposes:
- Being easier and more pleasant to author by humans than writing IR directly
- Being close enough to the structure of natural Python programs that
  the remaining gap can be bridged by a source-to-source transformation
"""

import collections
import contextlib
import inspect

# Dependency imports

from tensorflow_probability.python.experimental.auto_batching import instructions as inst

__all__ = [
    'ProgramBuilder'
]


class ProgramBuilder(object):
  """An auto-batching DSL context.

  Auto-batching DSL operations are methods on the `ProgramBuilder` object.  It's
  used like this:

  ```python
  ab = dsl.ProgramBuilder()

  def fib_type(arg_types):
    return arg_types[0]

  with ab.function(type_inference=fib_type) as fibonacci:
    n = ab.param('n')
    ab.var.cond = ab.primop(lambda n: n > 1)
    with ab.if_(ab.var.cond):
      ab.var.nm1 = ab.primop(lambda n: n - 1)
      ab.var.fibm1 = ab.call(fibonacci, [ab.var.nm1])
      ab.var.nm2 = ab.primop(lambda n: n - 2)
      ab.var.fibm2 = ab.call(fibonacci, [ab.var.nm2])
      ab.var.ans = ab.primop(lambda fibm1, fibm2: fibm1 + fibm2)
    with ab.else_():
      ab.var.ans = ab.const(1)
    ab.return_(ab.var.ans)

  prog = ab.program(main=fibonacci)
  # Now `prog` is a well-formed `instructions.Program`, and the context
  # `ab` is no longer needed.
  ```

  Note the sequence of method calls on `ProgramBuilder` corresponds to the
  source code of the `Program` being defined, not its runtime behavior.  This is
  because (a) functions are defined with a context manager (rather than a Python
  function) which executes its body immediately and exactly once; and (b)
  function call instructions (and primitive operations) are just recorded, not
  entered recursively.
  """

  # Implementation invariants:
  # - Outside a function body, `self._blocks` is `None`, `self._locals` is
  #   `None` and `self._pending_after_else_block` is `None`.  Only `function`
  #   and `program` methods are valid; `function` puts the context into the
  #   "inside a body" state during its yield.
  # - Inside a body, `function` and `program` calls are not permitted.  At the
  #   beginning and end of every public method call except `return_`, there is a
  #   last `Block` in `self._blocks` with a non-`None` instruction list, and a
  #   `None` terminator.  All other `Block`s in `self._blocks` have instruction
  #   lists and terminators, and will not be mutated further.
  # - Inside a body, `self._locals` maps the potentially non-unique names of
  #   in-scope local auto-batching variables to their unique global names.
  # - `self._pending_after_else_block` is not `None` if and only if an `if_` at
  #   this level just terminated, and we have not yet ascertained whether the
  #   next method call at this level is an `else_`.
  # - `self._var_defs` stores the list of globally unique names for currently
  #   defined variables.  Is an OrderedDict so that membership can be tested in
  #   O(1) while maintaining a stable traversal order (e.g., for printing the
  #   constructed `Program`).
  # - `self.functions` stores the list of defined `instructions.Function`
  #   objects.

  def __init__(self):
    """Creates an empty ProgramBuilder."""
    self._blocks = None
    self._locals = None
    self._magic_vars = _MagicVars(self)
    self._pending_after_else_block = None
    self._functions = []
    self._var_defs = collections.OrderedDict()
    self._varcount = 0

  def _fresh_block(self, name=None):
    return inst.Block(instructions=[], name=name)

  def _append_block(self, block, prev_terminator=None):
    if prev_terminator is None:
      prev_terminator = inst.GotoOp(block)
    assert self._blocks[-1].terminator is None, 'Internal invariant violation'
    assert block.instructions is not None, 'Internal invariant violation'
    assert block.terminator is None, 'Internal invariant violation'
    self._blocks[-1].terminator = prev_terminator
    self._blocks.append(block)

  def _prepare_for_instruction(self):
    if self._blocks is None:
      msg = "Can't invoke an instruction outside a function body"
      raise ValueError(msg)
    assert self._locals is not None, 'Internal invariant violation'
    if self._pending_after_else_block is not None:
      self._append_block(self._pending_after_else_block)
      self._pending_after_else_block = None

  def _update_last_instruction(self, new):
    self._blocks[-1].instructions[-1] = new

  @property
  def var(self):
    """Auto-batching variables visible in the current scope.

    Overrides `setattr` and `getattr` to provide a smooth interface
    to reading and defining variables:

    - `ProgramBuilder.var.foo = ProgramBuilder.{call,primop}`
      records an assignment to the auto-batched variable `foo`, possibly
      binding it, and

    - `ProgramBuilder.var.foo` reads from the auto-batched variable `foo`
      (if it is bound).

    Example:

    ```python
    ab = dsl.ProgramBuilder()

    ab.var.seven = ab.const(7)
    ```

    Returns:
      vars: A `_MagicVars` instance representing the local scope as above.
    """
    return self._magic_vars

  def __call__(self, pattern):
    """Prepares a multi-value return.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    ab((ab.var.two, ab.var.four)).pattern = ab.const((2, 4))
    ```

    The protocol is to create a magic pattern object by invoking the
    `ProgramBuilder` as a callable, passing the pattern to bind; then assigning
    the `pattern` attribute of the returned value to the operation whose values
    to accept.

    This is like this to work around limitations of embedding a DSL into Python:
    the assignment syntax `=` can be overridden _only_ for fields of objects,
    not for function calls.  It would have been nicer to implement
    `ab.pattern(...) = ...` but that's syntactically invalid Python.  Hence,
    putting the `pattern` token at the end of the phrase rather than the
    beginning.

    Args:
      pattern: A pattern of variables (e.g., from `ab.var.name`) to bind.

    Returns:
      pat_object: A `_MagicPattern` instance representing the putative binding.
        Invoke the `pattern =` attribute setter on that instance to actually
        bind this pattern as the output of a `primop`, `const`, or `call`.
    """
    return _MagicPattern(self, pattern)

  def local(self, name=None, define=True):
    """Declares a local variable in the current scope.

    This should typically not be needed, because `ProgramBuilder.var.foo =` can
    bind variables; however, may be helpful for a multivalue return (see
    `primop` or `call`).

    Args:
      name: Optional Python string to serve a mnemonic name in later compiler
        stages.  Variable names are automatically uniqued.  This variable can
        later be referred to with `ProgramBuilder.var.name`, as well as through
        any Python binding of the returned value.
      define: Boolean giving whether to mark this variable defined on creation.
        Default `True`.  Setting `False` is useful for speculatively uniquing a
        variable on its first appearance, before knowning whether said
        appearance is a write (in which case the variable becomes defined) or a
        read (which raises an error).

    Returns:
      var: A Python string representing this variable.  Suitable for passing
        to `primop`, `call`, `if_`, and `return_`.
    """
    self._prepare_for_instruction()
    self._varcount += 1
    if name is not None:
      var = '{}_{}'.format(name, self._varcount)
    else:
      var = 'var_{}'.format(self._varcount)
    if name is not None and name not in self._locals:
      self._locals[name] = var
    if define:
      self._mark_defined(var)
    return var

  def _mark_defined(self, var):
    if var not in self._var_defs:
      self._var_defs[var] = True

  def locals_(self, count, name=None):
    """Declares several variables at once.

    This is a convenience method standing for several invocations of `local`.

    Args:
      count: Python int.  The number of distinct variables to return.
      name: Optional Python string to serve a mnemonic name in later compiler
        stages.  Variable names are automatically uniqued.

    Returns:
      vars: A list of `count` Python strings representing these variables.
        Suitable for passing to `primop`, `call`, `if_`, and `return_`.
    """
    # TODO(axch): Should be able to get rid of the count argument by
    # returning in infinite generator of variables, if Python can
    # pattern-match a finite tuple against that effectively.
    return [self.local(name) for _ in range(count)]

  def param(self, name=None):
    """Declares a parameter of the function currently being defined.

    This make a local variable like `local`, but also makes it an input of the
    nearest enclosing function (created by `with ProgramBuilder.function()`).
    This is a separate method from `function` because the DSL wants to create
    Python bindings for the function name itself and all of its input
    parameters, and there is no way to convince the `with` syntax to do that.

    Args:
      name: Optional Python string to serve a mnemonic name in later compiler
        stages.  Variable names are automatically uniqued.

    Returns:
      var: A Python string representing this variable.  Suitable for passing
        to `primop`, `call`, `if_`, and `return_`.
    """
    var = self.local(name)
    self._functions[-1].vars_in.append(str(var))
    return var

  def primop(self, f, vars_in=None, vars_out=None):
    """Records a primitive operation.

    Example:

    ```
    ab = dsl.ProgramBuilder()

    ab.var.five = ab.const(5)
    # Implicit output binding
    ab.var.ten = ab.primop(lambda five: five + five)
    # Explicit output binding
    ab.primop(lambda: (5, 10), vars_out=[ab.var.five, ab.var.ten])
    ```

    Args:
      f: A Python callable, the primitive operation to perform.  Can be
        an inline lambda expression in simple cases.  Must return a list or
        tuple of results, one for each intended output variable.
      vars_in: A list of Python strings, giving the auto-batched variables
        to pass into the callable when invoking it.  If absent, `primop`
        will try to infer it by inspecting the argument list of the callable
        and matching against variables bound in the local scope.
      vars_out: A pattern of Python strings, giving the auto-batched variable(s)
        to which to write the result of the callable.  Defaults to the empty
        list.

    Raises:
      ValueError: If the definition is invalid, if the primop references
        undefined auto-batched variables, or if auto-detection of input
        variables fails.

    Returns:
      op: An `instructions.PrimOp` instance representing this operation.  If one
        subsequently assigns this to a local, via `ProgramBuilder.var.foo = op`,
        that local becomes the output pattern.
    """
    self._prepare_for_instruction()
    if vars_out is None:
      vars_out = []
    if vars_in is None:
      # Deduce the intended variable names from the argument list of the callee.
      # Expected use case: the callee is an inline lambda expression.
      args, varargs, varkw, _, kwonlyargs, _, _ = inspect.getfullargspec(f)
      vars_in = []
      for arg in args + kwonlyargs:
        if arg in self._locals:
          vars_in.append(self._locals[arg])
        else:
          raise ValueError('Auto-referencing unbound variable {}.'.format(arg))
      if varargs is not None:
        raise ValueError('Varargs are not supported for primops')
      if varkw is not None:
        raise ValueError('kwargs are not supported for primops')
    for var in vars_in:
      if var not in self._var_defs:
        raise ValueError('Referencing undefined variable {}.'.format(var))
    prim = inst.prim_op(_str_list(vars_in), inst.pattern_map(str, vars_out), f)
    self._blocks[-1].instructions.append(prim)
    for var in inst.pattern_traverse(vars_out):
      self._mark_defined(var)
    return prim

  def const(self, value, vars_out=None):
    """Records a constant or set of constants.

    Like `primop`, the output variables can be specified explicitly via the
    `vars_out` argument or implicitly by assigning the return value to
    some `ProgramBuilder.var.foo`.

    Args:
      value: A Python list of the constants to record.
      vars_out: A pattern of Python strings, giving the auto-batched variable(s)
        to which to write the result of the callable.  Defaults to the empty
        list.

    Returns:
      op: An `instructions.PrimOp` instance representing this operation.  If one
        subsequently assigns this to a local, via `ProgramBuilder.var.foo = op`,
        that local gets added to the list of output variables.
    """
    return self.primop(lambda: value, vars_in=[], vars_out=vars_out)

  @contextlib.contextmanager
  def if_(self, condition, then_name=None, else_name=None, continue_name=None):
    """Records a conditional operation and `true` first branch.

    The `false` branch, if present, must be guarded by a call to `else_`, below.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    ab.var.true = ab.const(True)
    with ab.if_(ab.var.true):
      ...  # The body of the `with` statement gives the `true` branch
    with ab.else_():  # The else_ clause is optional
      ...
    ```

    Args:
      condition: Python string giving the boolean variable that holds the branch
        condition.
      then_name: Optional Python string naming the true branch when the program
        is printed.
      else_name: Optional Python string naming the false branch when the program
        is printed.
      continue_name: Optional Python string naming the continuation after the if
        when the program is printed.

    Yields:
      Nothing.

    Raises:
      ValueError: If trying to condition on a variable that has not been
        written to.
    """
    # Should I have _prepare_for_instruction here?
    self._prepare_for_instruction()
    if condition not in self._var_defs:
      raise ValueError(
          'Undefined variable {} used as if condition.'.format(condition))
    then_block = self._fresh_block(name=then_name)
    else_block = self._fresh_block(name=else_name)
    after_else_block = self._fresh_block(name=continue_name)
    self._append_block(then_block, prev_terminator=inst.BranchOp(
        str(condition), then_block, else_block))
    yield
    # In case the enclosed code ended in a dangling if, close it
    self._prepare_for_instruction()
    # FIXME: Always adding this goto risks polluting the output with
    # excess gotos.  They can probably be cleaned up during label
    # resolution.
    self._append_block(else_block, prev_terminator=inst.GotoOp(
        after_else_block))
    self._pending_after_else_block = after_else_block

  @contextlib.contextmanager
  def else_(self, else_name=None, continue_name=None):
    """Records the `false` branch of a conditional operation.

    The `true` branch must be recorded (by `if_`, above) as the immediately
    preceding operation at the same nesting depth.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    ab.var.false = ab.const(False)
    with ab.if_(ab.var.false):
      ...
    with ab.else_():
      ...  # The body of the `with` statement gives the `false` branch
    ```

    Args:
      else_name: Optional Python string naming the false branch when the program
        is printed.  Overrides the `else_name`, if any, given in the
        corresponding `if_`.
      continue_name: Optional Python string naming the continuation after the if
        when the program is printed.  Overrides the `continue_name`, if any,
        given in the corresponding `if_`.

    Raises:
      ValueError: If not immediately preceded by an `if_`.

    Yields:
      Nothing.
    """
    if self._pending_after_else_block is None:
      msg = 'Detected else block with no preceding if'
      raise ValueError(msg)
    if else_name is not None:
      self._blocks[-1].name = else_name
    if continue_name is not None:
      self._pending_after_else_block.name = continue_name
    # Save the pending block in case the coming else block contains an if
    pending_block = self._pending_after_else_block
    self._pending_after_else_block = None
    yield
    # In case the enclosed code ended in a dangling if, close it
    self._prepare_for_instruction()
    # Now close this else block
    self._append_block(pending_block)

  def return_(self, vars_out):
    """Records a function return instruction.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    with ab.function(...) as f:
      ...
      ab.var.result = ...
      ab.return_(ab.var.result)
    ```

    A `return_` command must occur at the top level of the function definition
    (not inside any `if_`s), and must be the last statement therein.  You can
    always achieve this by assigning to a dedicated variable for the answer
    where you would otherwise return (and massaging your control flow).

    Args:
      vars_out: Pattern of Python strings giving the auto-batched variables to
        return.

    Raises:
      ValueError: If invoked more than once in a function body, or if trying to
        return variables that have not been written to.
    """
    # Assume the return_ call is at the top level, and the last statement in the
    # body.  If return_ is nested, the terminator may be overwritten
    # incorrectly.  If return_ is followed by something else, extra instructions
    # may get inserted before the return (becaue return_ doesn't set up a Block
    # to catch them).
    self._prepare_for_instruction()
    for var in inst.pattern_traverse(vars_out):
      if var not in self._var_defs:
        raise ValueError('Returning undefined variable {}.'.format(var))
    if self._functions[-1].vars_out:
      raise ValueError('Function body must have exactly one return_ statement')
    self._functions[-1].vars_out = inst.pattern_map(str, vars_out)
    self._blocks[-1].terminator = inst.halt_op()

  @contextlib.contextmanager
  def function(self, name=None, type_inference=None):
    """Registers a definition of an auto-batchable function.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    with ab.function(...) as f:
      ab.param('n')
      ...
      ab.return_(...)
    ```

    Note:
    - The `as` clause (here `f`) binds an `instructions.Function` object
      representing the function being defined (see Yields).
    - The formal parameters are given by calling `param` inside the
      `with` block.
    - The body of the `with` block registers the body of the function being
      defined.
    - The last statement registered in the `with` block must be a call to
      `return_`, or the `Function` will be malformed.

    The `function` method is a shorthand of `declare_function` followed by
    `define_function`.  The example is equivalent to:

    ```python
    ab = dsl.ProgramBuilder()

    f = ab.declare_function(...)
    with ab.define_function(f):
      ab.param('n')
      ...
      ab.return_(...)
    ```

    Args:
      name: Optional string naming this function when the program is printed.
      type_inference: A Python callable giving the type signature of the
        function being defined.  The callable will be invoked with a single
        argument giving the list of `instruction.Type` objects describing
        the arguments at a particular call site, and must return a list
        of `instruction.Type` objects describing the values that call site
        will return.

    Raises:
      ValueError: If invoked while defining a function, or if the function
        definition does not end in a `return_`.

    Yields:
      function: An `instructions.Function` object representing the function
        being defined.  It can be passed to `call` to call it (including
        recursively).  Note that Python scopes `as` bindings to the definition
        enclosing the `with`, so a function thus bound can be referred to after
        its body as well.
    """
    function = self.declare_function(name, type_inference)
    with self.define_function(function):
      yield function

  def declare_function(self, name=None, type_inference=None):
    """Forward-declares a function to be defined later with `define_function`.

    This useful for defining mutually recursive functions:
    ```python
    ab = dsl.ProgramBuilder()

    foo = ab.declare_function(...)

    with ab.function(...) as bar:
      ...
      ab.call(foo)

    with ab.define_function(foo):
      ...
      ab.call(bar)
    ```

    It is an error to call but never define a declared function.

    Args:
      name: Optional string naming this function when the program is printed.
      type_inference: A Python callable giving the type signature of the
        function being defined.  See `function`.

    Returns:
      function: An `instructions.Function` object representing the function
        being declared.  It can be passed to `call` to call it, and to
        `define_function` to define it.
    """
    return inst.Function(
        graph=None,
        vars_in=[],
        vars_out=[],
        type_inference=type_inference,
        name=name)

  @contextlib.contextmanager
  def define_function(self, function):
    """Registers a definition for a previously declared function.

    Usually, one would use the `function` method to declare and define a
    function at the same time.  Explicit use of `define_function` is only useful
    for mutual recursion or controlling code order separately from the call
    graph.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    foo = ab.declare_function(...)

    with ab.function(...) as bar:
      ...
      ab.call(foo)

    with ab.define_function(foo):
      ...
      ab.call(bar)
    ```

    Function bodies appear in the compiled `instructions.Program` in order of
    definition, not declaration.

    Note:
    - The formal parameters are given by calling `ab.param` inside the
      `with` block.
    - The body of the `with` block registers the body of the function being
      defined.
    - The last statement registered in the `with` block must be a `ab.return_`,
      or the `Function` will be malformed.

    Args:
      function: The function (from `declare_function`) to define.

    Yields:
      function: The `function` being defined, by symmetry with the
        `context.function` method.

    Raises:
      ValueError: If invoked while defining a function, if the `function`
        argument has already been defined, or if the function definition does
        not end in a `return_`.
    """
    if self._blocks is not None:
      raise ValueError('Nested function definitions not supported')
    msg = 'Internal invariant violation'
    assert self._locals is None, msg
    assert self._pending_after_else_block is None, msg
    if function.graph is not None:
      raise ValueError('Cannot redefine the function named {}.'.format(
          function.name))
    self._functions.append(function)
    if function.name is not None:
      block_name = 'enter_' + function.name
    else:
      block_name = 'entry'
    self._blocks = [self._fresh_block(name=block_name)]
    self._locals = {}
    yield function
    if self._blocks[-1].terminator is None:
      msg = 'Every function body must end in a return_.'
      raise ValueError(msg)
    function.graph = inst.ControlFlowGraph(self._blocks)
    self._blocks = None
    self._locals = None

  def call(self, function, vars_in, vars_out=None):
    """Registers a function call instruction.

    Example:
    ```
    ab = dsl.ProgramBuilder()

    # Define a function
    with ab.function(...) as func:
      ...
      # Call it (recursively)
      ab.var.thing = ab.call(func, ...)
      ...
    ```

    Args:
      function: The `instructions.Function` object representing the function to
        call.
      vars_in: Python strings giving the variables to pass in as inputs.
      vars_out: A pattern of Python strings, giving the auto-batched variable(s)
        to which to write the result of the call.  Defaults to the empty list.

    Raises:
      ValueError: If the call references undefined auto-batched variables.

    Returns:
      op: An `instructions.FunctionCallOp` representing the call.  If one
        subsequently assigns this to a local, via `ProgramBuilder.var.foo = op`,
        that local gets added to the list of output variables.
    """
    for var in vars_in:
      if var not in self._var_defs:
        raise ValueError('Referencing undefined variable {}.'.format(var))
    self._prepare_for_instruction()
    if vars_out is None:
      vars_out = []
    call = inst.FunctionCallOp(
        function, _str_list(vars_in), inst.pattern_map(str, vars_out))
    self._blocks[-1].instructions.append(call)
    for var in inst.pattern_traverse(vars_out):
      self._mark_defined(var)
    return call

  def module(self):
    """Returns the registered function definitions as an `instructions.Module`.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    with ab.function(...) as foo:
      ...  # Do stuff

    module = ab.module()
    ```

    Raises:
      ValueError: If invoked inside a function definition.

    Returns:
      module: The `instructions.Module` corresponding to all the definitions
        accumulated in this `ProgramBuilder`.
    """
    if self._blocks is not None:
      raise ValueError('Not finished defining function')
    msg = 'Internal invariant violation'
    assert self._locals is None, msg
    assert self._pending_after_else_block is None, msg
    var_defs = {str(var): inst.Type(None) for var in self._var_defs}
    return inst.Module(self._functions, var_defs)

  def program(self, main):
    """Returns the registered program as an `instructions.Program`.

    This is a helper method, equivalent to `self.module().program(main)`.

    Example:
    ```python
    ab = dsl.ProgramBuilder()

    with ab.function(...) as main:
      ...  # Do the stuff

    program = ab.program(main)
    ```

    Args:
      main: An `instructions.Function` object representing the main entry point.

    Raises:
      ValueError: If invoked inside a function definition, of if the intended
        `main` function was not defined.

    Returns:
      program: The `instructions.Program` corresponding to all the definitions
        accumulated in this `ProgramBuilder`.
    """
    return self.module().program(main)


class _MagicVars(object):
  """Magic object referring to the local variables in a function definition.

  Obtain with `ProgramBuilder.vars`.  Overrides `setattr` and `getattr` to
  provide a smooth interface to reading and defining variables:

  - `ProgramBuilder.var.foo = ProgramBuilder.{call,primop}`
    records an assignment to the auto-batched variable `foo`, possibly
    binding it, and

  - `ProgramBuilder.var.foo` reads from the auto-batched variable `foo`
    (if it is bound).
  """

  def __init__(self, context):
    super(_MagicVars, self).__setattr__('_context', context)

  def __setattr__(self, name, item):
    # _MagicVars is meant to be a friend class of ProgramBuilder; is purposely
    # a wrapper around the otherwise-private _locals field.
    # pylint: disable=protected-access
    if item is not None and item.vars_out:
      raise ValueError(
          'Outputs should be either declared explicitly '
          'or through `context.var`, but not both')
    context = self._context
    if name not in context._locals:
      # Updates context._locals
      context.local(name)
    variable = context._locals[name]
    if item is not None:
      context._update_last_instruction(item.replace(vars_out=str(variable)))
    context._mark_defined(variable)
    return variable

  def __getattr__(self, name):
    # _MagicVars is meant to be a friend class of ProgramBuilder; is purposely
    # a wrapper around the otherwise-private _locals field.
    # pylint: disable=protected-access
    context = self._context
    if name in context._locals:
      return context._locals[name]
    else:
      # Construct a name for the variable, but do not mark it defined.  If this
      # appears as an input to some reader, the reader will notice the variable
      # is undefined and signal an error.  But, if this appears as an input to
      # multi-value binding (see `_MagicPattern.pattern`), the binding will mark
      # it defined.
      return context.local(name, define=False)


class _MagicPattern(object):
  """Magic part of the protocol for multiple value returns.

  See `ProgramBuilder.__call__`.
  """

  def __init__(self, context, pattern):
    self._context = context
    self._pattern = pattern

  @property
  def pattern(self):
    pass

  @pattern.setter
  def pattern(self, item):
    # _MagicPattern is meant to be a friend class of ProgramBuilder.
    # pylint: disable=protected-access
    if item is not None:
      self._context._update_last_instruction(
          item.replace(vars_out=inst.pattern_map(str, self._pattern)))
    for var in inst.pattern_traverse(self._pattern):
      self._context._mark_defined(var)


def _str_list(lst):
  return [str(item) for item in lst]
