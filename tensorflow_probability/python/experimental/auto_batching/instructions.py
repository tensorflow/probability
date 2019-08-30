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
"""Instruction language for auto-batching virtual machine."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import inspect
import textwrap

# Dependency imports
from absl import logging
import numpy as np
import six
import tensorflow.compat.v2 as tf


__all__ = [
    'Program',
    'TensorType',
    'Type',
    'VariableAllocation',
    'ControlFlowGraph',
    'Block',
    'Function',
    'FunctionCallOp',
    'PrimOp',
    'PopOp',
    'BranchOp',
    'GotoOp',
    'PushGotoOp',
    'IndirectGotoOp',
    'pc_var',
    'push_op',
    'halt_op',
    'interpret',
    'extract_referenced_variables',
]


class Program(object):
  """An auto-batchable program.

  The primary ingredient of a Program is the control flow graph of
  operations to perform.  The operation language is a union that
  serves two purposes: one subset is designed to be convenient to run
  in Single Instruction Multiple Thread style, and the other to
  generate from an upstream Python-embedded DSL.

  As such, there are operations for explicit control transfers and
  stack management, as well as for interpreted function calls (pending
  lowering to explicit control transfers).  The primitive computations
  are encapsulated from the interpreter as Python functions.  It is
  not expected that one would author programs in this operation
  language directly, but rather generate them with an appropriate
  compiler.

  Lowering consists of eliminating `FunctionCallOp` in favor of a
  specific sequence of lower-level instructions.  A few choices for
  the lowered language may register as somewhat nonstandard:
  - The variable name space is global; the upstream compiler is
    expected to generate unique variable names.
  - There is no one stack; instead, every variable has its own stack.
    This reduces push traffic, because only the variables that are
    actually written to are ever pushed.
  - Return addresses for pending lowered function calls are stored in
    a reserved variable, that is otherwise in the same environment as
    the user variables.
  - There are no designated registers for function arguments or return
    values.  This is because all runtime storage is in Tensors, which
    need to have fixed types.  Instead, it is the (post-lowering)
    caller's responsibility to write the arguments into the formal
    parameters and to retrieve the returned value(s) from the
    variable(s) in which the callee left them.

  The post-lowering function call sequence is
  - Push the arguments to the formal parameters;
  - Pop any argument variables that are no longer used;
  - Store the desired return address and jump to the beginning of the function's
    body (with a single `PushGotoOp`);
  - When the function returns by executing `IndirectGotoOp`, assign the
    returned values to the variables that should receive them; and
  - Pop the variables holding the returned values.

  Note that this sequence requires that all calls in the source
  language be to statically known functions, and for every function to
  leave its results in the same variable(s) on every call (regardless
  of internal control flow).
  """

  def __init__(
      self, graph, functions, var_defs, vars_in, vars_out, var_alloc=None):
    """Initialize a new `Program`.

    Args:
      graph: A `ControlFlowGraph`.  This is the graph of basic blocks
        to execute.
      functions: A list of `Function`s giving the definitions of all
        the auto-batchable functions this `Program` may (recursively)
        call.
      var_defs: A dict mapping variable names to `Type` objects
        giving their pattern of held Tensors.  Each leaf of the pattern
        is a `TensorType` object giving the dtype and shape of that leaf.
        The shape excludes the batch dimension.
      vars_in: A list of the names of the variables in which to store
        the inputs when starting.
      vars_out: A pattern of the names of the variables from which to
        read the outputs when finished.
      var_alloc: A dict mapping variable names to allocation strategies (see
        `VariableAllocation`).  The semantics of an entry are "A proof has been
        found that this strategy suffices for this variable."
    """
    self.graph = graph
    self.functions = functions
    self.var_defs = var_defs
    self.vars_in = vars_in
    self.vars_out = vars_out
    if pc_var not in var_defs:
      var_defs[pc_var] = single_type(np.int32, ())
    if var_alloc is None:
      var_alloc = {name: VariableAllocation.FULL for name in var_defs.keys()}
    self.var_alloc = var_alloc

  def replace(self, var_defs=None, var_alloc=None):
    """Return a copy of `self` with `var_defs` and/or `var_alloc` replaced."""
    if var_defs is None and var_alloc is None:
      raise ValueError('Nothing to replace')
    return Program(
        self.graph, self.functions,
        var_defs if var_defs is not None else self.var_defs,
        self.vars_in, self.vars_out,
        var_alloc if var_alloc is not None else self.var_alloc)

  def main_function(self):
    """Return a representation of the main program as a `Function`."""
    return Function(self.graph, self.vars_in, self.vars_out, None, name='main')

  def __str__(self, indent=0, width=80):
    # A `Program` prints as
    #   var1 :: allocation strategy, dtype, shape=shape
    #   var2 :: allocation strategy, dtype, shape=shape
    #   ...
    #
    #   def function1
    #     ...
    #
    #   def function2
    #     ...
    #
    #   ...
    #
    #   def main
    #     ...
    #
    # In other words, the variable declarations in `self.var_defs` are printed
    # as a block first (one line per variable, without breaking), then the
    # `Function`s in the `Program`, separated by blank lines, and then the entry
    # graph, as a synthetic `Function` called "main".
    declarations = [
        ' ' * indent + '{} :: {}, {}'.format(
            var, self.var_alloc[var].name, self.var_defs[var])
        for var in sorted(self.var_defs)]
    functions = [func.__str__(indent, width) + '\n' for func in self.functions]
    main = self.main_function().__str__(indent, width)
    return '\n'.join(declarations + [''] + functions + [main])


# Some notes on the type representation:
# - TensorType describes the type (dtype and shape) of one Tensor subject to
#   auto-batching.  The shape does not include the batch dimension.
# - Type describes the type of one auto-batch variable.  Since a variable may
#   hold a potentially nested list or tuple of Tensors, the content of a Type is
#   the corresponding nested list or tuple (pattern) of TensorTypes.
# - Because of pattern matching, the left hand side of an assignment (either
#   PrimOp or FunctionCallOp) may be a pattern of variables.  The expected type
#   may therefore be a pattern of Types, each of which holds a pattern of
#   TensorTypes.
# - The type of the right hand side, however, is a pattern of TensorType only,
#   collapsing out any Type that may have been present at the return point of
#   the called function.  This way pattern matching works as expected regardless
#   of what pattern of tuple-holding variables the returned structure was
#   contained in.
class TensorType(collections.namedtuple('TensorType', ['dtype', 'shape'])):
  __slots__ = ()

  def __str__(self):
    return 'TensorType({}{})'.format(tf.as_dtype(self.dtype).name, self.shape)

  def __repr__(self):
    return 'TensorType({}{})'.format(tf.as_dtype(self.dtype).name, self.shape)

Type = collections.namedtuple('Type', ['tensors'])


def single_type(dtype, shape):
  return Type(TensorType(dtype, shape))


class Module(object):
  """A module of related auto-batchable functions.

  A `Module` stores multiple `Function`s that may call one another (possibly
  recursively).  A `Module` differs from a `Program` in that it has no
  pre-specified entry point.
  """

  def __init__(self, functions, var_defs):
    self.functions = functions
    self.var_defs = var_defs

  def lookup(self, func_name):
    for func in self.functions:
      if str(func.name) == str(func_name):
        return func
    return None

  def program(self, main):
    """Returns a `Program` corresponding to entering this `Module` at `main`."""
    if not isinstance(main, Function):
      main_name = main
      main = self.lookup(main_name)
      if main is None:
        raise ValueError('Function {} not found.'.format(main_name))
    call = FunctionCallOp(
        main, pattern_map(str, main.vars_in), pattern_map(str, main.vars_out))
    block = Block(instructions=[call], terminator=halt_op())
    cfg = ControlFlowGraph([block])
    return Program(
        cfg, self.functions, self.var_defs, main.vars_in, main.vars_out)


class VariableAllocation(object):
  """A token indicating how to allocate memory for an autobatched variable.

  In general, a variable holding data with a longer or more complex lifetime
  will need a more expensive storage strategy.

  Specifically, the four variable allocation strategies are:
  - `NULL`: Holds nothing.  Drops writes, raises on reads.  Useful for
    representing dummy variables that the user program never reads.
  - `TEMPORARY`: Holds one value per thread, but not across basic block
    boundaries.  Only usable for temporaries that live in a single basic block,
    and thus never experience joins (or vm execution loop crossings).  For such
    a variable, `push` just overwrites the whole Tensor; `pop` nulls the whole
    Tensor out.
  - `REGISTER`: Holds one value per thread, with no associated stack.  Useful
    for representing temporaries that do not cross (recursive) function calls,
    but do span multiple basic blocks.  For such a variable, `push` amounts to a
    `where`, with an optional runtime safety check for overwriting a defined
    value.
  - `FULL`: Holds a complete stack for each thread.  Used as a last resort, when
    a stack is unavoidable.

  The difference between `register` and `temporary` is that `register` is a
  `[batch_size] + event_shape` Tensor in the loop state of the toplevel
  `while_loop`, whereas `temporary` is represented as an empty tuple in the loop
  state, and only holds a Tensor during the execution of the
  `virtual_machine._run_block` call that uses it.  Consequently, `register`
  updating involves a `where`, but writing to a `temporary` produces 0 TF ops.
  Also, in the (envisioned) gather-scatter batching mode, the `temporary` Tensor
  will automatically only hold data for the live threads, whereas reading and
  writing a `register` will still require the gathers and scatters.
  """

  def __init__(self, name):
    self.name = name

  def __str__(self):
    return str(self.name)

  def __repr__(self):
    return 'VariableAllocation({})'.format(str(self.name))

VariableAllocation.NULL = VariableAllocation('null')
VariableAllocation.TEMPORARY = VariableAllocation('temporary')
VariableAllocation.REGISTER = VariableAllocation('register')
VariableAllocation.FULL = VariableAllocation('full')


class ControlFlowGraph(object):
  """A control flow graph (CFG)."""

  def __init__(self, blocks):
    """A control flow graph (CFG).

    A CFG is a set of basic `Block`s available in a program.  In this
    system, the `Block`s are ordered and indexed to support VM
    instruction selection, so the CFG also keeps the reverse map from
    `Block`s to their indexes.

    Args:
      blocks: Python list of `Block` objects, the content of the CFG.
        Any terminator instructions of said `Block` objects should
        refer to other `Block`s in the same CFG.  Otherwise,
        downstream passes or staging may fail.
    """
    self._blocks = blocks
    self._block_map = dict([(bl, i) for i, bl in enumerate(blocks)])

  @property
  def blocks(self):
    return self._blocks

  def block(self, index):
    """Returns the `Block` given by the input `index`.

    Args:
      index: A Python `int`.

    Returns:
      block: The `Block` at that location in the block list.
    """
    return self._blocks[index]

  def block_index(self, block):
    """Returns the `int` index of the given `Block`.

    Args:
      block: The block to look up. If `None`, returns the exit index.

    Returns:
      index: Python `int`, the index of the requested block.
    """
    if block is None:
      return self.exit_index()
    return self._block_map[block]

  def exit_index(self):
    """Returns the `int` index denoting "exit this CFG"."""
    return len(self._blocks)

  def enter_block(self):
    """Returns the entry `Block`."""
    return self._blocks[0]

  def __str__(self, indent=0, width=80):
    # A `ControlFlowGraph` prints as its sequence of `Block`s.
    # Lines are unconditionally broken between `Block`s.
    return '\n'.join([block.__str__(indent, width) for block in self._blocks])


class Block(object):
  """A basic block."""

  def __init__(self, instructions=None, terminator=None, name=None):
    """Initialize a `Block`.

    A basic block is a sequence of instructions that are always executed
    in order from first to last.  After the last instruction is executed,
    the block transfers control as indicated by the control transfer
    instruction in the `terminator` field.

    Pre-lowering, `FunctionCallOp` is admissible as an internal
    instruction in a `Block`, on the grounds that it returns to a fixed
    place, and the block is guaranteed to continue executing.

    Args:
      instructions: A list of `PrimOp`, `PopOp`, and `FunctionCallOp`
        instructions to execute in order.  Control transfer instructions (that
        do not return) are not permitted in this list.
      terminator: A single `BranchOp`, `GotoOp`, `PushGotoOp` or
        `IndirectGotoOp`, indicating how to transfer control out of this basic
        block.
      name: An object serving as the name of this `Block`, for display.
    """
    self.instructions = instructions
    self.terminator = terminator
    self.name = name

  def assign_instructions(self, instructions):
    """Assigns the body `instructions` and the `terminator` at once.

    This is a convenience method, to set a `Block`'s program content
    in one invocation instead of having to assign the `instructions`
    and the `terminator` fields separately.

    Args:
      instructions: A non-empty Python list of `Op` objects.  The last one must
        be a `BranchOp`, `GotoOp`, `PushGotoOp`, or `IndirectGotoOp`, and
        becomes the `terminator`.  The others, if any, must be `PrimOp`,
        `PopOp`, or `FunctionCallOp`, and become the `instructions`, in order.
    """
    self.terminator = instructions[-1]
    self.instructions = instructions[:-1]

  @property
  def label_str(self):
    """A string suitable for referring to this `Block` in printed output."""
    if self.name is not None:
      return str(self.name)
    else:
      return 'block-' + str(hash(self))

  def _string_for_instruction(self, inst, indent, width):
    try:
      return inst.__str__(indent=indent, width=width)
    except TypeError:
      # TypeError is what Python will throw if the __str__ method of the
      # inst object doesn't take the arguments I want to pass
      return ' ' * (indent + 2) + 'Malformed instruction {}'.format(inst)

  def __str__(self, indent=0, width=80):
    # A `Block` prints as
    #   label:
    #     op1
    #     op2
    #     ...
    #     terminator
    # There is always a line break between successive instructions, even if
    # several would otherwise fit on one line.  The instructions are also always
    # indented relative to the `Block` itself.
    label_line = ' ' * indent + self.label_str + ':'
    if self.instructions is not None:
      inst_lines = [self._string_for_instruction(inst, indent + 2, width)
                    for inst in self.instructions]
    else:
      inst_lines = []
    term_line = self._string_for_instruction(self.terminator, indent + 2, width)
    return '\n'.join([label_line] + inst_lines + [term_line])


class Function(object):
  """A function subject to auto-batching, callable with `FunctionCallOp`."""

  def __init__(self, graph, vars_in, vars_out, type_inference, name=None):
    """A `Function` is a control flow graph with input and output variables.

    Args:
      graph: A `ControlFlowGraph` comprising the function's body.
      vars_in: List of `string` giving the names of the formal parameters
        of the function.
      vars_out: Pattern of `string` giving the name(s) of the variables
        the function returns.  Ergo, functions must be canonicalized to
        place the return value(s) in the same-named variable(s) along
        every path to the exit.
      type_inference: A callable which takes a list of patterns of `TensorType`s
        corresponding to the data types of `vars_in`.  This callable must
        return a pattern of `TensorType`s corresponding to the structure
        assembled by the `return_vars`.
      name: Optional string denoting this `Function` in printed output.
    """
    self.graph = graph
    self.vars_in = vars_in
    self.vars_out = vars_out
    self.type_inference = type_inference
    self._name = name

  @property
  def name(self):
    if self._name is not None:
      return str(self._name)
    elif self.graph is not None:
      return self.graph.block(0).label_str
    else:
      return None

  def __str__(self, indent=0, width=80):
    # A `Function` prints as
    #   def name(param1, param2, ...):
    #     returns out1, out2, ...
    #     <block1>
    #     <block2>
    #     ...
    # The constituent blocks are printed in the order they appear in the control
    # flow graph, indented by two spaces relative to the `Function` itself.  If
    # the header or returns lines are too long, the print will first break the
    # line at the open paren and the returns token, respectively, and then at
    # the commas as needed.
    prefix = 'def ' + str(self.name) + '('
    header = _render_comma_list(prefix, self.vars_in, '):', indent, width)
    returns = [' ' * (indent + 2) + 'returns ' +
               _render_pattern(self.vars_out)]
    graph = self.graph.__str__(indent=indent+2, width=width)
    return '\n'.join(header + returns + [graph])


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class FunctionCallOp(collections.namedtuple(
    'FunctionCallOp', ['function', 'vars_in', 'vars_out'])):
  """Call a `Function`.

  This is a higher-level instruction, what in LLVM jargon is called an
  "intrinsic".  An upstream compiler may construct such instructions;
  there is a pass that lowers these to sequences of instructions the
  downstream VM can stage directly.

  This differs from `PrimOp` in that the function being called is
  itself implemented in this instruction language, and is subject to
  auto-batching by the downstream VM.

  A `FunctionCallOp` is required to statically know the identity of the
  `Function` being called.  This is because we want to copy the return
  values to their destinations at the caller side of the return
  sequence.  Why do we want that?  Because threads may diverge at
  function returns, thus needing to write the returned values to
  different caller variables.  Doing that on the callee side would
  require per-thread information about where to write the variables,
  which, in this design, is encoded in the program counter stack.
  Why, in turn, may threads diverge at function returns?  Because part
  of the point is to allow them to converge when calling the same
  function, even if from different points.

  Args:
    function: A `Function` object describing the function to call.
      This requires all call targets to be known statically.
    vars_in: list of strings.  The names of the VM variables whose
      current values to pass to the `function`.
    vars_out: pattern of strings.  The names of the VM variables
      where to save the results returned from `function`.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # A `FunctionCallOp` prints as
    #   out1, out2, ... = call(function-name, in1, in2, ...)
    # If that's is too long, the print breaks the line at the open paren first,
    # and then at whichever of the commas in the argument list are needed.  The
    # list of returned values is never broken across lines.
    if self.vars_out:
      prefix = _render_pattern(self.vars_out) + ' = call('
    else:
      prefix = 'call('
    tokens = [str(self.function.name)] + self.vars_in
    lines = _render_comma_list(prefix, tokens, ')', indent, width)
    return '\n'.join(lines)

  def replace(self, vars_out=None):
    """Return a copy of `self` with `vars_out` replaced."""
    if vars_out is None:
      raise ValueError('Nothing to replace')
    return FunctionCallOp(self.function, self.vars_in, vars_out)


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class PrimOp(collections.namedtuple(
    'PrimOp', ['vars_in', 'vars_out', 'function', 'skip_push_mask'])):
  """An arbitrary already-batched computation, a 'primitive operation'.

  These are the items of work on which auto-batching is applied.  The
  `function` must accept and produce Tensors with a batch dimension,
  and is free to stage any (batched) computation it wants.
  Restriction: the `function` must use the same computation substrate
  as the VM backend.  That is, if the VM is staging to XLA, the
  `function` will see XLA Tensor handles; if the VM is staging to
  graph-mode TensorFlow, the `function` will see TensorFlow Tensors;
  etc.

  The current values of the `vars_out` are saved on their respective
  stacks, and the results written to the new top.

  The exact contract for `function` is as follows:
  - It will be invoked with a list of positional (only) arguments,
    parallel to `vars_in`.
  - Each argument will be a pattern of Tensors (meaning, either one
    Tensor or a (potentially nested) list or tuple of Tensors),
    corresponding to the `Type` of that variable.
  - Each Tensor in the argument will have the `dtype` and `shape`
    given in the corresponding `TensorType`, and an additional leading
    batch dimension.
  - Some indices in the batch dimension may contain junk data, if the
    corresponding threads are not executing this instruction [this is
    subject to change based on the batch execution strategy].
  - The `function` must return a pattern of Tensors, or objects
    convertible to Tensors.
  - The returned pattern must be compatible with the `Type`s of
    `vars_out`.
  - The Tensors in the returned pattern must have `dtype` and `shape`
    compatible with the corresponding `TensorType`s of `vars_out`.
  - The returned Tensors will be broadcast into their respective
    positions if necessary.  The broadcasting _includes the batch
    dimension_: Thus, a returned Tensor of insufficient rank (e.g., a
    constant) will be broadcast across batch members.  In particular,
    a Tensor that carries the indended batch size but whose sub-batch
    shape is too low rank will broadcast incorrectly, and will result
    in an error.
  - If the `function` raises an exception, it will propagate and abort
    the entire computation.
  - Even in the TensorFlow backend, the `function` will be staged
    several times: at least twice during type inference (to ascertain
    the shapes of the Tensors it likes to return, as a function of the
    shapes of the Tensors it is given), and exactly once during
    executable graph construction.

  Args:
    vars_in: list of strings.  The names of the VM variables whose
      current values to pass to the `function`.
    vars_out: Pattern of strings.  The names of the VM variables
      where to save the results returned from `function`.
    function: Python callable implementing the computation.
    skip_push_mask: Set of strings, a subset of `vars_out`.  These VM variables
      will be updated in place rather than pushed.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # A `PrimOp` prints as
    #   out1, out2, ... = prim(in1, in2, ...) by
    #     <source code of the function>
    # If the header is too long, the print breaks the line at the open paren
    # first, and then at whichever of the commas in the argument list are
    # needed.  Neither the function source nor the list of returned
    # values are ever broken across lines.
    if self.vars_out:
      def name(v):
        return v + ('!' if v in self.skip_push_mask else '')
      vars_written = pattern_map(name, self.vars_out)
      prefix = _render_pattern(vars_written) + ' = prim('
    else:
      prefix = 'prim('
    header = _render_comma_list(prefix, self.vars_in, ') by', indent, width)
    code = _get_source_str(self.function)
    code_lines = [' ' * (indent + 2) + l for l in code.split('\n')]
    return '\n'.join(header + code_lines)

  def replace(self, vars_out=None):
    """Return a copy of `self` with `vars_out` replaced."""
    if vars_out is None:
      raise ValueError('Nothing to replace')
    return PrimOp(self.vars_in, vars_out, self.function, self.skip_push_mask)


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class PopOp(collections.namedtuple('PopOp', ['vars'])):
  """Restore the given variables from their stacks.

  The current top value of each popped variable is lost.

  Args:
    vars: list of strings: The names of the VM variables to restore.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # A `PopOp` prints as "pop(var1, var2, ...)".  If that's too long, the print
    # breaks the line at the open paren first, and then at whichever of the
    # commas are needed.
    prefix = 'pop('
    lines = _render_comma_list(prefix, self.vars, ')', indent, width)
    return '\n'.join(lines)


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class BranchOp(collections.namedtuple(
    'BranchOp', ['cond_var', 'true_block', 'false_block'])):
  """A conditional.

  Args:
    cond_var: The string name of the VM variable holding the condition.
      This variable must have boolean dtype and scalar data shape.
    true_block: The `Block` where to transfer logical threads whose
      condition is `True`.
    false_block: The `Block` where to transfer logical threads whose
      condition is `False`.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # A `BranchOp` prints as "branch(condition, true-label, false-label)".  If
    # that's too long, the print breaks the line at the open paren first, and
    # then at whichever of the commas are needed.
    prefix = 'branch('
    tokens = [
        self.cond_var, _label_str(self.true_block),
        _label_str(self.false_block)]
    lines = _render_comma_list(prefix, tokens, ')', indent, width)
    return '\n'.join(lines)


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class GotoOp(collections.namedtuple('GotoOp', ['block'])):
  """An unconditional jump.

  Use for skipping non-taken `if` branches, and for transferring
  control during the function call sequence.

  Args:
    block: The `Block` to jump to.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # A `GotoOp` that terminates execution of the current control flow graph
    # prints as `return`, since that's what the definitional interpreter does in
    # that circumstance.
    # Any other `GotoOp` prints as "goto(block-label)", breaking the line at the
    # open paren if needed.
    if self.block is None:
      return ' ' * indent + 'return'
    else:
      prefix = 'goto('
      lines = _render_comma_list(
          prefix, [_label_str(self.block)], ')', indent, width)
      return '\n'.join(lines)


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class PushGotoOp(collections.namedtuple(
    'PushGotoOp', ['push_block', 'goto_block'])):
  """Save an address for `IndirectGotoOp` and unconditionally jump to another.

  Use in the function call sequence.  The address is saved on the
  reserved "program counter" variable.

  Args:
    push_block: The `Block` that the matching `IndirectGotoOp` will jump to.
    goto_block: The `Block` to jump to.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # A `PushGotoOp` prints as "push_goto(push-label, goto-label)", breaking the
    # line at the open paren if needed.
    prefix = 'push_goto('
    labels = [_label_str(self.push_block), _label_str(self.goto_block)]
    lines = _render_comma_list(prefix, labels, ')', indent, width)
    return '\n'.join(lines)


# Inheriting from `namedtuple` as a compact way to make a value class:
# structural equality and hashing.
class IndirectGotoOp(collections.namedtuple('IndirectGotoOp', [])):
  """Jump to the address in the reserved "program counter" variable.

  Use to return from a function call.

  Also restores the previous saved program counter (under the one
  jumped to), enforcing proper nesting with invocations of
  `PushGotoOp`.
  """
  __slots__ = ()

  def __str__(self, indent=0, width=80):
    # An `IndirectGotoOp` prints as "indirect_jump", which is a more precise and
    # less committed term than "return".
    return ' ' * indent + 'indirect_jump'


pc_var = '__program_counter__'


def prim_op(vars_in, vars_out, function, skip_push_mask=None):
  if skip_push_mask is None:
    skip_push_mask = set()
  return PrimOp(vars_in, vars_out, function, skip_push_mask)


def push_op(vars_in, vars_out):
  """Returns an `Op` that pushes values from `vars_in` into `vars_out`.

  Args:
    vars_in: Python pattern of `string`, the variables to read.
    vars_out: Python pattern of `string`, matching with `vars_in`; the
      variables to write to.

  Returns:
    op: An `Op` that accomplishes the push.
  """
  def assign(*x):
    return x

  return prim_op(vars_in, vars_out, assign)


def halt():
  """Returns `None`, acting as a sentinel `Block` meaning "exit this graph"."""
  return None


def halt_op():
  """Returns a control transfer `Op` that means "exit this graph"."""
  return GotoOp(halt())


def is_return_op(op):
  halt_op_ = isinstance(op, GotoOp) and op.block is None
  return_op = isinstance(op, IndirectGotoOp)
  return halt_op_ or return_op


class NullVariable(collections.namedtuple('NullVariable', [])):
  """A Variable that contains no storage and drops writes.

  Efficiently represents user variables that are never read.

  This is a namedtuple so it can be stored in an environment
  that passes through TensorFlow `while_loop`.
  """

  def update(self, value, mask):
    del value
    del mask
    return self

  def push(self, mask):
    del mask
    return self

  def read(self):
    raise NotImplementedError

  def pop(self, mask):
    del mask
    return self


class TemporaryVariable(collections.namedtuple('TemporaryVariable', ['value'])):
  """A temporary Variable.

  Efficiently stores values whose lifetime is inside one basic block.
  """

  @staticmethod
  def empty():
    # The sentinel value for a TemporaryVariable is not `None`, because the
    # `nest` utilities in TensorFlow choke on `None`.  Need an empty tuple (or
    # list, or dictionary) instead.
    return TemporaryVariable(())

  @property
  def _empty(self):
    # No, pylint, implicit bool comparison doesn't work here.  The reason is
    # that `value` is either a Tensor or an empty tuple, and Tensors
    # (reasonably) do not permit implicit bool comparisons.
    # pylint: disable=g-explicit-bool-comparison
    return isinstance(self.value, tuple) and self.value == ()

  def update(self, value, mask):
    del mask
    return TemporaryVariable(value)

  def push(self, mask):
    del mask
    if not self._empty:
      raise ValueError('Pushing a non-empty temporary variable.')
    return self

  def read(self):
    if self._empty:
      raise ValueError('Reading from empty temporary variable.')
    return self.value

  def pop(self, mask):
    del mask
    if self._empty:
      raise ValueError('Popping from empty temporary variable.')
    return TemporaryVariable(())


class PythonVMVariable(object):
  """A pure-python implementation of the interface for VM variables.

  This `PythonVMVariable` class used only by the semantics-defining interpreter
  `interpret`, and as such is meant to be simple rather than high-performance.
  In particular, each variable is given exactly one value in each of its stack
  levels---there is no support for batching.

  To wit, the actual variable is a Python list giving the stack of values of
  that variable.  Reads and writes interact with the -1-indexed element.  Note
  that this architecture supports different variables having different stack
  heights, like the staging VM of which this is a model.

  The methods are all written consistently with functional style to support the
  `Environment` being written in functional style itself.
  """

  def __init__(self, name):
    self._name = name
    self._stack = [None]

  def push(self):
    self._stack.append(self._stack[-1])
    return self

  def update(self, value):
    self._stack[-1] = value
    return self

  def pop(self):
    self._stack.pop()
    if not self._stack:
      raise ValueError('Popped last value off variable {}.'.format(self._name))
    return self

  def read(self):
    if not self._stack:
      raise ValueError('Reading empty variable {}.'.format(self._name))
    return self._stack[-1]

  def __repr__(self):
    return 'PythonVariable({})'.format(self._stack)


class PythonBackend(object):
  """A factory for `PythonVMVariable` with the interface of a VM backend.

  Used in the definitional interpreter to form a simple model of non-batched
  computations.
  """

  def create_variable(self, name, alloc, type_):
    del alloc
    # This backend ignores the variable types, since there are no Tensors in the
    # definitional interpreter
    del type_
    return PythonVMVariable(name)

  @property
  def variable_class(self):
    return PythonVMVariable


class Environment(object):
  """An environment, giving values for the extant variables.

  The actual environment is a dictionary mapping each variable name to the
  backend-determined `Variable` object holding the stack of values of that
  variable.  Reads and writes pass through to the `Variable`s; this class
  just manages name resolution.

  The methods are all written in functional style to support using the same
  `Environment` to stage multiple different instructions.
  """

  def __init__(self, env_dict, backend, update=None):
    """Creates an `Environment` from the dictionary giving its content.

    Defensively copies the input dictionary.

    Args:
      env_dict: A dictionary mapping variable names to their values;
        e.g., from the `env_dict` method of a (different) `Environment`
        instance.
      backend: A factory for `Variable`s supporting a `create_variable(name,
        type)` method and a `variable_class` field.
      update: An optional dictionary mapping (some) variable names
        to their values.  If supplied, those variables' values are
        overwritten.
    """
    self.backend = backend
    self._env = dict(env_dict)
    if update is not None:
      self._env.update(update)

  @staticmethod
  def initialize(backend, var_alloc, var_defs, *args, **kwargs):
    """Initializes an `Environment` from type definitions.

    Args:
      backend: A factory for `Variable`s supporting a `create_variable(name,
        type)` method and a `variable_class` field.
      var_alloc: A Python dict mapping variable to `VariableAllocation`
        objects.
      var_defs: A Python dict mapping variable names to `Type`
        objects, as the `var_defs` field of `Program`.
      *args: Additional arguments to pass to the backend's `create_variable`
        method, if any.
      **kwargs: Additional keyword arguments to pass to the backend's
        `create_variable` method, if any.

    Returns:
      env: A properly initialized `Environment`.
    """
    alloc_counts = collections.defaultdict(int)

    def mk_tensor(name, type_):
      alloc_counts[var_alloc[name]] += 1
      return backend.create_variable(
          name, var_alloc[name], type_, *args, **kwargs)
    var_dict = {
        name: pattern_map(functools.partial(mk_tensor, name),
                          type_.tensors, leaf_type=TensorType)
        for name, type_ in six.iteritems(var_defs)}
    logging.debug(
        'Initialized environment: %d leaf variables, by allocation %s',
        sum(alloc_counts.values()), dict(alloc_counts))
    return Environment(var_dict, backend)

  def push(self, name, value, *args, **kwargs):
    """Pushes value to the variable of the given name.

    Does not mutate the `Environment`.

    Args:
      name: A Python `string` giving the name of the
        variable to push to.
      value: A object giving the value to push.
      *args: Additional arguments to pass to the `Variable`'s
        methods, if any.
      **kwargs: Additional keyword arguments to pass to the `Variable`'s
        methods, if any.

    Raises:
      ValueError: If `name` refers to a variable that was not part of
        the `var_defs` with which this Environment was created.

    Returns:
      new_var: An updated `Variable` incorporating the push.
    """
    # Could check dtypes and shapes here.  However, since they are only needed
    # for preallocating Tensor stacks, leaving the check off generalizes the
    # resulting interpreter.  Which may not be quite the right thing to do,
    # since the generalization may make this a worse model of the VM's real
    # behavior.
    if name not in self._env:
      raise ValueError('Pushing undeclared variable {}.'.format(name))
    def do_push(var, val):
      return var.push(*args, **kwargs).update(val, *args, **kwargs)
    return pattern_map2(
        do_push, self._env[name], value, leaf_type=self.backend.variable_class)

  def update(self, name, value, *args, **kwargs):
    """Writes value to the variable of the given name without pushing.

    Does not mutate the `Environment`.

    Args:
      name: A Python `string` giving the name of the
        variable to update.
      value: A object giving the value to update with.
      *args: Additional arguments to pass to the `Variable`'s
        methods, if any.
      **kwargs: Additional keyword arguments to pass to the `Variable`'s
        methods, if any.

    Raises:
      ValueError: If `name` refers to a variable that was not part of
        the `var_defs` with which this Environment was created.

    Returns:
      new_var: An updated `Variable` incorporating the update.
    """
    # Could check dtypes and shapes here.  However, since they are only needed
    # for preallocating Tensor stacks, leaving the check off generalizes the
    # resulting interpreter.  Which may not be quite the right thing to do,
    # since the generalization may make this a worse model of the VM's real
    # behavior.
    if name not in self._env:
      raise ValueError('Pushing undeclared variable {}.'.format(name))
    def do_update(var, val):
      return var.update(val, *args, **kwargs)
    return pattern_map2(
        do_update, self._env[name], value,
        leaf_type=self.backend.variable_class)

  def pop(self, name, *args, **kwargs):
    """Pops a value off the stack of the given variable.

    Does not mutate the `Environment`.

    Args:
      name: A Python `string` giving the name of the variable to pop.
      *args: Additional arguments to pass to the `Variable`'s
        `pop` method, if any.
      **kwargs: Additional keyword arguments to pass to the `Variable`'s
        `pop` method, if any.

    Raises:
      ValueError: If `name` refers to a variable that was not part of
        the `var_defs` with which this Environment was created.

    Returns:
      new_var: An updated `Variable` incorporating the pop.
    """
    def do_pop(var):
      return var.pop(*args, **kwargs)
    return pattern_map(
        do_pop, self._env[name], leaf_type=self.backend.variable_class)

  def read(self, name, *args, **kwargs):
    """Reads the current (top) value of the given variable.

    Args:
      name: A Python `string`, naming the variable to read.
      *args: Additional arguments to pass to the `Variable`'s
        `read` method, if any.
      **kwargs: Additional keyword arguments to pass to the `Variable`'s
        `read` method, if any.

    Returns:
      value: The value of that variable.

    Raises:
      ValueError: If `name` refers to a variable that was not part of
        the `var_defs` with which this Environment was created, or if
        the read stack is empty.
    """
    if name not in self._env:
      raise ValueError('Reading undeclared variable {}.'.format(name))
    def do_read(var):
      return var.read(*args, **kwargs)
    return pattern_map(
        do_read, self._env[name], leaf_type=self.backend.variable_class)

  def __setitem__(self, key, val):
    self._env[key] = val

  @property
  def env_dict(self):
    return self._env


def interpret(program, *inputs):
  """Interprets a program in this instruction language and returns the result.

  This is a definitional interpreter; its purpose is to define the
  semantics of the instruction language.  As such, it does no
  auto-batching, and generally strives to be as simple as possible.
  It also does not stage graph computations, so will only work in
  Eager mode TensorFlow.

  Args:
    program: The Program tuple to interpret.
    *inputs: Values to pass to the program.  The length of `inputs` must be
      the same as the length of `program.vars_in`.

  Returns:
    results: A tuple of results, which are the values of the variables listed
      in `program.out_vars` at termination.

  Raises:
    ValueError: If an internal invariant is violated, or an error is
      detected in the program being interpreted.
  """

  env = Environment.initialize(
      PythonBackend(), program.var_alloc, program.var_defs)
  for var, inp in pattern_zip(program.vars_in, inputs):
    env[var] = env.push(var, inp)
  next_block = program.graph.enter_block()
  while next_block is not None:
    block = next_block
    for op in block.instructions:
      if isinstance(op, PrimOp):
        if pc_var in pattern_flatten(op.vars_in):
          raise ValueError('Detected block reading program counter.')
        if pc_var in pattern_flatten(op.vars_out):
          raise ValueError('Detected block writing program counter.')
        ins = pattern_map(env.read, op.vars_in)
        outs = op.function(*ins)
        for var, out in pattern_zip(op.vars_out, outs):
          env[var] = env.push(var, out)
      elif isinstance(op, FunctionCallOp):
        if pc_var in pattern_flatten(op.vars_in):
          raise ValueError('Detected function call reading program counter.')
        if pc_var in pattern_flatten(op.vars_out):
          raise ValueError('Detected function call writing program counter.')
        ins = pattern_map(env.read, op.vars_in)
        outs = _invoke_fun(program, op.function, ins)
        for var, out in pattern_zip(op.vars_out, outs):
          env[var] = env.push(var, out)
      elif isinstance(op, PopOp):
        for var in op.vars:
          env[var] = env.pop(var)
      else:
        raise ValueError('Invalid instruction in block {}.'.format(op))
    op = block.terminator
    if isinstance(op, BranchOp):
      if pc_var == op.cond_var:
        raise ValueError('Detected branching on program counter.')
      cond = env.read(op.cond_var)
      if cond is None:
        raise ValueError('Branching on an unset variable.')
      if cond:
        next_block = op.true_block
      else:
        next_block = op.false_block
    elif isinstance(op, GotoOp):
      next_block = op.block
    elif isinstance(op, PushGotoOp):
      env[pc_var] = env.push(pc_var, op.push_block)
      next_block = op.goto_block
    elif isinstance(op, IndirectGotoOp):
      next_block = env.read(pc_var)
      env[pc_var] = env.pop(pc_var)
    else:
      raise ValueError('Invalid terminator instruction {}.'.format(op))
  return pattern_map(env.read, program.vars_out)


def _invoke_fun(program, function, inputs):
  program_for_function = Program(
      function.graph, program.functions,
      program.var_defs, function.vars_in, function.vars_out,
      program.var_alloc)
  return interpret(program_for_function, *inputs)


def extract_referenced_variables(node):
  """Extracts a set of the variable names referenced by the node in question.

  Args:
    node: Most structures from the VM, including ops, sequences of ops, blocks.

  Returns:
    varnames: `set` of variable names referenced by the node in question.
  """
  if isinstance(node, PrimOp):
    return set(pattern_flatten(node.vars_in) + pattern_flatten(node.vars_out))
  if isinstance(node, PopOp):
    return set(pattern_flatten(node.vars))
  if isinstance(node, Block):
    return (extract_referenced_variables(node.instructions) |
            extract_referenced_variables(node.terminator))
  if isinstance(node, BranchOp):
    return {node.cond_var}
  if isinstance(node, (GotoOp, PushGotoOp, IndirectGotoOp)):
    return {pc_var}
  if isinstance(node, ControlFlowGraph):
    return extract_referenced_variables(
        [node.block(i) for i in range(node.exit_index())])
  if isinstance(node, FunctionCallOp):
    return (set(pattern_flatten(node.vars_in)) |
            set(pattern_flatten(node.vars_out)))
  # Must come last to avoid matching NamedTuple instances.
  if isinstance(node, (list, tuple)):  # Lists of ops, blocks, etc.
    result = set()
    for x in node:
      result |= extract_referenced_variables(x)
    return result
  raise ValueError('Unexpected node type {}'.format(type(node)))


### Pattern-matching support

# A pattern is either
# - A list or tuple of patterns, or
# - A terminal object


def pattern_map(f, pattern, leaf_type=()):
  """Applies a function elementwise to a pattern.

  Args:
    f: Function to apply
    pattern: Objects
    leaf_type: Optional list of Python types to treat as leaves even though
      they may inherit from `list` or `tuple`.

  Returns:
    result: A structure of the same shape as objects, with every
      item `x` replaced by `f(x)`
  """
  if isinstance(pattern, leaf_type):
    return f(pattern)
  if isinstance(pattern, (list, tuple)):
    subs = [pattern_map(f, sub, leaf_type=leaf_type) for sub in pattern]
    return _sequence_like(pattern, subs)
  else:
    return f(pattern)


def pattern_traverse(pattern, leaf_type=()):
  """Yields every terminal object in a pattern."""
  if isinstance(pattern, (tuple, list)) and not isinstance(pattern, leaf_type):
    for sub in pattern:
      for item in pattern_traverse(sub, leaf_type=leaf_type):
        yield item
  else:
    yield pattern


def pattern_zip(pattern1, pattern2, leaf_type=()):
  """Yields every corresponding pair of objects in the given patterns.

  Args:
    pattern1: A pattern.
    pattern2: A pattern of matching shape.  Pattern2 may have lists or
      tuples where pattern1 has terminals; those pairs will be yielded.
    leaf_type: Optional list of Python types to treat as leaves even though
      they may inherit from `list` or `tuple`.

  Yields:
    item1: A terminal item from pattern1
    item2: The (potentially terminal) sub-pattern from pattern2 in the
      corresponding place.

  Raises:
    ValueError: If the patterns do not match.
  """
  if (isinstance(pattern1, (list, tuple))
      and not isinstance(pattern1, leaf_type)):
    if (isinstance(pattern2, (list, tuple))
        and not isinstance(pattern2, leaf_type)):
      if len(pattern1) != len(pattern2):
        raise ValueError(
            'Pattern size mismatch: expected {} items, got {}.'.format(
                len(pattern1), len(pattern2)))
      for sub1, sub2 in zip(pattern1, pattern2):
        for item1, item2 in pattern_zip(sub1, sub2, leaf_type=leaf_type):
          yield item1, item2
    else:
      raise ValueError(
          'Pattern shape mismatch; expected list or tuple, got {}.'.format(
              pattern2))
  else:
    yield pattern1, pattern2


def pattern_map2(f, pattern1, pattern2, leaf_type=()):
  """Applies f to every corresponding pair of objects in the given patterns.

  Args:
    f: Binary function to map.
    pattern1: A pattern.
    pattern2: A pattern of matching shape.  Pattern2 may have lists or
      tuples where pattern1 has terminals; those pairs will be matched.
    leaf_type: Optional list of Python types to treat as leaves even though
      they may inherit from `list` or `tuple`.

  Returns:
    results: A pattern of the same shape as pattern1.  Each terminal is the
      result of applying `f` to the corresponding pair of items from pattern1
      and pattern2.

  Raises:
    ValueError: If the patterns do not match.
  """
  if (isinstance(pattern1, (list, tuple))
      and not isinstance(pattern1, leaf_type)):
    if (isinstance(pattern2, (list, tuple))
        and not isinstance(pattern2, leaf_type)):
      if len(pattern1) != len(pattern2):
        raise ValueError(
            'Pattern size mismatch: expected {} items, got {}.'.format(
                len(pattern1), len(pattern2)))
      subs = [pattern_map2(f, sub1, sub2, leaf_type=leaf_type)
              for sub1, sub2 in zip(pattern1, pattern2)]
      return _sequence_like(pattern1, subs)
    else:
      raise ValueError(
          'Pattern shape mismatch; expected list or tuple, got {}.'.format(
              pattern2))
  else:
    return f(pattern1, pattern2)


def pattern_flatten(pattern, leaf_type=()):
  """Returns all the terminals in `pattern` as a list."""
  return list(pattern_traverse(pattern, leaf_type=leaf_type))


def _isnamedtuple(f):
  """Returns True if the argument is a namedtuple-like."""
  if tuple not in inspect.getmro(type(f)):
    return False
  if not hasattr(f, '_fields'):
    return False
  fields = getattr(f, '_fields')
  if not isinstance(fields, tuple):
    return False
  if not all(isinstance(f, str) for f in fields):
    return False
  return True


def _sequence_like(thing, elts):
  # Apparently, checking whether a Python object is a namedtuple (as distinct
  # from a regular tuple) is surprisingly difficult.
  if _isnamedtuple(thing):
    return type(thing)(*elts)
  else:
    return type(thing)(elts)


### Pretty-printing support


def _label_str(block):
  """Returns a string by which the given block may be named, even if `None`."""
  if block is None:
    return 'exit'
  else:
    return block.label_str


def _get_source_str(func):
  """Returns a string representing the code of `func`.  Doesn't fail."""
  try:
    # The astute reader may note that this will get everything on the source
    # lines where the function was defined, including tokens outside the actual
    # function definition (e.g., for a lambda expression).  Unfortunately,
    # Python does not maintain character-level source information, so getting
    # this precisely right is nontrivial:
    # http://xion.io/post/code/python-get-lambda-code.html
    return textwrap.dedent(inspect.getsource(func)).rstrip()
  except (IOError, TypeError):
    # IOError happens if inspect.getsource can't find the source file.
    # TypeError happens if it's a callable that is not a module, class, method,
    # function, traceback, frame, or code object (e.g., functools.partial)
    return str(func).strip()


def _render_comma_list(prefix, items, suffix, indent, width):
  """Renders the given items as a list of strings bounded by the given width.

  The result has the same non-whitespace content as
    prefix + ', '.join(items) + suffix.

  However, every line begins with at least `indent` spaces, and line breaks are
  added to try to make no line exceed the given `width`.

  Rules:
  - The `prefix` is never broken across lines.
  - If a line break is necessary, one occurs immediately after the prefix,
    and the subsequent lines are indented two additional spaces.
  - Individual `items` (and the `suffix`) may be broken across lines.
  - Line breaks are only inserted instead of whitespace.

  Args:
    prefix: Python string.  The output begins with this.  Not subject to
      wrapping.
    items: Python list of strings.  The output will contain these, in order,
      separated by commas and whitespace.  If not everything fits in one line, a
      line break is inserted between the prefix and the first item, and
      subsequently on spaces within or between items as needed.
    suffix: Python string.  Appended to the `items` without a comma.  Lines
      may be broken at spaces inside `suffix` if needed.
    indent: Python int.  The whole returned paragraph is indented this many
      spaces.  Note that non-first lines are indented two additional spaces
      relative to the first.
    width: Python int.  The returned lines strive to be no longer than `width`,
      counting the indentation.

  Returns:
    lines: Python list of strings, wrapped and indented.
  """
  # Algorithm:
  # - If everything fits on one line, do that
  # - Else, break immediately after the prefix, add extra indent,
  #   and lay out as many items on each line as fit.
  with_commas = [str(item) + ', ' for item in items]
  if with_commas:
    # Last item, if it exists, should not be followed by a comma
    with_commas[-1] = items[-1]
  total_length = (len(prefix) + sum([len(item) for item in with_commas])
                  + len(suffix))
  if total_length <= width - indent:
    return [' ' * indent + prefix + ''.join(with_commas) + suffix]
  else:
    wrapped = textwrap.wrap(
        text=''.join(with_commas) + suffix,
        width=width,
        initial_indent=' ' * (indent + 2),
        subsequent_indent=' ' * (indent + 2),
        break_long_words=False,
        break_on_hyphens=False)
    return [' ' * indent + prefix] + wrapped


def _render_pattern(pattern, nested=False):
  """Renders the given pattern as a string."""
  # TODO(axch) Think through breaking lines in patterns; currently just splats
  # the whole thing out on one line.
  if isinstance(pattern, (list, tuple)):
    subs = [_render_pattern(sub, nested=True) for sub in pattern]
    sub_str = ', '.join(subs)
    if nested:
      return '(' + sub_str + ')'
    else:
      return sub_str
  else:
    return str(pattern)


### Batch size detection support


def detect_batch_size(var_defs, init_vals, backend):
  """Returns the batch size implied by the top dimensions of the inputs."""
  var_def_dict = dict(var_defs)
  batch_size = 1
  determining_name = None
  for varname in sorted(init_vals.keys()):
    var_batch_size = _detect_batch_size_one_input(
        varname, var_def_dict[varname], init_vals[varname], backend)
    if batch_size == 1:
      batch_size = var_batch_size
      determining_name = varname
    elif var_batch_size == 1 or batch_size == var_batch_size:
      pass
    else:
      raise ValueError(
          'Inconsistent batch sizes: found {} in {} and {} in {}'.format(
              batch_size, determining_name, var_batch_size, varname))
  return batch_size


def _detect_batch_size_one_input(varname, type_, val, backend):
  """Returns the batch size implied by the Tensors in val."""
  batch_size = 1
  for _, item in pattern_zip(
      type_.tensors, val, leaf_type=TensorType):
    leaf_batch_size = backend.batch_size(item)
    if batch_size == 1:
      batch_size = leaf_batch_size
    elif leaf_batch_size == 1 or batch_size == leaf_batch_size:
      pass
    else:
      raise ValueError(
          'Inconsistent batch sizes in input {}: found {} and {}'.format(
              varname, batch_size, leaf_batch_size))
  return batch_size
