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
"""Lowering the full IR to stack machine instructions.

At present, only one pass is needed to make the whole instruction
language defined in instructions.py understandable by the virtual
machine defined in virtual_machine.py, namely lowering FunctionCallOp
instructions to sequences of push, pop, and goto.
"""

import collections

from tensorflow_probability.python.experimental.auto_batching import instructions as inst
from tensorflow_probability.python.experimental.auto_batching import liveness

__all__ = [
    'lower_function_calls'
]


DefinedInfo = collections.namedtuple(
    'DefinedInfo', ['defined_into_block', 'defined_out_instructions'])


# Definedness analysis is private for now because it is only used for
# lowering.  If reuse is desired, should probably migrate out to a
# more appropriately named module, and review for warts specific to
# use in the lowering setting.
def _definedness_analysis(graph, defined_in, liveness_map):
  """Computes the defined and live variables.

  Specifically, for each op in each `Block`, computes the set of
  variables that are both defined coming out of it, and live coming
  out of the previous instruction.  The purpose of this analysis is to
  compute where `_lower_function_calls_1` should put `PopOp`s to
  enforce the stack discipline: The difference between the set
  computed here and the set of variables that are live out of each
  instruction is exactly the set for which a `PopOp` should be added.

  Why compute liveness and definedness jointly, rather than separately
  and then intersect them?  Because the purpose is to compute where to
  put the `PopOp`s, so that at the point at which any defined variable
  becomes dead, there is exactly one `PopOp`.  Placing such a `PopOp`
  will cause the variable in question to cease being defined, so this
  pass removes it from the defined set in anticipation thereof.

  Note that the semantics assumed by this analysis is that the control
  flow graph being analyzed does not use variable stacks internally,
  but they will only be used to implement the function sequence when
  function calls are lowered.  For this reason, a variable is treated
  as not being defined (regardless of what may be on its stack) until
  a write (as from `PrimOp` or `FunctionCallOp`) to it occurs (unless
  it comes in defined, in the `defined_in` argument).

  Args:
    graph: The `ControlFlowGraph` on which to perform definedness
      analysis.
    defined_in: A Python list of `str`.  The set of variables that
      are defined on entry to this `graph`.
    liveness_map: Python `dict` mapping each `Block` in `graph` to a
      `LivenessInfo` tuple, as produced by `liveness_analysis`.

  Returns:
    defined_map: Python `dict` mapping each `Block` in `graph` to a
      `DefinedInfo` tuple.  Each of these has two fields:
      `defined_into_block` gives the `frozenset` of `str` variable
      names defined into the block, and `defined_out_instructions`
      gives a list parallel to the `Block`s instructions list, of
      variables defined out of that instruction in the block, and live
      into it.

  Raises:
    ValueError: If an invalid instruction is encountered, if a live
      variable is undefined, or if different paths into a `Block`
      cause different sets of variables to be defined.
  """
  defined_map = {}
  defined = frozenset(defined_in)
  def record_vars_defined_on_entry(block, defined):
    if block not in defined_map:
      defined_map[block] = DefinedInfo(defined, [])
    elif defined_map[block].defined_into_block != defined:
      msg = ('Inconsistent defined variable set on entry into {}.\n'
             'Had {}, getting {}.').format(
                 block, defined_map[block].defined_into_block, defined)
      raise ValueError(msg)
  record_vars_defined_on_entry(graph.block(0), defined)
  def check_live_variables_defined(defined, live):
    for name in live:
      if name not in defined:
        raise ValueError('Detected undefined live variable {}.'.format(name))
  for i in range(graph.exit_index()):
    block = graph.block(i)
    defined = defined_map[block].defined_into_block
    check_live_variables_defined(defined, liveness_map[block].live_into_block)
    defined = defined.intersection(liveness_map[block].live_into_block)
    # Loop invariant: At this point, `defined` is the set of variables
    # that are defined and live on entry into this op.
    for op, live_out in zip(
        block.instructions, liveness_map[block].live_out_instructions):
      if isinstance(op, (inst.PrimOp, inst.FunctionCallOp)):
        defined = defined.union(inst.pattern_flatten(op.vars_out))
      elif isinstance(op, inst.PopOp):
        defined = defined.difference(op.vars)
      else:
        raise ValueError('Invalid instruction in block {}.'.format(op))
      # At this point, `defined` is the set of variables that are
      # defined on exit from this op, and live on entry into this op.
      defined_map[block].defined_out_instructions.append(defined)
      check_live_variables_defined(defined, live_out)
      defined = defined.intersection(live_out)
      # At this point, `defined` is the set of variables that are
      # defined and live on exit from this op.
    op = block.terminator
    if isinstance(op, inst.BranchOp):
      record_vars_defined_on_entry(op.true_block, defined)
      record_vars_defined_on_entry(op.false_block, defined)
    elif isinstance(op, inst.GotoOp):
      record_vars_defined_on_entry(op.block, defined)
    elif isinstance(op, inst.PushGotoOp):
      record_vars_defined_on_entry(op.goto_block, defined)
    elif isinstance(op, inst.IndirectGotoOp):
      # Check that the return set is defined
      check_live_variables_defined(defined, liveness_map[None].live_into_block)
    else:
      raise ValueError('Invalid terminator instruction {}.'.format(op))
  return defined_map


class ControlFlowGraphBuilder(object):
  """Encapsulation of the basic CFG state changes needed for lowering."""

  def __init__(self):
    self.blocks = []

  def cur_block(self):
    return self.blocks[-1]

  def append_block(self, block):
    self.blocks.append(block)

  def append_instruction(self, op):
    self.cur_block().instructions.append(op)

  def maybe_add_pop(self, defined, live):
    poppable = defined.difference(live)
    if poppable:
      self.append_instruction(inst.PopOp(list(poppable)))

  def split_block(self, target):
    """Split the current block with a returnable jump to the given block.

    The terminator of the current block becomes the terminator of the new last
    block.  The current block gets a `PushGotoOp` pushing the new last block and
    jumping to the given target block.

    Args:
      target: The block to jump to.
    """
    new_block = inst.Block(
        instructions=[],
        terminator=self.cur_block().terminator)
    self.cur_block().terminator = inst.PushGotoOp(new_block, target)
    self.append_block(new_block)

  def end_block_with_tail_call(self, target):
    """End the current block with a jump to the given block.

    The terminator of the current block becomes a `GotoOp` to the target.
    No new block is created (as it would be in `split_block`), because
    by assumption there are no additional instructions to return to.

    Args:
      target: The block to jump to.
    """
    self.cur_block().terminator = inst.GotoOp(target)

  def maybe_adjust_terminator(self):
    """May change the last block's terminator instruction to a return.

    If the terminator meant "exit this control flow graph", change it
    to "return from this function".

    Raises:
      ValueError: If the terminator was a `BranchOp` that directly
        exited, because there is no "conditional indirect goto"
        instruction in the target IR.
    """
    op = self.cur_block().terminator
    if inst.is_return_op(op):
      self.cur_block().terminator = inst.IndirectGotoOp()
    if (isinstance(op, inst.BranchOp) and
        (op.true_block is None or op.false_block is None)):
      # Why not?  Because there is no "conditional indirect goto"
      # instruction in the target IR.  One solution is to
      # canonicalize away directly exiting branches, by replacing
      # them with a branch to a fresh empty block that just exits.
      raise ValueError('Cannot lower exiting BranchOp {}.'.format(op))

  def control_flow_graph(self):
    return inst.ControlFlowGraph(self.blocks)


def _lower_function_calls_1(
    builder, graph, defined_in, live_out, function=True):
  """Lowers one function body, destructively.

  Mutates the given `ControlFlowGraphBuilder`, inserting `Block`s
  representing the new body.  Some of these may be the same as some
  `Block`s in the input `graph`, mutated; others may be newly
  allocated.

  Args:
    builder: `ControlFlowGraphBuilder` constructing the answer.
    graph: The `ControlFlowGraph` to lower.
    defined_in: A Python list of `str`.  The set of variables that
      are defined on entry to this `graph`.
    live_out: A Python list of `str`.  The set of variables that are
      live on exit from this `graph`.
    function: Python `bool`.  If `True` (the default), assume this is
      a `Function` body and convert an "exit" transfer into
      `IndirectGotoOp`; otherwise leave it as (`Program`) "exit".

  Raises:
    ValueError: If an invalid instruction is encountered, if a live
      variable is undefined, if different paths into a `Block` cause
      different sets of variables to be defined, or if trying to lower
      function calls in a program that already has `IndirectGotoOp`
      instructions (they confuse the liveness analysis).
  """
  liveness_map = liveness.liveness_analysis(graph, set(live_out))
  defined_map = _definedness_analysis(graph, defined_in, liveness_map)
  for i in range(graph.exit_index()):
    block = graph.block(i)
    old_instructions = block.instructions
    # Resetting block.instructions here because we will build up the
    # list of new ones in place (via the `builder`).
    block.instructions = []
    builder.append_block(block)
    builder.maybe_add_pop(
        defined_map[block].defined_into_block,
        liveness_map[block].live_into_block)
    for op_i, (op, defined_out, live_out) in enumerate(zip(
        old_instructions,
        defined_map[block].defined_out_instructions,
        liveness_map[block].live_out_instructions)):
      if isinstance(op, inst.PrimOp):
        for name in inst.pattern_traverse(op.vars_in):
          if name in inst.pattern_flatten(op.vars_out):
            # Why not?  Because the stack discipline we are trying to
            # implement calls for popping variables as soon as they
            # become dead.  Now, if a PrimOp writes to the same
            # variable as it reads, the old version of that variable
            # dies.  Where to put the PopOp?  Before the PrimOp is no
            # good -- it still needs to be read.  After the PrimOp is
            # no good either -- it will pop the output, not the input.
            # Various solutions to this problem are possible, such as
            # adding a "drop the second-top element of this stack"
            # instruction, or orchestrating the pushes and pops
            # directly in the interpreter, but for now the simplest
            # thing is to just forbid this situation.
            # Fixing this is b/118884528.
            msg = 'Cannot lower PrimOp that writes to its own input {}.'
            raise ValueError(msg.format(name))
        builder.append_instruction(op)
        builder.maybe_add_pop(defined_out, live_out)
      elif isinstance(op, inst.FunctionCallOp):
        names_pushed_here = inst.pattern_flatten(op.vars_out)
        for name in inst.pattern_traverse(op.vars_in):
          if name in names_pushed_here:
            # For the same reason as above.
            # Fixing this is b/118884528.
            msg = 'Cannot lower FunctionCallOp that writes to its own input {}.'
            raise ValueError(msg.format(name))
        # The variables that were defined on entry to this instruction (i.e.,
        # not pushed here) but are not live out don't need to remain on their
        # stacks when the callee is entered.
        defined_in = defined_out.difference(names_pushed_here)
        to_pop = defined_in.difference(live_out)
        for new_op in _function_entry_stack_ops(op, to_pop):
          builder.append_instruction(new_op)
        if (op_i == len(old_instructions) - 1
            and _optimizable_tail_call(op, builder.cur_block())):
          builder.end_block_with_tail_call(op.function.graph.block(0))
          # The check that the tail call is optimizable is equivalent to
          # checking that the push-pop pair below would do nothing.
        else:
          builder.split_block(op.function.graph.block(0))
          builder.append_instruction(
              # These extra levels of list protect me (I hope) from the
              # auto-unpacking in the implementation of push_op, in the case of
              # a function returning exactly one Tensor.
              inst.push_op([op.function.vars_out], [op.vars_out]))
          builder.append_instruction(
              inst.PopOp(inst.pattern_flatten(op.function.vars_out)))
          # The only way this would actually add a pop is if some name written
          # by this call was a dummy variable.
          builder.maybe_add_pop(frozenset(names_pushed_here), live_out)
      elif isinstance(op, (inst.PopOp)):
        # Presumably, lowering is applied before any `PopOp`s are present.  That
        # said, simply propagating them is sound.  (But see the `PopOp` case in
        # `liveness_analysis`.)
        builder.append_instruction(op)
      else:
        raise ValueError('Invalid instruction in block {}.'.format(op))
    if function:
      builder.maybe_adjust_terminator()


def _is_indirect_return_op(op):
  return (inst.is_return_op(op)
          or (isinstance(op, inst.GotoOp) and _is_return_block(op.block)))


def _is_return_block(block):
  return (not block.instructions) and _is_indirect_return_op(block.terminator)


def _optimizable_tail_call(op, block):
  # A tail call is OK if no translation of the value is needed.  This requires
  # the function being tail-called to write its result into the same variable as
  # the caller's caller expects, which we check here.  Generally, this will tend
  # to happen with self-tail-calls.
  return (_is_indirect_return_op(block.terminator)
          and op.vars_out == op.function.vars_out)


def _function_entry_stack_ops(op, to_pop):
  """Computes a set of stack operations for the entry to a FunctionCallOp.

  The function calling convention is
  - Push the arguments to the formal parameters
  - Pop any now-dead arguments so they're not on the stack during the call
  - Jump to the beginning of the function body

  This can be a little tricky for a self-call, because then the arguments and
  the formal parameters live in the same name space and can collide.  This
  helper does something reasonable, and errors out when it can't.

  Args:
    op: FunctionCallOp instance giving the call to make stack operations for.
    to_pop: Set of names to make sure are popped before entering.

  Returns:
    ops: List of instruction objects that accomplish the goal.
  """
  push_from = []
  push_to = []
  caller_side_vars = inst.pattern_flatten(op.vars_in)
  callee_side_vars = inst.pattern_flatten(op.function.vars_in)
  for caller_side, callee_side in inst.pattern_zip(
      caller_side_vars, callee_side_vars):
    if caller_side == callee_side:
      # This can happen if this is a self-call and we're just passing the
      # variable through to itself
      if caller_side in to_pop:
        # The top of the stack is already correct, and the callee will pop our
        # unneeded value off it for us: skip the push and the pop.
        to_pop = to_pop.difference([caller_side])
      else:
        # The top of the stack is correct, but we need to push it anyway because
        # the callee will (eventually) pop but we still need the current value
        # when the callee returns.
        push_from.append(caller_side)
        push_to.append(callee_side)
    elif callee_side in caller_side_vars:
      # If the graph of transfers turns out not to be a DAG, can't implement it
      # without temporary space; and don't want to bother computing a safe order
      # even if it does.
      # Fixing this is b/135275883.
      msg = ('Cannot lower FunctionCallOp that reuses its own input {}'
             ' as a formal parameter.')
      raise ValueError(msg.format(caller_side))
    # Checking `elif caller_side in callee_side_vars` is redundant because
    # the callee_side check will trigger on that pair sooner or later.
    else:
      # Ordinary transfer: push now and then pop if needed.  The pop will not
      # interfere with the push because only caller-side variables can possibly
      # be popped.
      assert callee_side not in to_pop
      push_from.append(caller_side)
      push_to.append(callee_side)
  push_inst = inst.push_op(push_from, push_to)
  if to_pop:
    return [push_inst, inst.PopOp(list(to_pop))]
  else:
    return [push_inst]


def lower_function_calls(program):
  """Lowers a `Program` that may have (recursive) FunctionCallOp instructions.

  Mutates the `ControlFlowGraph` of the input program in place.  After
  lowering, the result CFG

  - Has no `FunctionCallOp` instructions

  - Obeys a stack discipline

  What is the stack discipline?  Every function body becomes a CFG
  subset that:

  - Never transfers control in except to the first block
    (corresponding to being called), or to a block stored with
    `PushGotoOp` (corresponding to a subroutine returning)

  - Never transfers control out except with `IndirectGotoOp`
    (corresponding to returning), or with a `PushGotoOp`
    (corresponding to calling a subroutine)

  - Every path through the graph has the following effect on the
    variable stacks:

    - The formal parameters receive exactly one net pop

    - The return variables receive exactly one net push

    - All other variable stacks are left as they are

    - No data is read except the top frames of the formal parameter
      stacks

  Why mutate in place?  Because tying the knot in the result seemed
  too hard without an explicit indirection between `Block`s and
  references thereto in various `Op`s.  Specifically, when building a
  new CFG to replicate the structure of an existing one, it is
  necessary to allocate `Block`s to serve as the targets of all
  `BranchOp`, `GotoOp` (and `FunctionCallOp`) before building those
  `Op`s, and then correctly reuse those `Block`s when processing said
  targets.  With an explicit indirection, however, it would have been
  possible to reuse the same `Label`s, simply creating a new mapping
  from them to `Block`s.

  Note that the semantics assumed by this transformation is that the
  CFGs being transformed do not use variable stacks internally, but
  they will only be used to implement the function sequence when
  function calls are lowered.  This semantics licenses placing
  `PopOp`s to enforce a stack discipline for `FunctionCallOp`s.

  Args:
    program: A `Program` whose function calls to lower.  `Block`s in
      the program may be mutated.

  Returns:
    lowered: A `Program` that defines no `Function`s and does not use the
      `FunctionCallOp` instruction.  May share structure with the input
      `program`.

  Raises:
    ValueError: If an invalid instruction is encountered, if a live
      variable is undefined, if different paths into a `Block` cause
      different sets of variables to be defined, or if trying to lower
      function calls in a program that already has loops (within a
      `Function` body) or `IndirectGotoOp` instructions (they confuse
      the liveness analysis).
  """
  builder = ControlFlowGraphBuilder()
  _lower_function_calls_1(
      builder, program.graph, program.vars_in,
      inst.pattern_flatten(program.vars_out), function=False)
  for func in program.functions:
    _lower_function_calls_1(
        builder, func.graph, func.vars_in, inst.pattern_flatten(func.vars_out))
  return inst.Program(
      builder.control_flow_graph(), [],
      program.var_defs, program.vars_in, program.vars_out,
      program.var_alloc)
