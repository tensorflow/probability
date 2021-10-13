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
"""Live variable analysis.

A variable is "dead" at some point if the compiler can find a proof that no
future instruction will read the value before that value is overwritten; "live"
otherwise.

This module implements a liveness analysis for the IR defined in
instructions.py.
"""

from absl import logging
import six

from tensorflow_probability.python.experimental.auto_batching import instructions as inst
from tensorflow_probability.python.experimental.auto_batching import liveness

__all__ = [
    'optimize'
]


def optimize(program):
  """Optimizes a `Program`'s variable allocation strategy.

  The variable allocation strategies determine how much memory the `Program`
  consumes, and how costly its memory access operations are (see
  `instructions.VariableAllocation`).  In general, a variable holding data with
  a longer or more complex lifetime will need a more expensive storage strategy.
  This analysis examines variables' liveness and opportunistically selects
  inexpensive sound allocation strategies.

  Specifically, the algorithm is to:
  - Run liveness analysis to determine the lifespan of each variable.
  - Assume optimistically that no variable needs to be stored at all
    (`instructions.VariableAllocation.NULL`).
  - Traverse the instructions and pattern-match conditions that require
    some storage:
    - If a variable is read by an instruction, it must be at least
      `instructions.VariableAllocation.TEMPORARY`.
    - If a variable is live out of some block (i.e., crosses a block boundary),
      it must be at least `instructions.VariableAllocation.REGISTER`.  This is
      because temporaries do not appear in the loop state in `execute`.
    - If a variable is alive across a call to an autobatched `Function`, it must
      be `instructions.VariableAllocation.FULL`, because that `Function` may
      push values to it that must not overwrite the value present at the call
      point.  (This can be improved by examining the call graph to see whether
      the callee really does push values to this variable, but that's future
      work.)

  Args:
    program: `Program` to optimize.

  Returns:
    program: A newly allocated `Program` with the same semantics but possibly
      different allocation strategies for some (or all) variables.  Each new
      strategy may be more efficient than the input `Program`'s allocation
      strategy for that variable (if the analysis can prove it safe), but will
      not be less efficient.
  """
  alloc = {var: inst.VariableAllocation.NULL for var in program.var_defs.keys()}
  # The program counter is always read
  _variable_is_read(inst.pc_var, alloc)
  _optimize_1(program.graph, inst.pattern_flatten(program.vars_out), alloc)
  for varname in program.vars_in:
    # Because there is a while_loop iteration between the inputs and the first
    # block.
    _variable_crosses_block_boundary(varname, alloc)
  for func in program.functions:
    _optimize_1(func.graph, inst.pattern_flatten(func.vars_out), alloc)
    for varname in func.vars_in:
      _variable_crosses_block_boundary(varname, alloc)
  null_vars = [k for k, v in six.iteritems(alloc)
               if v is inst.VariableAllocation.NULL]
  if null_vars:
    logging.warning('Found variables with NULL allocation. These are written '
                    'but never read: %s', null_vars)
  return program.replace(var_alloc=alloc)


def _vars_read_by(op):
  if isinstance(op, (inst.FunctionCallOp, inst.PrimOp)):
    return op.vars_in
  if isinstance(op, inst.BranchOp):
    return op.cond_var
  return []


def _variable_is_read(varname, alloc):
  if alloc[varname] is inst.VariableAllocation.NULL:
    alloc[varname] = inst.VariableAllocation.TEMPORARY


def _variable_crosses_block_boundary(varname, alloc):
  if (alloc[varname] is inst.VariableAllocation.NULL or
      alloc[varname] is inst.VariableAllocation.TEMPORARY):
    alloc[varname] = inst.VariableAllocation.REGISTER


def _variable_crosses_function_call_boundary(varname, alloc):
  alloc[varname] = inst.VariableAllocation.FULL


def _optimize_1(graph, live_out, alloc):
  """Optimize the variable allocation strategy for one CFG.

  Args:
    graph: `ControlFlowGraph` to traverse.
    live_out: Set of `str` variable names that are live out of this graph (i.e.,
      returned by the function this graph represents).
    alloc: Dictionary of allocation strategy deductions made so far.
      This is mutated; but no variable is moved to a cheaper strategy.
  """
  liveness_map = liveness.liveness_analysis(graph, set(live_out))
  if graph.exit_index() > 0:
    _variable_crosses_block_boundary(inst.pc_var, alloc)
  for i in range(graph.exit_index()):
    block = graph.block(i)
    for op, live_out in zip(
        block.instructions, liveness_map[block].live_out_instructions):
      for varname in inst.pattern_traverse(_vars_read_by(op)):
        _variable_is_read(varname, alloc)
      if isinstance(op, inst.FunctionCallOp):
        callee_writes = _indirectly_writes(op.function)
        for varname in live_out - set(inst.pattern_flatten(op.vars_out)):
          # A variable only needs the conservative storage strategy if it
          # crosses a call to some function that writes it (e.g., a recursive
          # self-call).
          if varname in callee_writes:
            _variable_crosses_function_call_boundary(varname, alloc)
          else:
            _variable_crosses_block_boundary(varname, alloc)
        # TODO(axch): Actually, the PC only needs a stack at this site if this
        # is not a tail call.
        _variable_crosses_function_call_boundary(inst.pc_var, alloc)
    if isinstance(block.terminator, inst.BranchOp):
      # TODO(axch): Actually, being read by BranchOp only implies
      # _variable_is_read.  However, the downstream VM doesn't know how to pop a
      # condition variable that is not needed after the BranchOp, so for now we
      # have to allocate a register for it.
      _variable_crosses_block_boundary(block.terminator.cond_var, alloc)
    for varname in liveness_map[block].live_out_of_block:
      _variable_crosses_block_boundary(varname, alloc)


def _directly_writes(graph):
  """Set of variables directly written by the given graph."""
  answer = set()
  for i in range(graph.exit_index()):
    block = graph.block(i)
    for op in block.instructions:
      if isinstance(op, inst.PrimOp):
        answer = answer.union(set(inst.pattern_flatten(op.vars_out)))
      elif isinstance(op, inst.FunctionCallOp):
        answer = answer.union(set(inst.pattern_flatten(op.vars_out)))
        # These because the caller writes them before the goto.
        answer = answer.union(set(inst.pattern_flatten(op.function.vars_in)))
      elif isinstance(op, inst.PopOp):
        # If the stack discipline is followed, any local variable will be
        # written by something before it is ever popped.  Formal parameters are
        # written by the caller and popped before returning.
        # Pops should not prevent a variable from being allocated as a register
        # instead of a full variable, because pops as such do not cause
        # registers to lose and data that a full variable would have kept.
        pass
  return answer


def _directly_calls(graph):
  """Set of Functions directly called by the given graph."""
  # TODO(axch): Deduplicate this and _directly_writes into a generic CFG
  # traversal?
  answer = set()
  for i in range(graph.exit_index()):
    block = graph.block(i)
    for op in block.instructions:
      if isinstance(op, inst.FunctionCallOp):
        answer.add(op.function)
  return answer


def _indirectly_writes(function):
  """Set of variables written by the given function including callees."""
  queue = [function]
  visited = set()
  answer = set()
  while queue:
    func = queue.pop()
    if func in visited:
      continue
    visited.add(func)
    answer = answer.union(_directly_writes(func.graph))
    queue += list(_directly_calls(func.graph))
  return answer
