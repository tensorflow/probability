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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow_probability.python.experimental.auto_batching import instructions as inst

__all__ = [
    'liveness_analysis'
]


LivenessInfo = collections.namedtuple(
    'LivenessInfo',
    ['live_into_block', 'live_out_instructions', 'live_out_of_block'])


def liveness_analysis(graph, live_out):
  """Computes liveness information for each op in each block.

  A "live" variable is one that this analysis cannot prove will not be
  read later in the execution of the program:
  https://en.wikipedia.org/wiki/Live_variable_analysis.

  Note that the semantics assumed by this analysis is that the control
  flow graph being analyzed does not use variable stacks internally,
  but they will only be used to implement the function sequence when
  function calls are lowered.  For this reason, a write (as from
  `PrimOp` and `FunctionCallOp`) is treated as killing a variable,
  even though the downstream virtual machine pushes the outputs of
  `PrimOp`.  This semantics is also why `lower_function_calls` can
  place `PopOp`s.

  The algorithm is to traverse the blocks in the CFG in reverse order,
  computing the in-block liveness map and the set of variables that
  are live into the entry of each block.

  - Within a block, proceed down the instructions in reverse order,
    updating the live variable set as each instruction is stepped
    over, and recording (a copy of) it.  The set of variables that are
    live out of the last instruction of a block is the union of the
    variables that are live into the blocks to which control may be
    transferred (plus anything the terminator instruction itself may
    read).

  - As coded, this assumes that all control transfers go later in the
    CFG.  Supporting loops will require equation solving.

  - As coded, crashes in the presence of `IndirectGotoOp`.

  The latter two limitations amount to requiring this liveness
  analysis only be used before lowering function calls, and that the
  body of every `Function` analyzed be loop-free.  Ergo, all recursive
  computations must cross `FunctionCallOp` boundaries in any program
  to which this is applied.

  Args:
    graph: The `ControlFlowGraph` on which to perform liveness analysis.
    live_out: A Python list of `str`.  The set of variables that are
      live on exit from this `graph`.

  Returns:
    liveness_map: Python `dict` mapping each `Block` in `graph` to a
      `LivenessInfo` tuple.  Each of these has three fields:
      `live_into_block` gives the `frozenset` of `str` variable names
      live into the block; `live_out_instructions` gives a list
      parallel to the `Block`'s instructions list, of variables live
      out of that instruction; and `live_out_of_block` gives the
      `frozenset` of `str` variable names live out of the block.

  Raises:
    ValueError: If an invalid instruction is encountered, or if trying
      to do liveness analysis in the presence of IndirectGotoOp or of
      backward control transfers.
  """
  answer = {None: LivenessInfo(frozenset(live_out), [], frozenset(live_out))}
  for i in reversed(range(graph.exit_index())):
    block = graph.block(i)
    block_answer = []
    op = block.terminator
    if isinstance(op, inst.BranchOp):
      if op.true_block not in answer or op.false_block not in answer:
        loser = op.true_block if op.true_block not in answer else op.false_block
        msg = ('Liveness analysis detected backward reference to {}.\n'
               'Liveness analysis of loops is not (yet) supported.')
        raise ValueError(msg.format(loser))
      live_in_true = answer[op.true_block].live_into_block
      live_in_false = answer[op.false_block].live_into_block
      live_out_of_block = frozenset(live_in_true.union(live_in_false))
      live_out = live_out_of_block.union([op.cond_var])
    elif isinstance(op, inst.GotoOp):
      if op.block not in answer:
        msg = ('Liveness analysis detected backward reference to {}.\n'
               'Liveness analysis of loops is not (yet) supported.')
        raise ValueError(msg.format(op.block))
      live_out = answer[op.block].live_into_block
      live_out_of_block = frozenset(live_out)
    elif isinstance(op, inst.PushGotoOp):
      raise ValueError("Liveness analysis didn't expect PushGotoOp.")
    elif isinstance(op, inst.IndirectGotoOp):
      raise ValueError("Liveness analysis didn't expect IndirectGotoOp.")
    else:
      raise ValueError('Invalid terminator instruction {}.'.format(op))
    for op in reversed(block.instructions):
      # Loop invariant: At this point, `live_out` is the set of
      # variables that are live out of this instruction.
      block_answer.insert(0, live_out)
      if isinstance(op, inst.PrimOp) or isinstance(op, inst.FunctionCallOp):
        live_out = live_out.difference(
            inst.pattern_flatten(op.vars_out)).union(op.vars_in)
      elif isinstance(op, inst.PopOp):
        # If this pop is already here, the variable being popped is
        # live, and must not be popped an extra time by the lowering
        # operation.  It may be reasonable to eliminate all pops
        # before running liveness analysis, though, as a way of
        # placing them in better locations.
        # - Alternative considered: define _lower_function_calls_1 to
        #   remove pops as encountered, after liveness analysis, and
        #   don't mark their variables live here.  Con: this putative
        #   structure would entangle the liveness analysis more with
        #   how it is used.
        live_out = live_out.union(op.vars)
      else:
        raise ValueError('Invalid instruction in block {}.'.format(op))
    answer[block] = LivenessInfo(live_out, block_answer, live_out_of_block)
  return answer
