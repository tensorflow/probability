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
"""Optimizing stack usage (pushes and pops)."""

import collections

from tensorflow_probability.python.experimental.auto_batching import instructions as inst


def fuse_pop_push(program):
  """Fuses pop+push sequences in the given `Program`.

  A stack pop followed by a stack push (with no intervening read) is equivalent
  to just updating the top of the stack.  The latter is more efficient for FULL
  variables, because it just updates the cache for the top, and avoids gathering
  from and scattering to the backing stack Tensor.

  This pass mutates the `ControlFlowGraph` of the input `Program` to convert
  pop+push sequences into updates.  The pass will work despite intervening
  instructions that interact with other Variables, but will not cross basic
  block boundaries.  As a side-effect, the pass moves non-optimized pops to the
  last place in their basic block where they are still sound.  This has no
  effect on the runtime behavior of the program.

  Args:
    program: A lowered `Program` whose pop+push sequences to fuse.  `Block`s in
      the program may be mutated.

  Returns:
    fused: A `Program` with statically redundant pop+push sequences eliminated
      in favor of `PrimOp`s with non-trivial `skip_push_mask` fields.

  Raises:
    ValueError: If the input `Program` has not been lowered (i.e., contains
      `FunctionCallOp`), or is ill-formed (e.g., contains invalid instructions).
  """
  for i in range(program.graph.exit_index()):
    block = program.graph.block(i)
    new_instructions = []
    waiting_for_pop = collections.OrderedDict([])
    for op in block.instructions:
      if isinstance(op, inst.PrimOp):
        skip_push = set()
        for var in inst.pattern_traverse(op.vars_in):
          if var in waiting_for_pop:
            new_instructions.append(inst.PopOp([var]))
            del waiting_for_pop[var]
        for var in inst.pattern_traverse(op.vars_out):
          if var in waiting_for_pop:
            skip_push.add(var)
            del waiting_for_pop[var]
        new_instructions.append(
            inst.prim_op(op.vars_in, op.vars_out, op.function, skip_push))
      elif isinstance(op, inst.FunctionCallOp):
        raise ValueError("pop-push fusion not implemented for pre-lowering")
      elif isinstance(op, inst.PopOp):
        for var in inst.pattern_traverse(op.vars):
          if var in waiting_for_pop:
            new_instructions.append(inst.PopOp([var]))
          else:
            waiting_for_pop[var] = True
      else:
        raise ValueError("Unrecognized op in pop-push fusion", op)
    if waiting_for_pop:
      new_instructions.append(inst.PopOp(list(waiting_for_pop.keys())))
    block.instructions = new_instructions
  return program
