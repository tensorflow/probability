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
"""A stackless auto-batching VM.

Borrows the stack, and conditional execution, from the host Python; manages only
divergence.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import heapq

# Dependency imports
from absl import logging
import six
from tensorflow_probability.python.experimental.auto_batching import instructions as inst


def execute(program, backend, block_code_cache, *inputs):
  """Executes a given program in stackless auto-batching mode.

  Compare `auto_batching.virtual_machine.execute`, which executes the program in
  full auto-batching mode.

  Advantages:
  - Stackless mode is much simpler than full mode.
  - Stackless mode incurs much less overhead, especially in TensorFlow.

  Disadvantages:
  - Stackless mode is only compatible with TensorFlow Eager.
  - Stackless mode cannot batch execution across function call boundaries.
    This is only relevant for recursive functions, and then only if
    they might converge at different stack depths.

  Algorithm:

  - Each function call is executed by its own (recursive) call to `_interpret`,
    with a current "active threads" mask.
    - This amounts to borrowing the stack from Python so we don't have to
      implement it.
    - This is why it is not possible to converge across function call
      boundaries.

  - The variable environment only has registers (and temporaries and nulls); no
    expensive full variables, since we are reusing the Python stack.

  - To execute one control flow graph:

    - Maintain (i) a current program counter, (ii) a current active threads
      mask, and (iii) a priority queue of waiting points, each with a mask
      giving the logical threads that are waiting to run from there.  All these
      masks should be disjoint.

    - At each time step, execute the basic block indicated by the current
      program counter, updating only the active threads.

      - Execute FunctionCallOp by recursively invoking `_interpret` (with only
        the threads that were active on entry to the FunctionCallOp).

    - If a block ends in a goto, enqueue the target, with the current active
      thread mask -- all these threads are waiting to resume there.

    - If a block ends in a branch, split the active threads according to the
      condition value and enqueue two entries, one for the true branch and one
      for the false branch.  This is how divergence happens.

    - At the end of each block (after enqueueing successors), dequeue the
      smallest program counter, and make active all the threads that were
      waiting there.  This is how re-convergence happens.

    - If the smallest remaining program counter is off the end of the graph,
      return.

  - Notes: (i) To avoid infinite regress, it's important to avoid actually
    enqueueing any blocks with an empty waiting thread mask; and (ii) In order
    to actually take advantage of re-convergence, we take care to coalesce
    queued entries waiting for the same block (by computing the or of their
    masks).

  This is a reimplementation in TensorFlow Eager of [1].

  [1] James Bradbury and Chunli Fu, "Automatic Batching as a Compiler Pass in
  PyTorch", Workshop on Systems for ML and Open Source Software at NeurIPS 2018.

  Args:
    program: A `instructions.Program` to execute.
    backend: Object implementing required backend operations.
    block_code_cache: Dict used to enable caching of defun+XLA across multiple
      calls to `execute`. If `None` is provided, we use a new dict per call to
      `execute` which can still achieve caching across depths of the call stack.
      This caching has no real effect unless calls to
      `backend.wrap_straightline_callable` have some effect.
    *inputs: Input arrays, each of shape `[batch_size, e1, ..., eE]`.  The batch
      size must be the same for all inputs.  The other dimensions must agree
      with the declared shapes of the variables they will be stored in, but need
      not in general be the same as one another.

  Returns:
    results: A list of the output values. Each returned value is an
      array of shape `[batch_size, e1, ..., eE]`.  The results are
      returned in the same order as the variables appear in
      `program.out_vars`.
  """
  init_vals = dict(zip(program.vars_in, inputs))
  batch_size = inst.detect_batch_size(program.var_defs, init_vals, backend)
  mask = backend.full_mask(batch_size)
  var_alloc = dict(program.var_alloc)
  if block_code_cache is None:
    block_code_cache = {}
  for var, alloc in six.iteritems(var_alloc):
    if alloc is inst.VariableAllocation.FULL:
      var_alloc[var] = inst.VariableAllocation.REGISTER
  return _interpret(
      program.replace(var_alloc=var_alloc), mask, backend, block_code_cache,
      *inputs)


def _split_fn_calls(instructions):
  """Yields lists of straight-line ops and individual `FunctionCallOp`s."""
  pending = []
  for op in instructions:
    if isinstance(op, inst.FunctionCallOp):
      if pending:
        yield pending
        pending = []
      yield op
    else:
      pending.append(op)
  if pending:
    yield pending


def _run_straightline(ops, backend, env_dict, mask):
  """Imperatively run a list of straight-line ops, return updated `env_dict`."""
  env = inst.Environment(env_dict, backend)
  for op in ops:
    if isinstance(op, inst.PrimOp):
      if (inst.pc_var in inst.pattern_flatten(op.vars_in) or
          inst.pc_var in inst.pattern_flatten(op.vars_out)):
        raise ValueError(
            'PrimOp reads or writes program counter: {}'.format(op))
      inputs = [inst.pattern_map(env.read, var_pat) for var_pat in op.vars_in]
      with _stackless_running():
        outputs = op.function(*inputs)
      new_vars = [(varname, env.push(varname, output, mask))
                  for varname, output in inst.pattern_zip(op.vars_out, outputs)]
    elif isinstance(op, inst.PopOp):
      new_vars = [(varname, env.pop(varname, mask)) for varname in op.vars]
    else:
      raise ValueError(
          'Invalid instruction in straightline segment: {}'.format(type(op)))
    env = inst.Environment(env.env_dict, env.backend, update=new_vars)
  return env.env_dict


def _interpret(program, mask, backend, block_code_cache, *inputs):
  """Worker function for `execute`; operates under a mask."""
  environment = inst.Environment.initialize(
      backend, program.var_alloc, program.var_defs, 0, backend.batch_size(mask))
  for var, inp in inst.pattern_zip(program.vars_in, inputs):
    environment[var] = environment.push(var, inp, mask)
  program_counter = 0  # Index of initial block
  queue = ExecutionQueue(backend)
  while program_counter != program.graph.exit_index():
    block = program.graph.block(program_counter)
    for split_idx, split in enumerate(_split_fn_calls(block.instructions)):
      if isinstance(split, inst.FunctionCallOp):
        op = split
        if (inst.pc_var in inst.pattern_flatten(op.vars_in) or
            inst.pc_var in inst.pattern_flatten(op.vars_out)):
          raise ValueError(
              'FunctionCallOp reads or writes program counter: {}'.format(op))
        inputs = [inst.pattern_map(environment.read, var_pat)
                  for var_pat in op.vars_in]
        # Option: Could gather+scatter at function boundaries.  Then the inner
        # interpreter would not need to accept the mask, but would need to
        # recompute the batch size and make a new mask of all ones.
        outputs = _invoke_fun(program, mask, backend, block_code_cache,
                              op.function, inputs)
        new_vars = [
            (varname, environment.push(varname, output, mask))
            for varname, output in inst.pattern_zip(op.vars_out, outputs)]
        environment = inst.Environment(
            environment.env_dict, environment.backend, update=new_vars)
      else:  # This split is not a FunctionCallOp.
        block_code_key = (id(program.graph), program_counter, split_idx)
        if block_code_key not in block_code_cache:
          logging.vlog(1, 'Fill block cache for block %s', block_code_key)
          varnames = inst.extract_referenced_variables(split)
          code = backend.wrap_straightline_callable(
              functools.partial(_run_straightline, split, environment.backend))
          block_code_cache[block_code_key] = (varnames, code)
        else:
          logging.vlog(1, 'Use cached code for block %s', block_code_key)
        varnames, code = block_code_cache[block_code_key]
        filtered_env = dict({  # Only pass variables relevant to these ops
            k: v for k, v in six.iteritems(environment.env_dict)
            if k in varnames})
        environment = inst.Environment(
            environment.env_dict,
            environment.backend,
            update=code(filtered_env, mask))
    op = block.terminator
    if isinstance(op, inst.BranchOp):
      if inst.pc_var == op.cond_var:
        raise ValueError('Branching on program counter: {}'.format(op))
      condition = environment.read(op.cond_var)
      true_index = program.graph.block_index(op.true_block)
      false_index = program.graph.block_index(op.false_block)
      queue.enqueue(true_index, mask & condition)
      queue.enqueue(false_index, mask & ~condition)
    elif isinstance(op, inst.GotoOp):
      next_index = program.graph.block_index(op.block)
      queue.enqueue(next_index, mask)
    else:
      raise TypeError('Unexpected op type: {}'.format(type(op)))
    program_counter, mask = queue.dequeue()
  # assert the queue is now empty
  return inst.pattern_map(environment.read, program.vars_out)


def _invoke_fun(program, mask, backend, block_code_cache, function, inputs):
  # TODO(axch): program_for_function computation is copied from instructions.py
  program_for_function = inst.Program(
      function.graph, program.functions,
      program.var_defs, function.vars_in, function.vars_out,
      program.var_alloc)
  return _interpret(program_for_function, mask, backend, block_code_cache,
                    *inputs)


class ExecutionQueue(object):
  """A priority queue of resumption points.

  Each resumption point is a pair of program counter to resume, and mask of
  threads that are waiting there.

  This class is a simple wrapper around Python's standard heapq implementation
  of priority queues.  There are just two subtleties:

  - Dequeue gets all the threads that were waiting at that point, by coalescing
    multiple entries if needed.

  - Enqueue drops entries with empty masks, because they need never be resumed.
  """

  def __init__(self, backend):
    self._backend = backend
    self._heap = []
    # Use insertion order as a tie-breaker when inserting, to prevent the heap
    # from trying to compare the mask objects to each other.
    self._insertion_ct = 0

  def enqueue(self, program_counter, mask):
    if self._backend.any(mask):
      self._insertion_ct += 1
      heapq.heappush(self._heap, (program_counter, self._insertion_ct, mask))

  def dequeue(self):
    program_counter, _, mask = heapq.heappop(self._heap)
    # Collect all threads waiting at this instruction
    while self._heap and program_counter == self._heap[0][0]:
      _, _, mask2 = heapq.heappop(self._heap)
      # TODO(axch): Sanity check: mask and mask2 should be disjoint
      mask = mask | mask2
    return program_counter, mask


_running = False


def is_running():
  """Returns whether the stackless machine is running a computation.

  This can be useful for writing special primitives that change their behavior
  depending on whether they are being staged, run stackless, inferred (see
  `type_inference.is_inferring`), or none of the above (i.e., dry-run execution,
  see `frontend.Context.batch`).

  Returns:
    running: Python `bool`, `True` if this is called in the dynamic scope of
      stackless running, otherwise `False`.
  """
  return _running


@contextlib.contextmanager
def _stackless_running():
  global _running
  old_running = _running
  try:
    _running = True
    yield
  finally:
    _running = old_running
