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
"""The auto-batching VM itself."""

import contextlib
import functools

# Dependency imports

from absl import logging
import six

from tensorflow_probability.python.experimental.auto_batching import instructions as inst


__all__ = [
    'execute',
    'is_staging',
]


def execute(program, args, max_stack_depth, backend, block_code_cache=None):
  """Executes or stages a complete auto-batching VM program.

  Whether this executes or stages computation depends on whether the backend has
  an eager or deferred computation model.

  The dimensions of the inputs and internal variables are split into
  one top batch dimension and an arbitrary number (here `E`) event
  dimensions.  The event rank may be different for different inputs,
  outputs, and internal variables.

  Args:
    program: A `instructions.Program` to execute or stage.
    args: Input values, a list of arrays, each of shape `[batch_size,
      e1, ..., eE]`.  The batch size must be the same for all inputs.
      The other dimensions must agree with the declared shapes of the
      variables they will be stored in, but need not in general be the
      same as one another.
    max_stack_depth: Python `int`. Maximum depth of stack to allocate.
    backend: Object implementing required backend operations.
    block_code_cache: Dict (allows cache to live across calls to `vm.execute`,
      or `None` (in which case a dict is created and used per call).

  Returns:
    results: A list of the output values. Each returned value is an
      array of shape `[batch_size, e1, ..., eE]`.  The results are
      returned in the same order as the variables appear in
      `program.out_vars`.
  """
  program = select_block_priority(program)
  halt_index = program.graph.exit_index()
  logging.vlog(1, 'Staging computation: %d blocks to stage', halt_index)
  valid_indices = range(halt_index)
  assert len(program.vars_in) == len(args)
  init_vals = dict(zip(program.vars_in, args))
  environment = _initialize_environment(
      program, init_vals, max_stack_depth, backend)
  next_block_index = _choose_next_op(environment, backend)

  if block_code_cache is None:
    block_code_cache = {}

  def make_run_block_callable(env):
    """Makes a i->next_env callable using cached, backend-wrapped _run_block."""
    def run_block_callable(i):
      if i not in block_code_cache:
        logging.vlog(1, 'Fill block code cache: block %d', i)
        block_code_cache[i] = backend.wrap_straightline_callable(
            lambda env_arg: _run_block(program.graph, i, env_arg, backend))
      else:
        logging.vlog(1, 'Use cached block code: block %d', i)
      return block_code_cache[i](env)
    return run_block_callable

  def cond(_, next_block_index):
    return backend.not_equal(next_block_index, halt_index)

  def body(env_dict, next_block_index):  # pylint:disable=missing-docstring
    # This `_staged_apply` turns into the block dispatch tree (see
    # docstring of `_staged_apply`).
    # Briefly, this will build a graph snippet for each basic block
    # in the control flow graph, and glue them together with a switch
    # on the runtime value of `next_block_index`.
    env_dict = backend.prepare_for_cond(env_dict)
    f = make_run_block_callable(env_dict)
    env_dict = backend.switch_case(
        next_block_index, [functools.partial(f, i) for i in valid_indices])
    next_block_index = _choose_next_op(
        inst.Environment(env_dict, backend), backend)
    return env_dict, next_block_index

  env_dict, _ = backend.while_loop(
      cond, body, [environment.env_dict, next_block_index])
  final_env = inst.Environment(env_dict, backend)
  return inst.pattern_map(final_env.read, program.vars_out)


def _run_block(graph, index, env_dict, backend):
  """Executes or stages one basic block.

  When staging, the `graph`, the `index`, and the `backend` are static.
  The `environment` contains the runtime values (e.g. `Tensor`s or
  `np.ndarray`s) being staged.

  Args:
    graph: The `instructions.ControlFlowGraph` of blocks that exist in
      the program being run.
    index: The int index of the specific block to execute or stage.
    env_dict: The current variable environment of the VM (for all logical
      threads). The program counter is included as a variable with a
      reserved name.
    backend: Object implementing required backend operations.

  Returns:
    new_environment: The new variable environment after performing the
      block on the relevant threads.

  Raises:
    ValueError: Invalid opcode or illegal operation (e.g. branch/read/write
      program counter).
  """
  environment = inst.Environment(env_dict, backend)
  program_counter = environment.read(inst.pc_var)
  block = graph.block(index)
  logging.debug('Staging block %d:\n%s', index, block)
  mask = backend.equal(program_counter, index)
  def as_index(block):
    return backend.broadcast_to_shape_of(
        graph.block_index(block), program_counter)
  for op in block.instructions:
    if isinstance(op, inst.PrimOp):
      if (inst.pc_var in inst.pattern_flatten(op.vars_in) or
          inst.pc_var in inst.pattern_flatten(op.vars_out)):
        raise ValueError(
            'PrimOp reading or writing program counter: {}'.format(op))
      inputs = [inst.pattern_map(environment.read, var_pat)
                for var_pat in op.vars_in]
      with _vm_staging():
        outputs = op.function(*inputs)
      def doit(varname, output, skip_push_mask):
        if varname in skip_push_mask:
          return environment.update(varname, output, mask)
        else:
          return environment.push(varname, output, mask)
      new_vars = [(varname, doit(varname, output, op.skip_push_mask))
                  for varname, output in inst.pattern_zip(op.vars_out, outputs)]
    elif isinstance(op, inst.PopOp):
      new_vars = [(varname, environment.pop(varname, mask))
                  for varname in op.vars]
    else:
      raise ValueError('Invalid instruction in block: {}'.format(type(op)))
    environment = inst.Environment(
        environment.env_dict, environment.backend, update=new_vars)
  op = block.terminator
  if isinstance(op, inst.BranchOp):
    if inst.pc_var == op.cond_var:
      raise ValueError('Branching on program counter: {}'.format(op))
    condition = environment.read(op.cond_var)
    next_index = backend.where(
        condition, as_index(op.true_block), as_index(op.false_block))
    environment[inst.pc_var] = environment.update(inst.pc_var, next_index, mask)
  elif isinstance(op, inst.GotoOp):
    environment[inst.pc_var] = environment.update(
        inst.pc_var, as_index(op.block), mask)
  elif isinstance(op, inst.PushGotoOp):
    environment[inst.pc_var] = environment.update(
        inst.pc_var, as_index(op.push_block), mask)
    environment[inst.pc_var] = environment.push(
        inst.pc_var, as_index(op.goto_block), mask)
  elif isinstance(op, inst.IndirectGotoOp):
    environment[inst.pc_var] = environment.pop(inst.pc_var, mask)
  else:
    raise TypeError('Unexpected op type: {}'.format(type(op)))
  return environment.env_dict


_staging = False


def is_staging():
  """Returns whether the virtual machine is staging a computation.

  This can be useful for writing special primitives that change their behavior
  depending on whether they are being staged, run stackless, inferred (see
  `type_inference.is_inferring`), or none of the above (i.e., dry-run execution,
  see `frontend.Context.batch`).

  Returns:
    staging: Python `bool`, `True` if this is called in the dynamic scope of
      VM staging, otherwise `False`.
  """
  return _staging


@contextlib.contextmanager
def _vm_staging():
  global _staging
  old_staging = _staging
  try:
    _staging = True
    yield
  finally:
    _staging = old_staging


def _choose_next_op(environment, backend):
  # At each step, execute the instruction whose program counter is
  # smallest (among those that have logical threads).
  return backend.reduce_min(environment.read(inst.pc_var))


def _initialize_environment(
    program, init_vals, max_stack_depth, backend):
  """Construct initial environment.

  This pre-allocates Tensors (in the form of the backend's
  `Variable`s) of the right types and shapes to hold all the data that
  may be needed, and populates the variables holding the inputs with
  their input initial values.

  Args:
    program: The `Program` to be executed.
    init_vals: A dict mapping input variable names to patterns of Tensors of
      shape `[batch_size, e1, ..., eE]`.  These are the initial input
      values. Note that batch size is currently inferred from
      init_vals[0].shape[0]. To support arity-0 functions, we would need to
      either infer this from the program or accept it as an input argument.
    max_stack_depth: Python `int` giving the number of frames to preallocate on
      each stack Tensor.
    backend: Object implementing required backend operations.

  Returns:
    environment: The initial variable environment (including an initial program
      counter at instruction 0).
  """
  var_alloc = program.var_alloc
  var_defs = program.var_defs
  _check_initial_dtypes(var_defs, init_vals, backend)
  batch_size = inst.detect_batch_size(var_defs, init_vals, backend)
  environment = inst.Environment.initialize(
      backend, var_alloc, var_defs, max_stack_depth, batch_size)
  mask = backend.full_mask(batch_size)
  for name, val in six.iteritems(init_vals):
    environment[name] = environment.push(name, val, mask)
  return _initialize_pc(environment, program, batch_size, mask, backend)


def _initialize_pc(environment, program, batch_size, mask, backend):
  """Write the requisite initial values to the program counter."""
  # Initializing the program counter is a bit subtle.  The initial function call
  # to main is a tail call, so doesn't change the PC's stack depth.  However,
  # main will try to "return" by executing IndirectGotoOp when it's done.  This
  # should have the effect of setting that thread's (top) PC to the halt index,
  # so the VM knows not to keep executing it.
  #
  # We arrange this effect here by initializing the PC to depth 2: the bottom
  # row is the halt index, and the top row is the initial PC, i.e., 0.  This
  # way, when main pops the PC at the end, the current value will become the
  # halt index, as desired.
  #
  # This solution may appear somewhat unfortunate, because it forces the PC to
  # have a stack, i.e., be allocated as a FULL variable.  However, if the
  # program has any IndirectGotoOp in it, we need a stack for the PC anyway; and
  # if it doesn't, then it's ok to allocate the PC as a REGISTER and drop the
  # halt index on the floor, because there won't be any IndirectGotoOp to read
  # it.
  #
  # Other alternatives considered:
  # - Define a REGISTER_WITH_DEFAULT variable allocation strategy, which is
  #   a register that writes a default value into itself when popped.
  # - Separate the PC into an explicit register holding the current PC, together
  #   with a Variable for the PC stack.  Then the latter could itself be a
  #   register (holding the halt index) if no other PC pushes are needed in the
  #   program.
  var_defs = program.var_defs
  halt_pc = backend.fill(
      program.graph.exit_index(), batch_size,
      dtype=var_defs[inst.pc_var].tensors.dtype, shape=[])
  environment[inst.pc_var] = environment.push(inst.pc_var, halt_pc, mask)
  start_pc = backend.fill(
      0, batch_size, dtype=var_defs[inst.pc_var].tensors.dtype, shape=[])
  environment[inst.pc_var] = environment.push(inst.pc_var, start_pc, mask)
  return environment


def _check_initial_dtypes(var_defs, init_vals, backend):
  var_def_dict = dict(var_defs)
  for varname, val in six.iteritems(init_vals):
    for tensor_type, subval in inst.pattern_zip(
        var_def_dict[varname].tensors, val, leaf_type=inst.TensorType):
      backend.assert_matching_dtype(
          tensor_type.dtype, subval, 'var name {}'.format(varname))


def select_block_priority(program):
  """Order `Block`s in `program` by execution priority."""
  msg = 'TODO(axch): Implement block strategy selection for Functions.'
  assert not program.functions, msg
  def sync_weight(block):
    # Sort all "trivial" blocks that don't call user-land operations ahead of
    # all others.  This critically relies on `sorted` being stable to work
    # correctly.
    # TODO(b/118911579): Respect user-specified sync priority when that happens.
    for op in block.instructions:
      if isinstance(op, (inst.PrimOp, inst.FunctionCallOp)):
        return 1
    return 0
  # Have to keep the first block because that's where control enters.
  # This is unfortunate if that block is over-heavy.
  # Could be fixed by
  # - Adding a field to Program for the initial value of the program counter, or
  # - Ensuring that the first block is an empty indirection block
  new_blocks = ([program.graph.block(0)]
                + sorted(program.graph.blocks[1:], key=sync_weight))
  new_graph = inst.ControlFlowGraph(new_blocks)
  return inst.Program(new_graph, [], program.var_defs,
                      program.vars_in, program.vars_out, program.var_alloc)
