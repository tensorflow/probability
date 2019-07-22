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
"""Type inference pass on functional control flow graph.

Until converged, we propagate type information (dtype and shape) from inputs
toward outputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools

from absl import logging
import six

from tensorflow_probability.python.internal.auto_batching import instructions

__all__ = [
    'infer_types',
    'infer_types_from_signature',
    'is_inferring',
    'signature',
    'type_of_pattern',
]

# Change to logging.warn to get these logs without a bunch of TF executor spam.
log_debug = logging.debug


def _is_determined(type_):
  for item in instructions.pattern_traverse(
      type_.tensors, leaf_type=instructions.TensorType):
    if item is None:
      return False
    if item.dtype is None:
      return False
  return True


def _merge_tensor_type(old_type, obtained_type, backend):
  """Merges an updated instructions.TensorType for a single field."""
  if old_type is None:
    return obtained_type
  # Update the inferred dtype.
  if obtained_type is None:
    return old_type
  # One of old_type or obtained_type must be a TensorType because that's when
  # the caller's pattern_map2 would call _merge_tensor_type.
  if not isinstance(old_type, instructions.TensorType):
    raise ValueError('Type mismatch: Expected struct type {}, got {}'.format(
        old_type, obtained_type))
  if not isinstance(obtained_type, instructions.TensorType):
    raise ValueError('Type mismatch: Expected tensor type {}, got {}'.format(
        old_type, obtained_type))
  dtype = old_type.dtype
  obtained_dtype = obtained_type.dtype
  if dtype is None:
    dtype = obtained_dtype
  elif obtained_dtype is not None:
    dtype = backend.merge_dtypes(dtype, obtained_dtype)
  # Update the inferred shape.
  shape = old_type.shape
  obtained_shape = obtained_type.shape
  if shape is None:
    shape = obtained_shape
  elif obtained_shape is not None:
    shape = backend.merge_shapes(shape, obtained_shape)
  return instructions.TensorType(dtype, shape)


def _merge_var(varname, obtained_type, inferred_types, backend):
  """Merges an updated auto-batching type for a single variable."""
  old_type = inferred_types[varname]
  new_type = instructions.pattern_map2(
      functools.partial(_merge_tensor_type, backend=backend),
      old_type.tensors, obtained_type,
      leaf_type=instructions.TensorType)
  inferred_types[varname] = instructions.Type(new_type)
  if old_type != inferred_types[varname]:
    log_debug('{}: {} -> {}'.format(varname, old_type, inferred_types[varname]))


def _merge_vars(varnames, vm_types, inferred_types, backend, log_message=None):
  """Merges an updated vm_type for multiple variables.

  Args:
    varnames: Pattern of `string` variable names to merge.
    vm_types: Pattern of `instructions.TensorType` describing the incoming
      values.
    inferred_types: Extant dictionary of inferred types.  This is read
      to obtain the currently inferred types for the `varnames` and mutated
      to incorporate information from `vm_types`.
    backend: Object implementing required backend operations.
    log_message: Optional `string` describing this operation for the log.
  """
  if log_message is not None:
    log_debug(log_message + ': {}'.format(varnames))
  for varname, vm_type in instructions.pattern_zip(
      varnames, vm_types, leaf_type=instructions.Type):
    _merge_var(varname, vm_type, inferred_types, backend)


def type_of_pattern(val, backend, preferred_type=None):
  """Returns the `instructions.Type` of `val`.

  Args:
    val: Pattern of backend-specific `Tensor`s or a Python or numpy constant.
    backend: Object implementing required backend operations.
    preferred_type: `instructions.Type` to prefer, if `t` is a constant.

  Returns:
    vm_type: Pattern of `instructions.TensorType` describing `t`
  """
  def at_leaf(preferred_leaf_type, obj):
    """Pattern match at a leaf of the preferred_type pattern."""
    if preferred_leaf_type is None:
      return instructions.pattern_map(backend.type_of, obj)
    if isinstance(preferred_leaf_type, instructions.TensorType):
      return backend.type_of(obj, preferred_leaf_type.dtype)
    # Otherwise, preferred_leaf_type must be a (nested) list or tuple of
    # TensorType, while obj is not a list or a tuple (of anything).  In this
    # case, pattern_map2 should have raised an error, but we can defensively
    # raise an error here as well.
    msg = 'Type mismatch: Expected structured type {}, got object {}.'.format(
        preferred_leaf_type, obj)
    raise ValueError(msg)
  if preferred_type is None:
    preferred_type = instructions.Type(None)
  return instructions.pattern_map2(
      at_leaf, preferred_type.tensors, val,
      leaf_type=instructions.TensorType)


def _process_block(block, visited, inferred_types, backend):
  """Executes a pass of type inference on a single `Block`."""
  for op in block.instructions:
    log_debug('handle op {}'.format(op))
    if isinstance(op, instructions.PrimOp):
      if not all(_is_determined(inferred_types[var]) for var in op.vars_in):
        continue
      types_in = [inferred_types[var] for var in op.vars_in]
      # Offer type hints for cases where we need to type non-Tensor literals.
      preferred_types_out = instructions.pattern_map(
          lambda var: inferred_types[var], op.vars_out)
      with _type_inferring():
        objs_out = backend.run_on_dummies(
            op.function, _add_incompatible_batch_dim(types_in))
      types_out = _strip_batch_dim(instructions.pattern_map2(
          lambda tp, val: type_of_pattern(val, backend, preferred_type=tp),
          preferred_types_out, objs_out, leaf_type=instructions.Type))
      _merge_vars(op.vars_out, types_out, inferred_types, backend,
                  log_message='update PrimOp vars_out')
    elif isinstance(op, instructions.FunctionCallOp):
      if not all(_is_determined(inferred_types[var]) for var in op.vars_in):
        continue
      # First, bind op.vars_in to op.function.vars_in.
      types_in = [inferred_types[var].tensors for var in op.vars_in]
      _merge_vars(op.function.vars_in, types_in, inferred_types, backend,
                  log_message='init function vars_in')
      # Execute type inference.
      types_out = op.function.type_inference(types_in)
      for leaf in instructions.pattern_traverse(
          types_out, leaf_type=instructions.TensorType):
        if not isinstance(leaf, instructions.TensorType):
          msg = ('Expected function output type to be '
                 'a nested list or tuple of TensorType, found {}.').format(leaf)
          raise TypeError(msg)
      # To help with typing recursive base-case return literals, we seed
      # return_vars types before stepping into the function.
      _merge_vars(op.function.vars_out, types_out, inferred_types, backend,
                  log_message='update function vars_out')
      # Finally, update op.vars_out with the results of type inference.
      _merge_vars(op.vars_out, types_out, inferred_types, backend,
                  log_message='update FunctionCall vars_out')
      # Step into function. Note: it will only be visited once, if recursive.
      _process_graph(op.function.graph, visited, inferred_types, backend)
  # No need to process block.terminator, because all the information
  # that carries about types is already carried by the variable names
  # being the same across blocks


def _process_graph(graph, visited, inferred_types, backend):
  """Executes a pass of type inference on a single `ControlFlowGraph`."""
  for i in range(graph.exit_index()):
    block = graph.block(i)
    if block not in visited:
      visited.add(block)
      _process_block(block, visited, inferred_types, backend)


_inferring = False


def is_inferring():
  """Returns whether type inference is running.

  This can be useful for writing special primitives that change their behavior
  depending on whether they are being inferred, staged (see
  `virtual_machine.is_staging`), or neither (i.e., dry-run execution, see
  `frontend.Context.batch`).

  Returns:
    inferring: Python `bool`, `True` if this is called in the dynamic scope of
      type inference, otherwise `False`.
  """
  return _inferring


@contextlib.contextmanager
def _type_inferring():
  global _inferring
  old_inferring = _inferring
  try:
    _inferring = True
    yield
  finally:
    _inferring = old_inferring


def infer_types(program, inputs, backend):
  """Infers the variable types of a given program.

  Args:
    program: `instructions.Program` whose types to infer.
    inputs: A `list` of backend-compatible tensors aligned with
      `program.vars_in`.
    backend: Backend implementation.

  Returns:
    typed: `instructions.Program` with types inferred.

  Raises:
    ValueError: If some types still remain incomplete after inference.
  """
  sig = signature(program, inputs, backend)
  return infer_types_from_signature(program, sig, backend)


def infer_types_from_signature(program, sig, backend):
  """Infers the variable types of a given program.

  Args:
    program: `instructions.Program` whose types to infer.
    sig: A `list` of (patterns of) `instructions.TensorType` aligned with
      `program.vars_in`.
    backend: Backend implementation.

  Returns:
    typed: `instructions.Program` with types inferred.

  Raises:
    ValueError: If some types still remain incomplete after inference.
  """
  # We start from whatever types are pre-specified (i.e. program counter).
  inferred_types = dict(program.var_defs)
  _merge_vars(program.vars_in, sig, inferred_types, backend)
  log_debug('after inputs: {}'.format(inferred_types))
  snapshot = {}
  while snapshot != inferred_types:  # Until stationary, iterate.
    snapshot = dict(inferred_types)
    visited = set()

    _process_graph(program.graph, visited, inferred_types, backend)
  log_debug('after inference: {}'.format(inferred_types))

  for varname, vm_type in six.iteritems(inferred_types):
    if not _is_determined(vm_type):
      raise ValueError('Incomplete type inference for variable {}'.format(
          varname))

  return program.replace(var_defs=inferred_types)


def signature(program, inputs, backend):
  """Computes a type signature for the given `inputs`.

  Args:
    program: `instructions.Program` for whose inputs to compute the signature.
    inputs: A `list` of backend-compatible tensors aligned with
      `program.vars_in`.
    backend: Backend implementation.

  Returns:
    sig: A `list` of (patterns of) `instructions.TensorType` aligned with
      `program.vars_in`.
  """
  # Include whatever type information is pre-specified.
  return [
      _strip_batch_dim(type_of_pattern(
          val, backend, preferred_type=program.var_defs[varname]))
      for varname, val in zip(program.vars_in, inputs)]


def _add_incompatible_batch_dim(type_pat):
  """Adds a batch dim incompatible with all other known dims."""
  new_batch_dim = 2
  for tp in instructions.pattern_traverse(
      type_pat, leaf_type=instructions.TensorType):
    new_batch_dim = max(new_batch_dim, max((0,) + tp.shape) + 1)
  log_debug('using incompatible batch dim %d', new_batch_dim)
  def add_batch_dim_one_var(type_):
    return instructions.Type(instructions.pattern_map(
        lambda t: instructions.TensorType(t.dtype, (new_batch_dim,) + t.shape),
        type_.tensors, leaf_type=instructions.TensorType))
  return instructions.pattern_map(
      add_batch_dim_one_var, type_pat, leaf_type=instructions.Type)


def _strip_batch_dim(type_):
  return instructions.pattern_map(
      lambda t: instructions.TensorType(t.dtype, t.shape[1:]),
      type_, leaf_type=instructions.TensorType)
