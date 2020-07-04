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
"""Experimental Numpy backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import nest


__all__ = [
    'foldl',
    'map_fn',
    'pfor',
    'vectorized_map',
    'scan',
]


JAX_MODE = False


def _foldl_jax(fn, elems, initializer=None, parallel_iterations=10,  # pylint: disable=unused-argument
               back_prop=True, swap_memory=False, name=None):  # pylint: disable=unused-argument
  """tf.foldl, in JAX."""
  if initializer is None:
    initializer = nest.map_structure(lambda el: el[0], elems)
    elems = nest.map_structure(lambda el: el[1:], elems)
  if len(set(nest.flatten(nest.map_structure(len, elems)))) != 1:
    raise ValueError(
        'Mismatched element sizes: {}'.format(nest.map_structure(len, elems)))
  from jax import lax  # pylint: disable=g-import-not-at-top
  return lax.scan(
      lambda carry, el: (fn(carry, el), None), initializer, elems)[0]


def _foldl(fn, elems, initializer=None, parallel_iterations=10,  # pylint: disable=unused-argument
           back_prop=True, swap_memory=False, name=None):  # pylint: disable=unused-argument
  """tf.foldl, in numpy."""
  elems_flat = nest.flatten(elems)
  if initializer is None:
    initializer = nest.map_structure(lambda el: el[0], elems)
    elems_flat = [el[1:] for el in elems_flat]
  if len({len(el) for el in elems_flat}) != 1:
    raise ValueError(
        'Mismatched element sizes: {}'.format(nest.map_structure(len, elems)))
  carry = initializer
  for el in zip(*elems_flat):
    carry = fn(carry, nest.pack_sequence_as(elems, el))
  return carry


def _map_fn(  # pylint: disable=unused-argument
    fn,
    elems,
    dtype=None,
    parallel_iterations=None,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    name=None,
    fn_output_signature=None):
  """Numpy implementation of tf.map_fn."""
  if fn_output_signature is not None and nest.is_nested(fn_output_signature):
    # If fn returns a tuple, then map_fn returns a tuple as well; and similarly
    # for lists and more complex nestings.  We do not support this behavior at
    # this time, so we raise an error explicitly instead of silently doing the
    # wrong thing.
    raise NotImplementedError
  if JAX_MODE:
    from jax import tree_util  # pylint: disable=g-import-not-at-top
    elems_flat, in_tree = tree_util.tree_flatten(elems)
    elems_zipped = zip(*elems_flat)
    def func(flat_args):
      unflat_args = tree_util.tree_unflatten(in_tree, flat_args)
      return fn(unflat_args)
    return np.stack([func(x) for x in elems_zipped])

  if isinstance(elems, np.ndarray):
    return np.array([fn(x) for x in elems])

  # In the NumPy backend, we do not yet support map_fn over lists, tuples, or
  # other structures.
  raise NotImplementedError


def _vectorized_map(fn, elems):
  """Numpy implementation of tf.vectorized_map."""
  if JAX_MODE:
    from jax import vmap  # pylint: disable=g-import-not-at-top
    return vmap(fn)(elems)

  # In the NumPy backend, we don't actually vectorize.
  return _map_fn(fn, elems)


def _scan(  # pylint: disable=unused-argument
    fn,
    elems,
    initializer=None,
    parallel_iterations=10,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    reverse=False,
    name=None):
  """Scan implementation."""

  if reverse:
    elems = nest.map_structure(lambda x: x[::-1], elems)

  if initializer is None:
    if nest.is_nested(elems):
      raise NotImplementedError
    initializer = elems[0]
    elems = elems[1:]
    prepend = [[initializer]]
  else:
    prepend = None

  def func(arg, x):
    return nest.flatten(fn(nest.pack_sequence_as(initializer, arg),
                           nest.pack_sequence_as(elems, x)))

  arg = nest.flatten(initializer)
  if JAX_MODE:
    from jax import lax  # pylint: disable=g-import-not-at-top
    def scan_body(arg, x):
      arg = func(arg, x)
      return arg, arg
    _, out = lax.scan(scan_body, arg, nest.flatten(elems))
  else:
    out = [[] for _ in range(len(arg))]
    for x in zip(*nest.flatten(elems)):
      arg = func(arg, x)
      for i, z in enumerate(arg):
        out[i].append(z)

  if prepend is not None:
    out = [pre + list(o) for (pre, o) in zip(prepend, out)]

  ordering = (lambda x: x[::-1]) if reverse else (lambda x: x)
  return nest.pack_sequence_as(
      initializer, [ordering(np.array(o)) for o in out])


# --- Begin Public Functions --------------------------------------------------


foldl = utils.copy_docstring(
    'tf.foldl',
    _foldl_jax if JAX_MODE else _foldl)

map_fn = utils.copy_docstring(
    'tf.map_fn',
    _map_fn)

vectorized_map = utils.copy_docstring(
    'tf.vectorized_map',
    _vectorized_map)


def pfor(fn, n):
  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top
    return jax.vmap(fn)(np.arange(n))
  outs = [fn(i) for i in range(n)]
  flat_outs = [nest.flatten(o) for o in outs]
  return nest.pack_sequence_as(
      outs[0], [np.array(o) for o in zip(*flat_outs)])


scan = utils.copy_docstring(
    'tf.scan',
    _scan)

