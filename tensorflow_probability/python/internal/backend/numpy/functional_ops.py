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

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'map_fn',
    'scan',
]


JAX_MODE = False


def _map_fn(  # pylint: disable=unused-argument
    fn,
    elems,
    dtype=None,
    parallel_iterations=None,
    back_prop=True,
    swap_memory=False,
    infer_shape=True,
    name=None):
  """Numpy implementation of tf.map_fn."""
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
  out = []
  if initializer is None:
    arg = elems[0]
    elems = elems[1:]
  else:
    arg = initializer

  for x in elems:
    arg = fn(arg, x)
    out.append(arg)
  return np.array(out)


# --- Begin Public Functions --------------------------------------------------


map_fn = utils.copy_docstring(
    tf.map_fn,
    _map_fn)

scan = utils.copy_docstring(
    tf.scan,
    _scan)

