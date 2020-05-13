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
"""Numpy implementations of TensorFlow top-level control flow functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow_probability.python.internal.backend.numpy import _utils as utils
from tensorflow_probability.python.internal.backend.numpy import ops


__all__ = [
    'cond',
    'no_op',
    'while_loop',
    # 'case',
    # 'dynamic_partition',
    # 'dynamic_stitch',
    # 'map_fn',
    # 'scan',
]


JAX_MODE = False


def _cond_jax(pred, true_fn=None, false_fn=None, name=None):  # pylint: disable=missing-docstring
  from jax import lax  # pylint: disable=g-import-not-at-top

  del name
  def overridden_true_fn(x):
    del x
    return true_fn()

  def overridden_false_fn(x):
    del x
    return false_fn()
  return lax.cond(pred, None, overridden_true_fn, None, overridden_false_fn)


def _cond(pred, true_fn=None, false_fn=None, name=None):
  del name
  return true_fn() if pred else false_fn()


def _no_op(name=None):  # pylint: disable=unused-argument
  pass


def _while_loop(cond, body, loop_vars,  # pylint: disable=redefined-outer-name
                shape_invariants=None, parallel_iterations=10,  # pylint: disable=unused-argument
                back_prop=True, swap_memory=False,  # pylint: disable=unused-argument
                maximum_iterations=None, name=None):  # pylint: disable=unused-argument
  """Numpy implementation of `tf.while_loop`."""
  i = 0
  while (cond(*loop_vars) and
         (maximum_iterations is None or i < maximum_iterations)):
    loop_vars = body(*loop_vars)
    i += 1
  return loop_vars


def _while_loop_jax(cond, body, loop_vars,  # pylint: disable=redefined-outer-name
                    shape_invariants=None, parallel_iterations=10,  # pylint: disable=unused-argument
                    back_prop=True, swap_memory=False,  # pylint: disable=unused-argument
                    maximum_iterations=None, name=None):  # pylint: disable=unused-argument
  """Jax implementation of `tf.while_loop`."""
  from jax import lax  # pylint: disable=g-import-not-at-top
  if maximum_iterations is None:
    def override_body_fn(args):
      return body(*args)
    def override_cond_fn(args):
      return cond(*args)
    return lax.while_loop(override_cond_fn, override_body_fn, loop_vars)
  elif back_prop:
    def override_body_fn(args, _):
      c = cond(*args)
      sc = ops.get_static_value(c)
      if sc is None:
        args = lax.cond(c, args, lambda args: body(*args), args,
                        lambda args: args)
      elif sc:
        args = body(*args)
      return args, ()

    loop_vars, _ = lax.scan(
        override_body_fn, loop_vars, xs=None, length=maximum_iterations)
    return loop_vars
  else:
    def override_body_fn(args):
      i, args = args
      return i + 1, body(*args)
    def override_cond_fn(args):
      i, args = args
      return cond(*args) & (i < maximum_iterations)
    return lax.while_loop(
        override_cond_fn, override_body_fn, (np.array(0), loop_vars))[1]


# --- Begin Public Functions --------------------------------------------------

cond = utils.copy_docstring(
    'tf.cond',
    _cond_jax if JAX_MODE else _cond)

no_op = utils.copy_docstring(
    'tf.no_op',
    _no_op)

while_loop = utils.copy_docstring(
    'tf.while_loop',
    _while_loop_jax if JAX_MODE else _while_loop)
