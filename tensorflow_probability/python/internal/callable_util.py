# Copyright 2020 The TensorFlow Probability Authors.
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
"""Utilities for handling Python callables."""

import tensorflow.compat.v2 as tf

JAX_MODE = False
NUMPY_MODE = False


def get_output_spec(fn, *args, **kwargs):
  """Traces a callable to determine shape and dtype of its return value(s).

  Tracing incurs the Python execution cost of running the callable to build a
  temporary graph, but does not execute any Tensor operations and has no
  side effects in the calling graph. It is supported only in TensorFlow and
  JAX. Calls to this method using the Numpy backend may incur the full execution
  cost of `fn`.

  Args:
    fn: Python `callable` accepting (structures of) `Tensor` arguments and
      returning (structures) of `Tensor`s.
    *args: `Tensor` and/or `tf.TensorSpec` instances representing positional
      arguments to `fn`.
    **kwargs: `Tensor` and/or `tf.TensorSpec` instances representing named
      arguments to `fn`.
  Returns:
    structured_outputs: Object or structure of objects corresponding to the
      value(s) returned by `fn`. These objects have `.shape` and
      `.dtype` attributes; nothing else about them is guaranteed by the API.
  """

  if NUMPY_MODE:
    def _as_tensor(t):
      if isinstance(t, tf.TensorSpec):
        return tf.zeros(t.shape, dtype=t.dtype)
      return t
    args, kwargs = tf.nest.map_structure(_as_tensor, (args, kwargs))
    return fn(*args, **kwargs)

  if JAX_MODE:
    import jax  # pylint: disable=g-import-not-at-top
    return jax.eval_shape(fn, *args, **kwargs)

  def _as_tensor_spec(t):
    if isinstance(t, tf.TensorSpec):
      return t
    return tf.TensorSpec.from_tensor(tf.convert_to_tensor(t))
  arg_specs, kwarg_specs = tf.nest.map_structure(_as_tensor_spec,
                                                 (args, kwargs))
  return tf.function(fn, autograph=False).get_concrete_function(
      *arg_specs, **kwarg_specs).structured_outputs

