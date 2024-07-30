# Copyright 2021 The TensorFlow Probability Authors.
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
"""TensorFlow backend."""

import types
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import prefer_static
from fun_mc.dynamic.backend_tensorflow import util

tnp = tf.experimental.numpy

_lax = types.ModuleType('lax')
_lax.cond = tf.cond
_lax.stop_gradient = tf.stop_gradient

_nn = types.ModuleType('nn')
_nn.softmax = tf.nn.softmax
_nn.one_hot = tf.one_hot


class _ShapeDtypeStruct:
  pass


jax = types.ModuleType('jax')
jax.ShapeDtypeStruct = _ShapeDtypeStruct
jax.jit = tf.function
jax.lax = _lax
jax.custom_gradient = tf.custom_gradient
jax.nn = _nn


class _JNP(types.ModuleType):

  def __getattr__(self, name):
    return getattr(tnp, name)


jnp = _JNP('numpy')
jnp.dtype = tf.DType
# These are technically provided by TensorFlow, but only after numpy mode is
# enabled.
jnp.ndarray = tf.Tensor
jnp.float32 = tf.float32
jnp.float64 = tf.float64


__all__ = [
    'distribute_lib',
    'prefer_static',
    'jnp',
    'jax',
    'tfp',
    'util',
]
