# Copyright 2019 The TensorFlow Probability Authors.
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
"""Tests for layers."""

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.experimental import nn as tfn
from tensorflow_probability.python.internal import test_util


class AffineMeanFieldNormal(tfn.Layer):
  """This interesting layer produces an MVNDiag via affine transformation."""

  def __init__(self, output_size, input_size, dtype=tf.float32, name=None):
    super(AffineMeanFieldNormal, self).__init__(name=name)
    self._kernel = tf.Variable(
        tfn.initializers.glorot_normal()([input_size, output_size], dtype),
        name='kernel')
    self._bias = tf.Variable(
        tfn.initializers.glorot_normal()([output_size], dtype),
        name='bias')

  @property
  def kernel(self):
    return self._kernel

  @property
  def bias(self):
    return self._bias

  def __call__(self, x):
    if (isinstance(x, independent.Independent) and
        isinstance(x.distribution, normal.Normal)):
      x = x.distribution.loc
    else:
      x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
    y = self.bias + tf.matmul(x, self.kernel)
    return independent.Independent(
        normal.Normal(loc=y, scale=1), reinterpreted_batch_ndims=1)


@test_util.test_all_tf_execution_regimes
class LayerTest(test_util.TestCase):

  def test_correctly_returns_non_tensor(self):
    input_size = 3
    output_size = 5
    f = AffineMeanFieldNormal(output_size, input_size)
    x = tf.zeros([2, 1, input_size])
    y = f(x)
    self.assertIsInstance(y, independent.Independent)
    self.assertAllEqual((2, 1, output_size), y.distribution.loc.shape)


@test_util.test_all_tf_execution_regimes
class SequentialTest(test_util.TestCase):

  def test_works_correctly(self):
    input_size = 3
    output_size = 5
    model = tfn.Sequential([
        lambda x: tf.reshape(x, [-1, input_size]),
        AffineMeanFieldNormal(output_size=5, input_size=input_size),
        AffineMeanFieldNormal(output_size=output_size, input_size=5),
    ])
    self.assertLen(model.trainable_variables, 4)
    self.evaluate([v.initializer for v in model.trainable_variables])
    self.assertLen(model.layers, 3)
    self.assertEqual(
        '<Sequential: name=lambda__AffineMeanFieldNormal_AffineMeanFieldNormal>',
        str(model))

    x = tf.zeros([2, 1, input_size])
    y = model(x)
    self.assertIsInstance(y, independent.Independent)
    self.assertAllEqual((2, output_size), y.distribution.loc.shape)

  def test_summary(self):
    model = tfn.Sequential([
        lambda x: tf.reshape(x, [-1, 3]),
        AffineMeanFieldNormal(output_size=5, input_size=3),
        AffineMeanFieldNormal(output_size=2, input_size=5),
    ])
    self.assertEqual('trainable size: 32  /  0.000 MiB  /  {float32: 32}',
                     model.summary().split('\n')[-1])


@test_util.test_all_tf_execution_regimes
class KernelBiasLayerTest(test_util.TestCase):

  def test_works_correctly(self):
    pass


if __name__ == '__main__':
  test_util.main()
