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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions
tfn = tfp.experimental.nn


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

  def eval(self, x, is_training=True):
    if (isinstance(x, tfd.Independent) and
        isinstance(x.distribution, tfd.Normal)):
      x = x.distribution.loc
    else:
      x = tf.convert_to_tensor(x, dtype_hint=tf.float32, name='x')
    y = self.bias + tf.matmul(x, self.kernel)
    self._set_extra_loss(tf.norm(y, axis=-1))
    self._set_extra_result(tf.shape(x))
    return tfd.Independent(tfd.Normal(loc=y, scale=1),
                           reinterpreted_batch_ndims=1)


@test_util.test_all_tf_execution_regimes
class LayerTest(test_util.TestCase):

  def test_correctly_returns_non_tensor(self):
    input_size = 3
    output_size = 5
    f = AffineMeanFieldNormal(output_size, input_size)
    x = tf.zeros([2, 1, input_size])
    y = f(x)
    self.assertIsInstance(y, tfd.Independent)
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
    self.assertIsInstance(y, tfd.Independent)
    self.assertAllEqual((2, output_size), y.distribution.loc.shape)

    extra_loss = [l.extra_loss for l in model.layers
                  if getattr(l, 'extra_loss', None) is not None]
    extra_result = [l.extra_result for l in model.layers
                    if getattr(l, 'extra_result', None) is not None]

    self.assertIsNone(model.extra_result)
    self.assertAllEqual([(2,), (2,)], [x.shape for x in extra_loss])

    extra_loss_, extra_result_, model_extra_loss_ = self.evaluate([
        extra_loss, extra_result, model.extra_loss])
    self.assertAllGreaterEqual(extra_loss_, 0.)
    self.assertAllEqual([[2, 3], [2, 5]], extra_result_)
    self.assertAllClose(sum(extra_loss_), model_extra_loss_,
                        rtol=1e-3, atol=1e-3)

  def test_summary(self):
    model = tfn.Sequential([
        lambda x: tf.reshape(x, [-1, 3]),
        AffineMeanFieldNormal(output_size=5, input_size=3),
        AffineMeanFieldNormal(output_size=2, input_size=5),
    ])
    self.assertEqual('trainable size: 32  /  0.000 MiB  /  {float32: 32}',
                     model.summary().split('\n')[-1])


@test_util.test_all_tf_execution_regimes
class LambdaTest(test_util.TestCase):

  def test_basic(self):
    shift = tf.Variable(1.)
    scale = tfp.util.TransformedVariable(1., tfb.Exp())
    f = tfn.Lambda(
        eval_fn=lambda x: tfd.Normal(loc=x + shift, scale=scale),
        extra_loss_fn=lambda x: tf.norm(x.loc),
        # `scale` will be tracked through the distribution but not `shift`.
        also_track=shift)
    x = tf.zeros([1, 2])
    y = f(x)
    self.assertIsInstance(y, tfd.Normal)
    self.assertLen(f.trainable_variables, 2)
    if tf.executing_eagerly():
      # We want to specifically check the values when in eager mode to ensure
      # we're not leaking graph tensors. The precise value doesn't matter.
      self.assertGreaterEqual(f.extra_loss, 0.)
    self.assertIsNone(f.extra_result)


if __name__ == '__main__':
  tf.test.main()
