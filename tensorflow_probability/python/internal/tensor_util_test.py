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
"""Tests for tensor_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf

import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions


class FakeModule(tf.Module):

  def __init__(self, x, name=None):
    super(FakeModule, self).__init__(name)
    self.x = tensor_util.convert_nonref_to_tensor(x)

  @property
  def dtype(self):
    return tf.as_dtype(self.x.dtype)

  @property
  def shape(self):
    return tf.TensorShape(self.x.shape)


tf.register_tensor_conversion_function(
    base_type=FakeModule,
    conversion_func=lambda d, *_, **__: tf.convert_to_tensor(d.x))


@test_util.test_all_tf_execution_regimes
class ConvertNonrefToTensorTest(test_util.TestCase):

  def test_np_object(self):
    x = np.array(0.)
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIsInstance(y, tf.Tensor)
    self.assertEqual(x, self.evaluate(y))

  def test_tf_tensor(self):
    x = tf.constant(1.)
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIs(x, y)

  def test_tf_variable(self):
    x = tf.Variable(1., trainable=True)
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIs(x, y)
    x = tf.Variable(1., trainable=False)
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIs(x, y)

  def test_tf_module(self):
    x = FakeModule(np.array(1.))
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIs(x, y)

    x = FakeModule(tf.Variable(2., trainable=True))
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIs(x, y)

    x = FakeModule(tf.Variable(2., trainable=False))
    y = tensor_util.convert_nonref_to_tensor(x)
    self.assertIs(x, y)

  def test_end_to_end(self):
    x = tf.constant(-0.5)
    d = tfd.Normal(loc=0., scale=tfp.util.DeferredTensor(x, tf.math.exp))
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(x)
      negloglik = -d.log_prob(0.)
    g = tape.gradient(negloglik, [x])
    self.assertAllNotNone(g)
    self.assertEqual([1.], self.evaluate(g))


@test_util.test_all_tf_execution_regimes
class IsRefTest(test_util.TestCase):

  def test_various_types(self):
    self.assertFalse(tensor_util.is_ref(0.))
    self.assertFalse(tensor_util.is_ref([tf.Variable(0.)]))  # Note!
    self.assertFalse(tensor_util.is_ref(np.array(0., np.float32)))
    self.assertFalse(tensor_util.is_ref(tf.constant(0.)))
    self.assertTrue(tensor_util.is_ref(FakeModule(0.)))
    self.assertTrue(tensor_util.is_ref(FakeModule(tf.Variable(0.))))
    self.assertTrue(tensor_util.is_ref(tf.Variable(0.)))


@test_util.test_all_tf_execution_regimes
class VariableTrackingUtils(test_util.TestCase):

  def test_discover_trainable_variables(self):
    expected_vars = (
        tf.Variable(0., name='a'),
        tf.Variable(1., name='b'),
        tf.Variable(2., name='d'),
    )
    input_ = [
        tf.constant(1),
        expected_vars[0],
        (
            tf.constant(2),
            expected_vars[1],
            {
                'c': tf.constant(3),
                'd': expected_vars[2],
                'e': tf.Variable(3., name='e', trainable=False),
            },
        ),
    ]
    actual_vars = tensor_util.discover_trainable_variables(input_)
    self.assertAllIs(expected_vars, actual_vars)

  def test_discover_variables(self):
    expected_vars = (
        tf.Variable(0., name='a'),
        tf.Variable(1., name='b'),
        tf.Variable(2., name='d'),
        tf.Variable(3., name='e', trainable=False),
    )
    input_ = [
        tf.constant(1),
        expected_vars[0],
        (
            tf.constant(2),
            expected_vars[1],
            {
                'c': tf.constant(3),
                'd': expected_vars[2],
                'e': expected_vars[3],
            },
        ),
    ]
    actual_vars = tensor_util.discover_variables(input_)
    self.assertAllIs(expected_vars, actual_vars)

  def test_is_variable(self):
    self.assertTrue(tensor_util.is_variable(tf.Variable(0.)))
    self.assertTrue(tensor_util.is_variable(tf.Variable(0., trainable=False)))

  def test_is_trainable_variable(self):
    self.assertTrue(tensor_util.is_trainable_variable(tf.Variable(0.)))
    self.assertFalse(tensor_util.is_trainable_variable(
        tf.Variable(0., trainable=False)))

  def test_is_module(self):
    m = FakeModule(1.)
    self.assertTrue(tensor_util.is_module(m))
    self.assertFalse(tensor_util.is_module(tf.Variable(0.)))


if __name__ == '__main__':
  tf.test.main()
