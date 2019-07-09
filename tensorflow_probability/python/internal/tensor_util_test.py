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

from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


class FakeModule(tf.Module):

  def __init__(self, x, name=None):
    super(FakeModule, self).__init__(name)
    self.x = x

  @property
  def dtype(self):
    return tf.as_dtype(self.x.dtype)

  @property
  def shape(self):
    return tf.TensorShape(self.x.shape)


tf.register_tensor_conversion_function(
    base_type=FakeModule,
    conversion_func=lambda d, *_, **__: tf.convert_to_tensor(d.x))


@test_util.run_all_in_graph_and_eager_modes
class ConvertImmutableToTensorTest(tf.test.TestCase):

  def test_np_object(self):
    x = np.array(0.)
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIsInstance(y, tf.Tensor)
    self.assertEqual(x, self.evaluate(y))

  def test_tf_tensor(self):
    x = tf.constant(1.)
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIs(x, y)

  def test_tf_variable(self):
    x = tf.Variable(1., trainable=True)
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIs(x, y)
    x = tf.Variable(1., trainable=False)
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIs(x, y)

  def test_tf_module(self):
    x = FakeModule(1.)
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIsInstance(y, tf.Tensor)
    self.assertEqual(1., self.evaluate(y))

    x = FakeModule(tf.Variable(2., trainable=True))
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIs(x, y)

    x = FakeModule(tf.Variable(2., trainable=False))
    y = tensor_util.convert_immutable_to_tensor(x)
    self.assertIs(x, y)


class IsImmutableTest(tf.test.TestCase):

  def test_various_types(self):
    self.assertFalse(tensor_util.is_mutable(0.))
    self.assertFalse(tensor_util.is_mutable(FakeModule(0.)))
    self.assertFalse(tensor_util.is_mutable([tf.Variable(0.)]))  # Note!
    self.assertFalse(tensor_util.is_mutable(np.array(0., np.float32)))
    self.assertFalse(tensor_util.is_mutable(tf.constant(0.)))
    self.assertTrue(tensor_util.is_mutable(FakeModule(tf.Variable(0.))))
    self.assertTrue(tensor_util.is_mutable(tf.Variable(0.)))


if __name__ == '__main__':
  tf.test.main()
