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
"""Tests for `random_variable`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util

tfd = tfp.distributions
tfn = tfp.experimental.nn


@test_util.test_all_tf_execution_regimes
class RandomVariableTest(test_util.TestCase):

  def test_default_arguments(self):
    x = tfn.util.RandomVariable(tfd.Normal(0, 1))
    x1 = tf.convert_to_tensor(x)
    x2 = tf.convert_to_tensor(x)
    x.reset()
    x3 = tf.convert_to_tensor(x)
    [x1_, x2_, x3_] = self.evaluate([x1, x2, x3])
    self.assertAllEqual(x1_, x2_)
    self.assertNotEqual(x1_, x3_)
    self.assertEqual(tf.float32, x.dtype)
    self.assertEqual(None, x.shape)
    self.assertStartsWith(x.name, 'Normal')

  def test_non_default_arguments(self):
    x = tfn.util.RandomVariable(
        tfd.Bernoulli(probs=[[0.25], [0.5]]),
        tfd.Distribution.mean,
        dtype=tf.float32,
        shape=[2, None],
        name='custom')
    x_ = self.evaluate(x + 0.)
    self.assertAllEqual([[0.25], [0.5]], x_)
    self.assertEqual(tf.float32, x.dtype)
    self.assertAllEqual([2, None], tensorshape_util.as_list(x.shape))
    self.assertEqual('custom', x.name)

  def test_set_shape(self):
    x = tfn.util.RandomVariable(
        tfd.Bernoulli(probs=[[0.25], [0.5]]),
        tfd.Distribution.mean,
        dtype=tf.float32,
        shape=[2, None],
        name='custom')
    self.assertAllEqual([2, None], tensorshape_util.as_list(x.shape))
    x.set_shape([None, 1])
    self.assertAllEqual([2, 1], tensorshape_util.as_list(x.shape))

  def test_non_xla_graph_throws_exception(self):
    if tf.config.functions_run_eagerly():
      self.skipTest('Graph mode test only.')
    x = tfn.util.RandomVariable(tfd.Normal(0, 1))
    @tf.function(autograph=False, experimental_compile=True)
    def run():
      tf.convert_to_tensor(x)
    run()
    with self.assertRaisesRegexp(ValueError, r'different graph context'):
      tf.convert_to_tensor(x)

  def test_nested_graphs(self):
    if tf.config.functions_run_eagerly():
      self.skipTest('Graph mode test only.')
    x = tfn.util.RandomVariable(
        tfd.Normal(0, 1), tfd.Distribution.mean, name='rv')
    @tf.function(autograph=False, experimental_compile=True)
    def run():
      @tf.function(autograph=False, experimental_compile=True)
      def _inner():
        return tf.convert_to_tensor(x, name='inner')
      y = _inner()
      z = tf.convert_to_tensor(x, name='outer')
      return y, z
    with self.assertRaisesRegexp(ValueError, r'different graph context'):
      run()


if __name__ == '__main__':
  tf.test.main()
