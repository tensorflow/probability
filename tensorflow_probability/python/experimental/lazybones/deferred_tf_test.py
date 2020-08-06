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
# Lint as: python3
"""Tests for tensorflow_probability.experimental.lazybones.deferred."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

lb = tfp.experimental.lazybones


@test_util.test_all_tf_execution_regimes
class DeferredTFTest(test_util.TestCase):

  def test_tf(self):
    tfw = lb.DeferredInput(tf)  # Neat!
    a = tfw.convert_to_tensor([[1.], [2]])
    x = tfw.constant([[1., 2.], [2., 3.]])
    b = tfw.matmul(x, a)
    c = tfw.reduce_sum(b)

    self.assertTrue((b.shape == (2, 1)).eval())
    with lb.DeferredScope():
      b.value = tf.constant([4., 5, 6.])
      self.assertEqual(15, self.evaluate(c.eval()))
    self.assertEqual(13, self.evaluate(c.eval()))

    x.value = tf.ones((3, 2))
    self.assertEqual((3, 1), b.shape.eval())
    self.assertEqual(9, self.evaluate(c.eval()))

  def test_tfd_Normal(self):
    tfd = lb.DeferredInput(tfp.distributions)
    tfw = lb.DeferredInput(tf)
    loc = tfw.constant([1., 100.])
    scale = tfw.Variable(1.)
    self.evaluate([scale.initializer.eval()])

    dist = tfd.Normal(loc, scale)
    sample = dist.sample(5000, seed=test_util.test_seed())
    sample_mean = tfw.reduce_mean(sample, axis=0)
    sample_std = tfw.math.reduce_std(sample, axis=0)
    [sample_mean_, sample_std_] = self.evaluate(lb.DeferredInput([
        sample_mean, sample_std]).eval())
    self.assertAllClose([1., 100.], sample_mean_, rtol=.05)
    self.assertAllClose([1., 1.], sample_std_, rtol=.12)

    loc.value = tf.constant([100., 50., 1.])
    # TODO(jvdillon): Fix collision for TF graph's `tf.Variable.eval`.
    scale.value = scale.assign(3.).eval()
    self.assertEqual((5000, 3), sample.shape.eval())
    [sample_mean_, sample_std_] = self.evaluate(lb.DeferredInput([
        sample_mean, sample_std]).eval())
    self.assertAllClose([100., 50., 1.], sample_mean_, rtol=.12)
    self.assertAllClose([3., 3., 3.], sample_std_, rtol=.12)

    mu = tfw.constant(10.)
    sigma = tfw.constant(1.)
    prior_dist = tfd.Normal(mu, sigma)
    x_scale = tfw.constant(2.)
    x = tfd.Normal(prior_dist.sample(5000, seed=test_util.test_seed()), x_scale)
    x_sample = x.sample(seed=test_util.test_seed())

    sample_mean = tfw.reduce_mean(x_sample)
    expected_std = tfw.sqrt(sigma**2 + x_scale**2)
    sample_std = tfw.math.reduce_std(x_sample)
    [
        expected_mean_,
        sample_mean_,
        expected_std_,
        sample_std_,
    ] = self.evaluate(lb.DeferredInput([
        mu, sample_mean, expected_std, sample_std]).eval())
    self.assertAllClose(expected_mean_, sample_mean_, rtol=.05)
    self.assertAllClose(expected_std_, sample_std_, rtol=.26)

    mu.value = tf.constant(3.)
    sigma.value = tf.constant(3.)
    x_scale.value = tf.constant(1.)
    sample_mean = tfw.reduce_mean(x_sample)
    expected_std = tfw.sqrt(sigma**2 + x_scale**2)
    sample_std = tfw.math.reduce_std(x_sample)
    [
        expected_mean_,
        sample_mean_,
        expected_std_,
        sample_std_,
    ] = self.evaluate(lb.DeferredInput([
        mu, sample_mean, expected_std, sample_std]).eval())
    self.assertAllClose(expected_mean_, sample_mean_, rtol=.05)
    self.assertAllClose(expected_std_, sample_mean_, rtol=.12)

  def test_tf_stack(self):
    tfw = lb.DeferredInput(tf)
    x = tfw.stack([lb.DeferredInput(tfw.constant(1.)), 2.])
    self.assertAllEqual([1., 2.], self.evaluate(x.eval()))

  def test_tf_unstack(self):
    tfw = lb.DeferredInput(tf)
    x, y = tfw.unstack([lb.DeferredInput(tfw.constant(1.)), 2.],
                       _static_iter_len=2)
    x_, y_ = self.evaluate(lb.DeferredInput([x, y]).eval())
    self.assertEqual(1., x_)
    self.assertEqual(2., y_)
    z = tfw.unstack([lb.DeferredInput(tfw.constant(1.)), 2.],
                    _static_iter_len=2)
    self.assertAllEqual([1., 2.], self.evaluate(z.eval()))
    with self.assertRaisesRegex(ValueError, r'values to unpack'):
      x, y, z = tfw.unstack([lb.DeferredInput(tfw.constant(1.)), 2.],
                            _static_iter_len=2)


if __name__ == '__main__':
  tf.test.main()
