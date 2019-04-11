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
"""Tests for the Sample distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.run_all_in_graph_and_eager_modes
class SampleDistributionTest(tf.test.TestCase, parameterized.TestCase):

  def test_everything_scalar(self):
    s = tfd.Sample(tfd.Normal(loc=0, scale=1), 5, validate_args=True)
    x = s.sample(seed=tfp_test_util.test_seed())
    actual_lp = s.log_prob(x)
    # Sample.log_prob will reduce over event space, ie, dims [0, 2]
    # corresponding to sizes concat([[5], [2]]).
    expected_lp = tf.reduce_sum(input_tensor=s.distribution.log_prob(x), axis=0)
    x_, actual_lp_, expected_lp_ = self.evaluate([x, actual_lp, expected_lp])
    self.assertEqual((5,), x_.shape)
    self.assertEqual((), actual_lp_.shape)
    self.assertAllClose(expected_lp_, actual_lp_, atol=0, rtol=1e-3)

  def test_everything_nonscalar(self):
    s = tfd.Sample(
        tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1), 1), [5, 4],
        validate_args=True)
    x = s.sample([6, 1], seed=tfp_test_util.test_seed())
    actual_lp = s.log_prob(x)
    # Sample.log_prob will reduce over event space, ie, dims [2, 3, 5]
    # corresponding to sizes concat([[5, 4], [2]]).
    expected_lp = tf.reduce_sum(
        input_tensor=s.distribution.log_prob(
            tf.transpose(a=x, perm=[0, 1, 3, 4, 2, 5])),
        axis=[2, 3])
    x_, actual_lp_, expected_lp_ = self.evaluate([x, actual_lp, expected_lp])
    self.assertEqual((6, 1, 3, 5, 4, 2), x_.shape)
    self.assertEqual((6, 1, 3), actual_lp_.shape)
    self.assertAllClose(expected_lp_, actual_lp_, atol=0, rtol=1e-3)

  def test_mixed_scalar(self):
    s = tfd.Sample(tfd.Independent(tfd.Normal(loc=[0., 0], scale=1), 1),
                   3, validate_args=False)
    x = s.sample(4, seed=tfp_test_util.test_seed())
    lp = s.log_prob(x)
    self.assertEqual((4, 3, 2), x.shape)
    self.assertEqual((4,), lp.shape)

  def test_kl_divergence(self):
    q_scale = 2.
    p = tfd.Sample(
        tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1), 1), [5, 4],
        validate_args=True)
    q = tfd.Sample(
        tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=2.), 1), [5, 4],
        validate_args=True)
    actual_kl = tfd.kl_divergence(p, q)
    expected_kl = ((5 * 4) *
                   (0.5 * q_scale**-2. - 0.5 + np.log(q_scale)) *  # Actual KL.
                   np.ones([3]) * 2)  # Batch, events.
    self.assertAllClose(expected_kl, self.evaluate(actual_kl))

  def test_transformed_affine(self):
    sample_shape = 3
    mvn = tfd.Independent(tfd.Normal(loc=[0., 0], scale=1), 1)
    aff = tfb.Affine(scale_tril=[[0.75, 0.],
                                 [0.05, 0.5]])

    def expected_lp(y):
      x = aff.inverse(y)  # Ie, tf.random.normal([4, 3, 2])
      fldj = aff.forward_log_det_jacobian(x, event_ndims=1)
      return tf.reduce_sum(input_tensor=mvn.log_prob(x) - fldj, axis=1)

    # Transform a Sample.
    d = tfd.TransformedDistribution(
        tfd.Sample(mvn, sample_shape, validate_args=True),
        bijector=aff)
    y = d.sample(4, seed=tfp_test_util.test_seed())
    actual_lp = d.log_prob(y)
    self.assertAllEqual((4,) + (sample_shape,) + (2,), y.shape)
    self.assertAllEqual((4,), actual_lp.shape)
    self.assertAllClose(
        *self.evaluate([expected_lp(y), actual_lp]),
        atol=0., rtol=1e-3)

    # Sample a Transform.
    d = tfd.Sample(
        tfd.TransformedDistribution(mvn, bijector=aff),
        sample_shape,
        validate_args=True)
    y = d.sample(4, seed=tfp_test_util.test_seed())
    actual_lp = d.log_prob(y)
    self.assertAllEqual((4,) + (sample_shape,) + (2,), y.shape)
    self.assertAllEqual((4,), actual_lp.shape)
    self.assertAllClose(
        *self.evaluate([expected_lp(y), actual_lp]),
        atol=0., rtol=1e-3)

  def test_transformed_exp(self):
    sample_shape = 3
    mvn = tfd.Independent(tfd.Normal(loc=[0., 0], scale=1), 1)
    exp = tfb.Exp()

    def expected_lp(y):
      x = exp.inverse(y)  # Ie, tf.random.normal([4, 3, 2])
      fldj = exp.forward_log_det_jacobian(x, event_ndims=1)
      return tf.reduce_sum(input_tensor=mvn.log_prob(x) - fldj, axis=1)

    # Transform a Sample.
    d = tfd.TransformedDistribution(
        tfd.Sample(mvn, sample_shape, validate_args=True),
        bijector=exp)
    y = d.sample(4, seed=tfp_test_util.test_seed())
    actual_lp = d.log_prob(y)
    self.assertAllEqual((4,) + (sample_shape,) + (2,), y.shape)
    self.assertAllEqual((4,), actual_lp.shape)
    # If `TransformedDistribution` didn't scale the jacobian by
    # `_sample_distribution_size`, then `scale_fldj` would need to be `False`.
    self.assertAllClose(
        *self.evaluate([expected_lp(y), actual_lp]),
        atol=0., rtol=1e-3)

    # Sample a Transform.
    d = tfd.Sample(
        tfd.TransformedDistribution(mvn, bijector=exp),
        sample_shape,
        validate_args=True)
    y = d.sample(4, seed=tfp_test_util.test_seed())
    actual_lp = d.log_prob(y)
    self.assertAllEqual((4,) + (sample_shape,) + (2,), y.shape)
    self.assertAllEqual((4,), actual_lp.shape)
    # Regardless of whether `TransformedDistribution` scales the jacobian by
    # `_sample_distribution_size`, `scale_fldj` is `True`.
    self.assertAllClose(
        *self.evaluate([expected_lp(y), actual_lp]),
        atol=0., rtol=1e-3)

  @parameterized.parameters(
      'mean',
      'stddev',
      'variance',
      'mode',
  )
  def test_summary_statistic(self, attr):
    sample_shape = [5, 4]
    mvn = tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1), 1)
    d = tfd.Sample(mvn, sample_shape, validate_args=True)
    self.assertEqual((3,), d.batch_shape)
    expected_stat = (
        getattr(mvn, attr)()[:, tf.newaxis, tf.newaxis, :] *
        tf.ones([3, 5, 4, 2]))
    actual_stat = getattr(d, attr)()
    self.assertAllEqual(*self.evaluate([expected_stat, actual_stat]))

  def test_entropy(self):
    sample_shape = [3, 4]
    mvn = tfd.Independent(tfd.Normal(loc=0, scale=[[0.25, 0.5]]), 1)
    d = tfd.Sample(mvn, sample_shape, validate_args=True)
    expected_entropy = 12 * tf.reduce_sum(
        input_tensor=mvn.distribution.entropy(),
        axis=-1)
    actual_entropy = d.entropy()
    self.assertAllEqual(*self.evaluate([expected_entropy, actual_entropy]))


if __name__ == '__main__':
  tf.test.main()
