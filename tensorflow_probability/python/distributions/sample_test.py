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
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class SampleDistributionTest(test_util.TestCase):

  def test_everything_scalar(self):
    s = tfd.Sample(tfd.Normal(loc=0, scale=1), 5, validate_args=True)
    x = s.sample(seed=test_util.test_seed())
    actual_lp = s.log_prob(x)
    # Sample.log_prob will reduce over event space, ie, dims [0, 2]
    # corresponding to sizes concat([[5], [2]]).
    expected_lp = tf.reduce_sum(s.distribution.log_prob(x), axis=0)
    x_, actual_lp_, expected_lp_ = self.evaluate([x, actual_lp, expected_lp])
    self.assertEqual((5,), x_.shape)
    self.assertEqual((), actual_lp_.shape)
    self.assertAllClose(expected_lp_, actual_lp_, atol=0, rtol=1e-3)

  def test_everything_nonscalar(self):
    s = tfd.Sample(
        tfd.Independent(tfd.Normal(loc=tf.zeros([3, 2]), scale=1), 1), [5, 4],
        validate_args=True)
    x = s.sample([6, 1], seed=test_util.test_seed())
    actual_lp = s.log_prob(x)
    # Sample.log_prob will reduce over event space, ie, dims [2, 3, 5]
    # corresponding to sizes concat([[5, 4], [2]]).
    expected_lp = tf.reduce_sum(
        s.distribution.log_prob(tf.transpose(a=x, perm=[0, 1, 3, 4, 2, 5])),
        axis=[2, 3])
    x_, actual_lp_, expected_lp_ = self.evaluate([x, actual_lp, expected_lp])
    self.assertEqual((6, 1, 3, 5, 4, 2), x_.shape)
    self.assertEqual((6, 1, 3), actual_lp_.shape)
    self.assertAllClose(expected_lp_, actual_lp_, atol=0, rtol=1e-3)

  def test_mixed_scalar(self):
    s = tfd.Sample(tfd.Independent(tfd.Normal(loc=[0., 0], scale=1), 1),
                   3, validate_args=False)
    x = s.sample(4, seed=test_util.test_seed())
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
      return tf.reduce_sum(mvn.log_prob(x) - fldj, axis=1)

    # Transform a Sample.
    d = tfd.TransformedDistribution(
        tfd.Sample(mvn, sample_shape, validate_args=True),
        bijector=aff)
    y = d.sample(4, seed=test_util.test_seed())
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
    y = d.sample(4, seed=test_util.test_seed())
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
      return tf.reduce_sum(mvn.log_prob(x) - fldj, axis=1)

    # Transform a Sample.
    d = tfd.TransformedDistribution(
        tfd.Sample(mvn, sample_shape, validate_args=True),
        bijector=exp)
    y = d.sample(4, seed=test_util.test_seed())
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
    y = d.sample(4, seed=test_util.test_seed())
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
    expected_entropy = 12 * tf.reduce_sum(mvn.distribution.entropy(), axis=-1)
    actual_entropy = d.entropy()
    self.assertAllEqual(*self.evaluate([expected_entropy, actual_entropy]))

  @test_util.tf_tape_safety_test
  def test_gradients_through_params(self):
    loc = tf.Variable(tf.zeros([4, 5, 3]), shape=tf.TensorShape(None))
    scale = tf.Variable(tf.ones([]), shape=tf.TensorShape(None))
    # In real life, you'd really always want `sample_shape` to be
    # `trainable=False`.
    sample_shape = tf.Variable([1, 2], shape=tf.TensorShape(None))
    dist = tfd.Sample(
        tfd.Independent(tfd.Logistic(loc=loc, scale=scale),
                        reinterpreted_batch_ndims=1),
        sample_shape=sample_shape,
        validate_args=True)
    with tf.GradientTape() as tape:
      loss = -dist.log_prob(0.)
    self.assertLen(dist.trainable_variables, 3)
    grad = tape.gradient(loss, [loc, scale, sample_shape])
    self.assertAllNotNone(grad[:-1])
    self.assertIs(grad[-1], None)

  @test_util.tf_tape_safety_test
  def test_variable_shape_change(self):
    loc = tf.Variable(tf.zeros([4, 5, 3]), shape=tf.TensorShape(None))
    scale = tf.Variable(tf.ones([]), shape=tf.TensorShape(None))
    # In real life, you'd really always want `sample_shape` to be
    # `trainable=False`.
    sample_shape = tf.Variable([1, 2], shape=tf.TensorShape(None))
    dist = tfd.Sample(
        tfd.Independent(tfd.Logistic(loc=loc, scale=scale),
                        reinterpreted_batch_ndims=1),
        sample_shape=sample_shape,
        validate_args=True)
    self.evaluate([v.initializer for v in dist.trainable_variables])

    x = dist.mean()
    y = dist.sample([7, 2], seed=test_util.test_seed())
    loss_x = -dist.log_prob(x)
    loss_0 = -dist.log_prob(0.)
    batch_shape = dist.batch_shape_tensor()
    event_shape = dist.event_shape_tensor()
    [x_, y_, loss_x_, loss_0_, batch_shape_, event_shape_] = self.evaluate([
        x, y, loss_x, loss_0, batch_shape, event_shape])
    self.assertAllEqual([4, 5, 1, 2, 3], x_.shape)
    self.assertAllEqual([7, 2, 4, 5, 1, 2, 3], y_.shape)
    self.assertAllEqual([4, 5], loss_x_.shape)
    self.assertAllEqual([4, 5], loss_0_.shape)
    self.assertAllEqual([4, 5], batch_shape_)
    self.assertAllEqual([1, 2, 3], event_shape_)
    self.assertLen(dist.trainable_variables, 3)

    with tf.control_dependencies([
        loc.assign(tf.zeros([])),
        scale.assign(tf.ones([3, 1, 2])),
        sample_shape.assign(6),
    ]):
      x = dist.mean()
      y = dist.sample([7, 2], seed=test_util.test_seed())
      loss_x = -dist.log_prob(x)
      loss_0 = -dist.log_prob(0.)
      batch_shape = dist.batch_shape_tensor()
      event_shape = dist.event_shape_tensor()
    [x_, y_, loss_x_, loss_0_, batch_shape_, event_shape_] = self.evaluate([
        x, y, loss_x, loss_0, batch_shape, event_shape])
    self.assertAllEqual([3, 1, 6, 2], x_.shape)
    self.assertAllEqual([7, 2, 3, 1, 6, 2], y_.shape)
    self.assertAllEqual([3, 1], loss_x_.shape)
    self.assertAllEqual([3, 1], loss_0_.shape)
    self.assertAllEqual([3, 1], batch_shape_)
    self.assertAllEqual([6, 2], event_shape_)
    self.assertLen(dist.trainable_variables, 3)

  def test_variable_sample_shape_exception(self):
    loc = tf.Variable(tf.zeros([4, 5, 3]), shape=tf.TensorShape(None))
    scale = tf.Variable(tf.ones([]), shape=tf.TensorShape(None))
    sample_shape = tf.Variable([[1, 2]], shape=tf.TensorShape(None))
    with self.assertRaisesWithPredicateMatch(
        Exception,
        'Argument `sample_shape` must be either a scalar or a vector.'):
      dist = tfd.Sample(
          tfd.Independent(tfd.Logistic(loc=loc, scale=scale),
                          reinterpreted_batch_ndims=1),
          sample_shape=sample_shape,
          validate_args=True)
      self.evaluate([v.initializer for v in dist.trainable_variables])
      self.evaluate(dist.mean())


if __name__ == '__main__':
  tf.test.main()
