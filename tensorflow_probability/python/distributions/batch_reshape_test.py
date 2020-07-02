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
"""Tests for BatchReshape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor


@test_util.test_all_tf_execution_regimes
class _BatchReshapeTest(object):

  def make_wishart(self, dims, new_batch_shape, old_batch_shape):
    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))

    scale = self.dtype([
        [[1., 0.5],
         [0.5, 1.]],
        [[0.5, 0.25],
         [0.25, 0.75]],
    ])
    scale = np.reshape(np.concatenate([scale, scale], axis=0),
                       old_batch_shape + [dims, dims])
    scale_ph = tf1.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    wishart = tfd.WishartTriL(
        df=5,
        scale_tril=DeferredTensor(scale_ph, tf.linalg.cholesky),
        validate_args=True)
    reshape_wishart = tfd.BatchReshape(
        distribution=wishart,
        batch_shape=new_batch_shape_ph,
        validate_args=True)

    return wishart, reshape_wishart

  def test_matrix_variate_sample_and_log_prob(self):
    if tf.executing_eagerly():
      # TODO(b/122840816): Modify this test so that it runs in eager mode or
      # document that the test is not intended to run in eager mode.
      return

    dims = 2
    seed = test_util.test_seed()
    new_batch_shape = [4]
    old_batch_shape = [2, 2]
    wishart, reshape_wishart = self.make_wishart(
        dims, new_batch_shape, old_batch_shape)

    batch_shape = reshape_wishart.batch_shape_tensor()
    event_shape = reshape_wishart.event_shape_tensor()

    expected_sample_shape = [3, 1] + new_batch_shape + [dims, dims]
    x = wishart.sample([3, 1], seed=seed)
    expected_sample = tf.reshape(x, expected_sample_shape)
    actual_sample = reshape_wishart.sample([3, 1], seed=seed)

    expected_log_prob_shape = [3, 1] + new_batch_shape
    expected_log_prob = tf.reshape(wishart.log_prob(x), expected_log_prob_shape)
    actual_log_prob = reshape_wishart.log_prob(expected_sample)

    [
        batch_shape_,
        event_shape_,
        expected_sample_,
        actual_sample_,
        expected_log_prob_,
        actual_log_prob_,
    ] = self.evaluate([
        batch_shape,
        event_shape,
        expected_sample,
        actual_sample,
        expected_log_prob,
        actual_log_prob,
    ])

    self.assertAllEqual(new_batch_shape, batch_shape_)
    self.assertAllEqual([dims, dims], event_shape_)
    self.assertAllClose(expected_sample_, actual_sample_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(new_batch_shape, reshape_wishart.batch_shape)
    self.assertAllEqual([dims, dims], reshape_wishart.event_shape)
    self.assertAllEqual(expected_sample_shape, actual_sample.shape)
    self.assertAllEqual(expected_log_prob_shape, actual_log_prob.shape)

  def test_matrix_variate_stats(self):
    dims = 2
    new_batch_shape = [4]
    old_batch_shape = [2, 2]
    wishart, reshape_wishart = self.make_wishart(
        dims, new_batch_shape, old_batch_shape)

    expected_scalar_stat_shape = new_batch_shape
    expected_matrix_stat_shape = new_batch_shape + [dims, dims]

    expected_entropy = tf.reshape(wishart.entropy(), expected_scalar_stat_shape)
    actual_entropy = reshape_wishart.entropy()

    expected_mean = tf.reshape(wishart.mean(), expected_matrix_stat_shape)
    actual_mean = reshape_wishart.mean()

    expected_mode = tf.reshape(wishart.mode(), expected_matrix_stat_shape)
    actual_mode = reshape_wishart.mode()

    expected_stddev = tf.reshape(wishart.stddev(), expected_matrix_stat_shape)
    actual_stddev = reshape_wishart.stddev()

    expected_variance = tf.reshape(wishart.variance(),
                                   expected_matrix_stat_shape)
    actual_variance = reshape_wishart.variance()

    [
        expected_entropy_,
        actual_entropy_,
        expected_mean_,
        actual_mean_,
        expected_mode_,
        actual_mode_,
        expected_stddev_,
        actual_stddev_,
        expected_variance_,
        actual_variance_,
    ] = self.evaluate([
        expected_entropy,
        actual_entropy,
        expected_mean,
        actual_mean,
        expected_mode,
        actual_mode,
        expected_stddev,
        actual_stddev,
        expected_variance,
        actual_variance,
    ])

    self.assertAllClose(expected_entropy_, actual_entropy_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mode_, actual_mode_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_stddev_, actual_stddev_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(expected_scalar_stat_shape, actual_entropy.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_mean.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_mode.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_stddev.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_variance.shape)

  def make_normal(self, new_batch_shape, old_batch_shape):
    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))

    scale = self.dtype(0.5 + np.arange(
        np.prod(old_batch_shape)).reshape(old_batch_shape))
    scale_ph = tf1.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    normal = tfd.Normal(loc=self.dtype(0), scale=scale_ph, validate_args=True)
    reshape_normal = tfd.BatchReshape(
        distribution=normal, batch_shape=new_batch_shape_ph, validate_args=True)
    return normal, reshape_normal

  def test_scalar_variate_sample_and_log_prob(self):
    if tf.executing_eagerly():
      # TODO(b/122840816): Modify this test so that it runs in eager mode or
      # document that the test is not intended to run in eager mode.
      return

    seed = test_util.test_seed()

    new_batch_shape = [2, 2]
    old_batch_shape = [4]

    normal, reshape_normal = self.make_normal(
        new_batch_shape, old_batch_shape)

    batch_shape = reshape_normal.batch_shape_tensor()
    event_shape = reshape_normal.event_shape_tensor()

    expected_sample_shape = new_batch_shape
    x = normal.sample(seed=seed)
    expected_sample = tf.reshape(x, expected_sample_shape)
    actual_sample = reshape_normal.sample(seed=seed)

    expected_log_prob_shape = new_batch_shape
    expected_log_prob = tf.reshape(normal.log_prob(x), expected_log_prob_shape)
    actual_log_prob = reshape_normal.log_prob(expected_sample)

    [
        batch_shape_,
        event_shape_,
        expected_sample_,
        actual_sample_,
        expected_log_prob_,
        actual_log_prob_,
    ] = self.evaluate([
        batch_shape,
        event_shape,
        expected_sample,
        actual_sample,
        expected_log_prob,
        actual_log_prob,
    ])
    self.assertAllEqual(new_batch_shape, batch_shape_)
    self.assertAllEqual([], event_shape_)
    self.assertAllClose(expected_sample_, actual_sample_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(new_batch_shape, reshape_normal.batch_shape)
    self.assertAllEqual([], reshape_normal.event_shape)
    self.assertAllEqual(expected_sample_shape, actual_sample.shape)
    self.assertAllEqual(expected_log_prob_shape, actual_log_prob.shape)

  def test_scalar_variate_stats(self):
    new_batch_shape = [2, 2]
    old_batch_shape = [4]

    normal, reshape_normal = self.make_normal(new_batch_shape, old_batch_shape)

    expected_scalar_stat_shape = new_batch_shape

    expected_entropy = tf.reshape(normal.entropy(), expected_scalar_stat_shape)
    actual_entropy = reshape_normal.entropy()

    expected_mean = tf.reshape(normal.mean(), expected_scalar_stat_shape)
    actual_mean = reshape_normal.mean()

    expected_mode = tf.reshape(normal.mode(), expected_scalar_stat_shape)
    actual_mode = reshape_normal.mode()

    expected_stddev = tf.reshape(normal.stddev(), expected_scalar_stat_shape)
    actual_stddev = reshape_normal.stddev()

    expected_variance = tf.reshape(normal.variance(),
                                   expected_scalar_stat_shape)
    actual_variance = reshape_normal.variance()

    [
        expected_entropy_,
        actual_entropy_,
        expected_mean_,
        actual_mean_,
        expected_mode_,
        actual_mode_,
        expected_stddev_,
        actual_stddev_,
        expected_variance_,
        actual_variance_,
    ] = self.evaluate([
        expected_entropy,
        actual_entropy,
        expected_mean,
        actual_mean,
        expected_mode,
        actual_mode,
        expected_stddev,
        actual_stddev,
        expected_variance,
        actual_variance,
    ])
    self.assertAllClose(expected_entropy_, actual_entropy_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mode_, actual_mode_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_stddev_, actual_stddev_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(expected_scalar_stat_shape, actual_entropy.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_mean.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_mode.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_stddev.shape)
    self.assertAllEqual(expected_scalar_stat_shape, actual_variance.shape)

  def make_mvn(self, dims, new_batch_shape, old_batch_shape):
    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = tf1.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = tfd.MultivariateNormalDiag(scale_diag=scale_ph, validate_args=True)
    reshape_mvn = tfd.BatchReshape(
        distribution=mvn, batch_shape=new_batch_shape_ph, validate_args=True)
    return mvn, reshape_mvn

  def test_vector_variate_sample_and_log_prob(self):
    if tf.executing_eagerly():
      # TODO(b/122840816): Modify this test so that it runs in eager mode or
      # document that the test is not intended to run in eager mode.
      return

    dims = 3
    seed = test_util.test_seed()
    new_batch_shape = [2, 1]
    old_batch_shape = [2]
    mvn, reshape_mvn = self.make_mvn(
        dims, new_batch_shape, old_batch_shape)

    batch_shape = reshape_mvn.batch_shape_tensor()
    event_shape = reshape_mvn.event_shape_tensor()

    expected_sample_shape = [3] + new_batch_shape + [dims]
    x = mvn.sample(3, seed=seed)
    expected_sample = tf.reshape(x, expected_sample_shape)
    actual_sample = reshape_mvn.sample(3, seed=seed)

    expected_log_prob_shape = [3] + new_batch_shape
    expected_log_prob = tf.reshape(mvn.log_prob(x), expected_log_prob_shape)
    actual_log_prob = reshape_mvn.log_prob(expected_sample)

    [
        batch_shape_,
        event_shape_,
        expected_sample_,
        actual_sample_,
        expected_log_prob_,
        actual_log_prob_,
    ] = self.evaluate([
        batch_shape,
        event_shape,
        expected_sample,
        actual_sample,
        expected_log_prob,
        actual_log_prob,
    ])
    self.assertAllEqual(new_batch_shape, batch_shape_)
    self.assertAllEqual([dims], event_shape_)
    self.assertAllClose(expected_sample_, actual_sample_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_log_prob_, actual_log_prob_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(new_batch_shape, reshape_mvn.batch_shape)
    self.assertAllEqual([dims], reshape_mvn.event_shape)
    self.assertAllEqual(expected_sample_shape, actual_sample.shape)
    self.assertAllEqual(expected_log_prob_shape, actual_log_prob.shape)

  def test_vector_variate_stats(self):
    dims = 3
    new_batch_shape = [2, 1]
    old_batch_shape = [2]
    mvn, reshape_mvn = self.make_mvn(
        dims, new_batch_shape, old_batch_shape)

    expected_scalar_stat_shape = new_batch_shape

    expected_entropy = tf.reshape(mvn.entropy(), expected_scalar_stat_shape)
    actual_entropy = reshape_mvn.entropy()

    expected_vector_stat_shape = new_batch_shape + [dims]

    expected_mean = tf.reshape(mvn.mean(), expected_vector_stat_shape)
    actual_mean = reshape_mvn.mean()

    expected_mode = tf.reshape(mvn.mode(), expected_vector_stat_shape)
    actual_mode = reshape_mvn.mode()

    expected_stddev = tf.reshape(mvn.stddev(), expected_vector_stat_shape)
    actual_stddev = reshape_mvn.stddev()

    expected_variance = tf.reshape(mvn.variance(), expected_vector_stat_shape)
    actual_variance = reshape_mvn.variance()

    expected_matrix_stat_shape = new_batch_shape + [dims, dims]

    expected_covariance = tf.reshape(mvn.covariance(),
                                     expected_matrix_stat_shape)
    actual_covariance = reshape_mvn.covariance()

    [
        expected_entropy_,
        actual_entropy_,
        expected_mean_,
        actual_mean_,
        expected_mode_,
        actual_mode_,
        expected_stddev_,
        actual_stddev_,
        expected_variance_,
        actual_variance_,
        expected_covariance_,
        actual_covariance_,
    ] = self.evaluate([
        expected_entropy,
        actual_entropy,
        expected_mean,
        actual_mean,
        expected_mode,
        actual_mode,
        expected_stddev,
        actual_stddev,
        expected_variance,
        actual_variance,
        expected_covariance,
        actual_covariance,
    ])
    self.assertAllClose(expected_entropy_, actual_entropy_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mean_, actual_mean_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_mode_, actual_mode_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_stddev_, actual_stddev_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_variance_, actual_variance_,
                        atol=0., rtol=1e-6)
    self.assertAllClose(expected_covariance_, actual_covariance_,
                        atol=0., rtol=1e-6)
    if not self.is_static_shape:
      return
    self.assertAllEqual(expected_scalar_stat_shape, actual_entropy.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_mean.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_mode.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_stddev.shape)
    self.assertAllEqual(expected_vector_stat_shape, actual_variance.shape)
    self.assertAllEqual(expected_matrix_stat_shape, actual_covariance.shape)

  def test_bad_reshape_size(self):
    dims = 2
    new_batch_shape = [2, 3]
    old_batch_shape = [2]   # 2 != 2*3

    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = tf1.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = tfd.MultivariateNormalDiag(scale_diag=scale_ph, validate_args=True)

    if self.is_static_shape or tf.executing_eagerly():
      with self.assertRaisesRegexp(
          ValueError, (r'`batch_shape` size \(6\) must match '
                       r'`distribution\.batch_shape` size \(2\)')):
        tfd.BatchReshape(
            distribution=mvn,
            batch_shape=new_batch_shape_ph,
            validate_args=True)

    else:
      with self.assertRaisesOpError(r'Shape sizes do not match.'):
        self.evaluate(
            tfd.BatchReshape(
                distribution=mvn,
                batch_shape=new_batch_shape_ph,
                validate_args=True).sample(seed=test_util.test_seed()))

  def test_non_positive_shape(self):
    dims = 2
    old_batch_shape = [4]
    if self.is_static_shape:
      # Unknown first dimension does not trigger size check. Note that
      # any dimension < 0 is treated statically as unknown.
      new_batch_shape = [-1, 0]
    else:
      new_batch_shape = [-2, -2]  # -2 * -2 = 4, same size as the old shape.

    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = tf1.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = tfd.MultivariateNormalDiag(scale_diag=scale_ph, validate_args=True)

    if self.is_static_shape or tf.executing_eagerly():
      with self.assertRaisesRegexp(ValueError, r'.*must be >=(-1| 0).*'):
        tfd.BatchReshape(
            distribution=mvn,
            batch_shape=new_batch_shape_ph,
            validate_args=True)

    else:
      with self.assertRaisesOpError(r'.*must be >=(-1| 0).*'):
        self.evaluate(
            tfd.BatchReshape(
                distribution=mvn,
                batch_shape=new_batch_shape_ph,
                validate_args=True).sample(seed=test_util.test_seed()))

  def test_non_vector_shape(self):
    if tf.executing_eagerly():
      # TODO(b/122840816): Modify this test so that it runs in eager mode or
      # document that the test is not intended to run in eager mode.
      return

    dims = 2
    new_batch_shape = 2
    old_batch_shape = [2]

    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))

    scale = np.ones(old_batch_shape + [dims], self.dtype)
    scale_ph = tf1.placeholder_with_default(
        scale, shape=scale.shape if self.is_static_shape else None)
    mvn = tfd.MultivariateNormalDiag(scale_diag=scale_ph, validate_args=True)

    if self.is_static_shape:
      with self.assertRaisesRegexp(ValueError, r'.*must be a vector.*'):
        tfd.BatchReshape(
            distribution=mvn,
            batch_shape=new_batch_shape_ph,
            validate_args=True)

    else:
      with self.assertRaisesOpError(r'.*must be a vector.*'):
        self.evaluate(
            tfd.BatchReshape(
                distribution=mvn,
                batch_shape=new_batch_shape_ph,
                validate_args=True).sample(seed=test_util.test_seed()))

  def test_broadcasting_explicitly_unsupported(self):
    old_batch_shape = [4]
    new_batch_shape = [1, 4, 1]
    rate_ = self.dtype([1, 10, 2, 20])

    rate = tf1.placeholder_with_default(
        rate_, shape=old_batch_shape if self.is_static_shape else None)
    poisson_4 = tfd.Poisson(rate, validate_args=True)
    new_batch_shape_ph = (
        tf.constant(np.int32(new_batch_shape)) if self.is_static_shape else
        tf1.placeholder_with_default(np.int32(new_batch_shape), shape=None))
    poisson_141_reshaped = tfd.BatchReshape(
        poisson_4, new_batch_shape_ph, validate_args=True)

    x_4 = self.dtype([2, 12, 3, 23])
    x_114 = self.dtype([2, 12, 3, 23]).reshape(1, 1, 4)

    if self.is_static_shape or tf.executing_eagerly():
      with self.assertRaisesRegexp(NotImplementedError,
                                   'too few batch and event dims'):
        poisson_141_reshaped.log_prob(x_4)
      with self.assertRaisesRegexp(NotImplementedError,
                                   'unexpected batch and event shape'):
        poisson_141_reshaped.log_prob(x_114)
      return

    with self.assertRaisesOpError('too few batch and event dims'):
      self.evaluate(poisson_141_reshaped.log_prob(x_4))

    with self.assertRaisesOpError('unexpected batch and event shape'):
      self.evaluate(poisson_141_reshaped.log_prob(x_114))

  def test_at_most_one_implicit_dimension(self):
    batch_shape = tf.Variable([-1, -1])
    self.evaluate(batch_shape.initializer)
    with self.assertRaisesOpError('At most one dimension can be unknown'):
      d = tfd.BatchReshape(tfd.Normal(0, 1), batch_shape, validate_args=True)
      self.evaluate(d.sample(seed=test_util.test_seed()))

  def test_mutated_at_most_one_implicit_dimension(self):
    batch_shape = tf.Variable([1, 1])
    self.evaluate(batch_shape.initializer)
    dist = tfd.Normal([[0]], [[1]])
    d = tfd.BatchReshape(dist, batch_shape, validate_args=True)
    self.evaluate(d.sample(seed=test_util.test_seed()))
    with self.assertRaisesOpError('At most one dimension can be unknown'):
      with tf.control_dependencies([batch_shape.assign([-1, -1])]):
        self.evaluate(d.sample(seed=test_util.test_seed()))

  def test_default_event_space_bijector_shape(self):
    dist = tfd.Uniform(low=[1., 2., 3., 6.], high=10., validate_args=True)
    batch_shape = [2, 2, 1]
    reshape_dist = tfd.BatchReshape(
        dist, batch_shape=batch_shape, validate_args=True)
    x = self.evaluate(
        dist._experimental_default_event_space_bijector()(
            10. * tf.ones(dist.batch_shape)))
    x_reshape = self.evaluate(
        reshape_dist._experimental_default_event_space_bijector()(
            10. * tf.ones(reshape_dist.batch_shape)))
    self.assertAllEqual(tf.reshape(x, batch_shape), x_reshape)

  def test_default_event_space_bijector_scalar_congruency(self):
    dist = tfd.Triangular(low=2., high=10., peak=7., validate_args=True)
    reshape_dist = tfd.BatchReshape(dist, batch_shape=(), validate_args=True)
    eps = 1e-6
    bijector_test_util.assert_scalar_congruency(
        reshape_dist._experimental_default_event_space_bijector(),
        lower_x=2+eps, upper_x=10-eps, eval_func=self.evaluate, rtol=.15)

  def test_default_event_space_bijector_bijective_and_finite(self):
    batch_shape = [5, 1, 4]
    batch_size = np.prod(batch_shape)
    low = tf.Variable(
        np.linspace(-5., 5., batch_size).astype(self.dtype),
        shape=(batch_size,) if self.is_static_shape else None)
    dist = tfd.Uniform(
        low=low,
        high=30.,
        validate_args=True)
    reshape_dist = tfd.BatchReshape(
        dist, batch_shape=batch_shape, validate_args=True)
    x = np.linspace(
        -10., 10., batch_size).astype(self.dtype).reshape(batch_shape)
    y = np.linspace(
        5., 30 - 1e-4, batch_size).astype(self.dtype).reshape(batch_shape)

    self.evaluate(low.initializer)
    bijector_test_util.assert_bijective_and_finite(
        reshape_dist._experimental_default_event_space_bijector(),
        x, y, eval_func=self.evaluate, event_ndims=0, rtol=1e-4)


@test_util.test_all_tf_execution_regimes
class BatchReshapeStaticTest(_BatchReshapeTest, test_util.TestCase):

  dtype = np.float32
  is_static_shape = True


@test_util.test_all_tf_execution_regimes
class BatchReshapeDynamicTest(_BatchReshapeTest, test_util.TestCase):

  dtype = np.float64
  is_static_shape = False


if __name__ == '__main__':
  tf.test.main()
