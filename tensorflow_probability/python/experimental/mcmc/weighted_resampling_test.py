# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the _License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for weighted resampling methods."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import _resample_using_log_points
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import _scatter_nd_batch
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_deterministic_minimum_error
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_independent
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_stratified
from tensorflow_probability.python.experimental.mcmc.weighted_resampling import resample_systematic
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import test_util

tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class _SMCResamplersTest(test_util.TestCase):

  # TODO(b/153689734): rewrite so as not to use `move_dimension`.
  def test_categorical_resampler_chi2(self):
    strm = test_util.test_seed_stream()
    # Test categorical resampler using chi-squared test.
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    num_probs = 50
    num_distributions = 3
    unnormalized_probs = tfd.Uniform(
        low=self.dtype(0),
        high=self.dtype(1.)).sample([num_distributions, num_probs], seed=strm)
    probs = unnormalized_probs / tf.reduce_sum(
        unnormalized_probs, axis=-1, keepdims=True)

    # chi-squared test is valid as long as `num_samples` is
    # large compared to `num_probs`.
    num_particles = 10000
    num_samples = 2

    sample = self.maybe_compiler(resample_independent)(
        tf.math.log(dist_util.move_dimension(probs,
                                             source_idx=-1,
                                             dest_idx=0)),
        num_particles,
        [num_samples],
        seed=strm)
    elems = tf.range(num_probs)
    initializer = tf.zeros([num_samples, num_distributions], dtype=sample.dtype)
    counts = tf.scan(
        lambda _, x: tf.reduce_sum(  # pylint: disable=g-long-lambda
            tf.cast(tf.equal(sample, x), sample.dtype), axis=0),
        elems,
        initializer)
    counts = dist_util.move_dimension(
        tf.cast(counts, self.dtype), source_idx=0, dest_idx=-1)
    expected_samples = probs * num_particles
    chi2 = tf.reduce_sum(
        (counts - expected_samples)**2 / expected_samples, axis=-1)
    self.assertAllLess(
        tfd.Chi2(df=self.dtype(num_probs - 1)).cdf(chi2), 0.99995)

  def test_categorical_resampler_zero_final_class(self):
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    probs = self.dtype([1.0, 0.0])
    resampled = self.maybe_compiler(resample_independent)(
        tf.math.log(probs), 1000, [], seed=test_util.test_seed_stream())
    self.assertAllClose(resampled, tf.zeros((1000,), dtype=tf.int32))

  def test_systematic_resampler_zero_final_class(self):
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    probs = self.dtype([1.0, 0.0])
    resampled = self.maybe_compiler(resample_systematic)(
        tf.math.log(probs), 1000, [], seed=test_util.test_seed_stream())
    self.assertAllClose(resampled, tf.zeros((1000,), dtype=tf.int32))

  def test_categorical_resampler_large(self):
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    num_probs = 10000
    log_probs = tf.fill([num_probs], -tf.math.log(self.dtype(num_probs)))
    self.evaluate(self.maybe_compiler(resample_independent)(
        log_probs, num_probs, [], seed=test_util.test_seed_stream()))

  def test_systematic_resampler_large(self):
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    num_probs = 10000
    log_probs = tf.fill([num_probs], -tf.math.log(self.dtype(num_probs)))
    self.evaluate(self.maybe_compiler(resample_systematic)(
        log_probs, num_probs, [],
        seed=test_util.test_seed_stream()))

  # TODO(b/153689734): rewrite so as not to use `move_dimension`.
  def test_systematic_resampler_means(self):
    strm = test_util.test_seed_stream()
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    # Distinct samples with this resampler aren't independent
    # so a chi-squared test is inappropriate.
    # However, because it reduces variance by design, we
    # can, with high probability,  place sharp bounds on the
    # values of the sample means.
    num_distributions = 3
    num_probs = 16
    probs = tfd.Uniform(
        low=self.dtype(0.0),
        high=self.dtype(1.0)).sample([num_distributions, num_probs], seed=strm)
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    num_samples = 10000
    num_particles = 20
    resampled = self.maybe_compiler(resample_systematic)(
        tf.math.log(dist_util.move_dimension(probs,
                                             source_idx=-1,
                                             dest_idx=0)),
        num_particles, [num_samples], seed=test_util.test_seed_stream())
    resampled = dist_util.move_dimension(resampled,
                                         source_idx=0,
                                         dest_idx=-1)
    # TODO(dpiponi): reimplement this test in vectorized form rather than with
    # loops.
    for i in range(num_distributions):
      samples = _scatter_nd_batch(
          resampled[:, i, :, tf.newaxis],
          tf.ones((num_samples, num_particles),
                  dtype=self.dtype),
          (num_samples, num_probs),
          batch_dims=1)
      # N.B.: _scatter_nd_batch returns numpy arrays in Eager mode
      wsamples = tf.convert_to_tensor(samples) / num_particles
      wsamples_, probs_ = self.evaluate([wsamples, probs[i]])

      # TODO(dpiponi): it should be possible to compute the exact distribution
      # of these means and choose `atol` in a more principled way.
      self.assertAllMeansClose(wsamples_, probs_, axis=0, atol=0.01)

  # TODO(b/153689734): rewrite so as not to use `move_dimension`.
  def test_minimum_error_resampler_means(self):
    strm = test_util.test_seed_stream()
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    # Distinct samples with this resampler aren't independent
    # so a chi-squared test is inappropriate.
    # However, because it reduces variance by design, we
    # can, with high probability,  place sharp bounds on the
    # values of the sample means.
    num_distributions = 3
    num_probs = 8
    probs = tfd.Uniform(
        low=self.dtype(0.0),
        high=self.dtype(1.0)).sample([num_distributions, num_probs], seed=strm)
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    num_samples = 4
    num_particles = 10000
    resampled = self.maybe_compiler(resample_deterministic_minimum_error)(
        tf.math.log(dist_util.move_dimension(probs,
                                             source_idx=-1,
                                             dest_idx=0)),
        num_particles, [num_samples], seed=test_util.test_seed_stream())
    resampled = dist_util.move_dimension(resampled,
                                         source_idx=0,
                                         dest_idx=-1)
    # TODO(dpiponi): reimplement this test in vectorized form rather than with
    # loops.
    for i in range(num_distributions):
      samples = _scatter_nd_batch(
          resampled[:, i, :, tf.newaxis],
          tf.ones((num_samples, num_particles)),
          (num_samples, num_probs),
          batch_dims=1)
      # N.B.: _scatter_nd_batch returns numpy arrays in Eager mode
      wsamples = tf.convert_to_tensor(samples) / num_particles
      wsamples_, probs_ = self.evaluate([wsamples, probs[i]])

      self.assertAllMeansClose(
          wsamples_, probs_, axis=0, atol=1.0 / num_particles)

  # TODO(b/153689734): rewrite so as not to use `move_dimension`.
  def test_stratified_resampler_means(self):
    strm = test_util.test_seed_stream()
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    # Distinct samples with this resampler aren't independent
    # so a chi-squared test is inappropriate.
    # However, because it reduces variance by design, we
    # can, with high probability,  place sharp bounds on the
    # values of the sample means.
    num_distributions = 3
    num_probs = 8
    probs = tfd.Uniform(
        low=self.dtype(0.0),
        high=self.dtype(1.0)).sample([num_distributions, num_probs], seed=strm)
    probs = probs / tf.reduce_sum(probs, axis=-1, keepdims=True)
    num_samples = 4
    num_particles = 10000
    resampled = self.maybe_compiler(resample_stratified)(
        tf.math.log(dist_util.move_dimension(probs,
                                             source_idx=-1,
                                             dest_idx=0)),
        num_particles, [num_samples], seed=test_util.test_seed_stream())
    resampled = dist_util.move_dimension(resampled,
                                         source_idx=0,
                                         dest_idx=-1)
    # TODO(dpiponi): reimplement this test in vectorized form rather than with
    # loops.
    for i in range(num_distributions):
      samples = _scatter_nd_batch(
          resampled[:, i, :, tf.newaxis],
          tf.ones((num_samples, num_particles)),
          (num_samples, num_probs),
          batch_dims=1)
      # N.B.: _scatter_nd_batch returns numpy arrays in Eager mode
      wsamples = tf.convert_to_tensor(samples) / num_particles
      wsamples_, probs_ = self.evaluate([wsamples, probs[i]])

      self.assertAllMeansClose(wsamples_, probs_, axis=0, atol=0.1)

  def test_resample_using_extremal_log_points(self):
    if self.use_xla and tf.executing_eagerly():
      self.skipTest('No need to test XLA under all execution regimes.')

    n = 8

    one = self.dtype(1)
    almost_one = np.nextafter(one, self.dtype(0))
    sample_shape = []
    log_points_zero = tf.math.log(tf.zeros(n, self.dtype))
    log_points_almost_one = tf.fill([n], tf.math.log(almost_one))
    log_probs_beginning = tf.math.log(
        tf.concat([[one], tf.zeros(n - 1, dtype=self.dtype)], axis=0))
    log_probs_end = tf.math.log(tf.concat([tf.zeros(n - 1, dtype=self.dtype),
                                           [one]], axis=0))

    sample_fn = self.maybe_compiler(_resample_using_log_points)
    indices = sample_fn(
        log_probs_beginning, sample_shape, log_points_zero)
    self.assertAllEqual(indices, tf.zeros(n, dtype=tf.float32))

    indices = sample_fn(
        log_probs_end, sample_shape, log_points_zero)
    self.assertAllEqual(indices, tf.fill([n], n - 1))

    indices = sample_fn(
        log_probs_beginning, sample_shape, log_points_almost_one)
    self.assertAllEqual(indices, tf.zeros(n, dtype=tf.float32))

    indices = sample_fn(
        log_probs_end, sample_shape, log_points_almost_one)
    self.assertAllEqual(indices, tf.fill([n], n - 1))

  def resample_with_target_distribution(self):
    particles = np.linspace(0., 500., num=2500, dtype=np.float32)
    log_weights = tfd.Poisson(20.).log_prob(particles)

    # Resample particles to target a Poisson(20.) distribution.
    new_particles, _, new_log_weights = resample(
        particles, log_weights,
        resample_fn=resample_systematic,
        seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllMeansClose(new_particles, 20., axis=0, atol=1e-2)
    self.assertAllClose(
        tf.reduce_sum(tf.nn.softmax(new_log_weights) * new_particles),
        20.,
        atol=1e-2)

    # Reweight the resampled particles to target a Poisson(30.) distribution.
    new_particles, _, new_log_weights = resample(
        particles, log_weights,
        resample_fn=resample_systematic,
        target_log_weights=tfd.Poisson(30).log_prob(particles),
        seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllMeansClose(new_particles, 20., axis=0, atol=1e-2)
    self.assertAllClose(
        tf.reduce_sum(tf.nn.softmax(new_log_weights) * new_particles),
        30., atol=1.)

  def maybe_compiler(self, f):
    if self.use_xla:
      return tf.function(f, autograph=False, jit_compile=True)
    return f  # Do not compile.


class SMCResamplersTestFloat32(_SMCResamplersTest):
  dtype = np.float32
  use_xla = False


class SMCResamplersTestFloat32XLACompiled(_SMCResamplersTest):
  dtype = np.float32
  use_xla = True


class SMCResamplersTestFloat64(_SMCResamplersTest):
  dtype = np.float64
  use_xla = False


del _SMCResamplersTest  # Don't try to run tests from the base class.


if __name__ == '__main__':
  test_util.main()
