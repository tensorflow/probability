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
"""Tests for surrogate posteriors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import functools
# Dependency imports
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfb = tfp.bijectors
tfd = tfp.distributions


@test_util.test_all_tf_execution_regimes
class TrainableLocationScale(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'ScalarLaplace',
       'event_shape': [],
       'batch_shape': [],
       'distribution_fn': tfd.Laplace,
       'dtype': np.float64},
      {'testcase_name': 'VectorNormal',
       'event_shape': [2],
       'batch_shape': [3, 1],
       'distribution_fn': tfd.Normal,
       'dtype': np.float32},)
  def test_has_correct_ndims_and_gradients(
      self, event_shape, batch_shape, distribution_fn, dtype):

    initial_loc = np.ones(batch_shape + event_shape)
    dist = tfp.experimental.vi.build_trainable_location_scale_distribution(
        initial_loc=_build_tensor(initial_loc, dtype=dtype,
                                  use_static_shape=True),
        initial_scale=1e-6,
        event_ndims=len(event_shape),
        distribution_fn=distribution_fn,
        validate_args=True)
    self.evaluate([v.initializer for v in dist.trainable_variables])
    self.assertAllClose(self.evaluate(dist.sample()),
                        initial_loc,
                        atol=1e-4)  # Much larger than initial_scale.
    self.assertAllEqual(dist.event_shape.as_list(), event_shape)
    self.assertAllEqual(dist.batch_shape.as_list(), batch_shape)
    for v in dist.trainable_variables:
      self.assertAllEqual(v.shape.as_list(), batch_shape + event_shape)

    # Test that gradients are available wrt the variational parameters.
    self.assertNotEmpty(dist.trainable_variables)
    with tf.GradientTape() as tape:
      posterior_logprob = dist.log_prob(initial_loc)
    grad = tape.gradient(posterior_logprob,
                         dist.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))


@test_util.test_all_tf_execution_regimes
class FactoredSurrogatePosterior(test_util.TestCase):

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': tf.TensorShape([4]),
       'constraining_bijectors': [tfb.Sigmoid()],
       'dtype': np.float64, 'use_static_shape': True},
      {'testcase_name': 'ListEvent',
       'event_shape': [tf.TensorShape([3]),
                       tf.TensorShape([]),
                       tf.TensorShape([2, 2])],
       'constraining_bijectors': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float32, 'use_static_shape': False},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': tf.TensorShape([1]), 'y': tf.TensorShape([])},
       'constraining_bijectors': None,
       'dtype': np.float64, 'use_static_shape': True},
  )
  def test_specifying_event_shape(self, event_shape,
                                  constraining_bijectors,
                                  dtype, use_static_shape):
    seed = test_util.test_seed_stream()
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=tf.nest.map_structure(
                functools.partial(_build_tensor,
                                  dtype=np.int32,
                                  use_static_shape=use_static_shape),
                event_shape),
            constraining_bijectors=constraining_bijectors,
            initial_unconstrained_loc=functools.partial(
                tf.random.uniform, minval=-2., maxval=2., dtype=dtype),
            seed=seed(),
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])
    posterior_sample_ = self.evaluate(surrogate_posterior.sample(seed=seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))
    posterior_event_shape = self.evaluate(
        surrogate_posterior.event_shape_tensor())

    # Test that the posterior has the specified event shape(s).
    tf.nest.map_structure(
        self.assertAllEqual, event_shape, posterior_event_shape)

    # Test that all sample Tensors have the expected shapes.
    check_shape = lambda s, x: self.assertAllEqual(s, x.shape)
    tf.nest.map_structure(check_shape, event_shape, posterior_sample_)

    self.assertAllEqual([], posterior_logprob_.shape)

    # Test that gradients are available wrt the variational parameters.
    self.assertNotEmpty(surrogate_posterior.trainable_variables)
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample_)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  @parameterized.named_parameters(
      {'testcase_name': 'TensorEvent',
       'event_shape': [4],
       'initial_loc': np.array([[[0.9, 0.1, 0.5, 0.7]]]),
       'implicit_batch_shape': [1, 1],
       'constraining_bijectors': tfb.Sigmoid(),
       'dtype': np.float32, 'use_static_shape': False},
      {'testcase_name': 'ListEvent',
       'event_shape': [[3], [], [2, 2]],
       'initial_loc': [np.array([0.1, 7., 3.]),
                       0.1,
                       np.array([[1., 0], [-4., 2.]])],
       'implicit_batch_shape': [],
       'constraining_bijectors': [tfb.Softplus(), None, tfb.FillTriangular()],
       'dtype': np.float64, 'use_static_shape': True},
      {'testcase_name': 'DictEvent',
       'event_shape': {'x': [2], 'y': []},
       'initial_loc': {'x': np.array([[0.9, 1.2]]),
                       'y': np.array([-4.1])},
       'implicit_batch_shape': [1],
       'constraining_bijectors': None,
       'dtype': np.float32, 'use_static_shape': False},
  )
  def test_specifying_initial_loc(self, event_shape, initial_loc,
                                  implicit_batch_shape,
                                  constraining_bijectors,
                                  dtype, use_static_shape):
    initial_loc = tf.nest.map_structure(
        lambda s: _build_tensor(s, dtype=dtype,  # pylint: disable=g-long-lambda
                                use_static_shape=use_static_shape),
        initial_loc)

    if constraining_bijectors is not None:
      initial_unconstrained_loc = tf.nest.map_structure(
          lambda x, b: x if b is None else b.inverse(x),
          initial_loc, constraining_bijectors)
    else:
      initial_unconstrained_loc = initial_loc

    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=event_shape,
            initial_unconstrained_loc=initial_unconstrained_loc,
            initial_unconstrained_scale=1e-6,
            constraining_bijectors=constraining_bijectors,
            validate_args=True))
    self.evaluate([v.initializer
                   for v in surrogate_posterior.trainable_variables])
    posterior_sample_ = self.evaluate(surrogate_posterior.sample(
        seed=test_util.test_seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))

    self.assertAllEqual(implicit_batch_shape, posterior_logprob_.shape)

    # Check that the samples have the correct structure and that the sampled
    # values are close to the initial locs.
    tf.nest.map_structure(functools.partial(self.assertAllClose, atol=1e-4),
                          self.evaluate(initial_loc),
                          posterior_sample_)

  def test_that_gamma_fitting_example_runs(self):

    # Build model.
    Root = tfd.JointDistributionCoroutine.Root  # pylint: disable=invalid-name
    def model_fn():
      concentration = yield Root(tfd.Exponential(1.))
      rate = yield Root(tfd.Exponential(1.))
      y = yield tfd.Sample(  # pylint: disable=unused-variable
          tfd.Gamma(concentration=concentration, rate=rate),
          sample_shape=4)
    model = tfd.JointDistributionCoroutine(model_fn)

    # Build surrogate posterior.
    surrogate_posterior = (
        tfp.experimental.vi.build_factored_surrogate_posterior(
            event_shape=model.event_shape_tensor()[:-1],
            constraining_bijectors=[tfb.Softplus(), tfb.Softplus()]))

    # Fit model.
    y = [0.2, 0.5, 0.3, 0.7]
    losses = tfp.vi.fit_surrogate_posterior(
        lambda rate, concentration: model.log_prob((rate, concentration, y)),
        surrogate_posterior,
        num_steps=5,  # Don't optimize to completion.
        optimizer=tf.optimizers.Adam(0.1),
        sample_size=10)

    # Compute posterior statistics.
    with tf.control_dependencies([losses]):
      posterior_samples = surrogate_posterior.sample(100)
      posterior_mean = [tf.reduce_mean(x) for x in posterior_samples]
      posterior_stddev = [tf.math.reduce_std(x) for x in posterior_samples]

    self.evaluate(tf1.global_variables_initializer())
    _ = self.evaluate(losses)
    _ = self.evaluate(posterior_mean)
    _ = self.evaluate(posterior_stddev)


def _build_tensor(ndarray, dtype, use_static_shape):
  # Enforce parameterized dtype and static/dynamic testing.
  ndarray = np.asarray(ndarray).astype(dtype)
  return tf1.placeholder_with_default(
      input=ndarray, shape=ndarray.shape if use_static_shape else None)

if __name__ == '__main__':
  tf.test.main()
