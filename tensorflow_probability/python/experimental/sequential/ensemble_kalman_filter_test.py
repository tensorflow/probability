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
"""Tests for the Ensemble Kalman Filter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util


tfd = tfp.distributions
tfs = tfp.experimental.sequential


@test_util.test_all_tf_execution_regimes
class EnsembleKalmanFilterTest(test_util.TestCase):

  def test_ensemble_kalman_filter_expect_mvn(self):

    state = tfs.EnsembleKalmanFilterState(step=0, particles=[1.], extra=None)

    with self.assertRaises(ValueError):
      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[0.],
          observation_fn=lambda t, p, e: (tfd.Poisson(rate=p), e))

  def test_ensemble_kalman_filter_constant_univariate_shapes(self):
    # Simple transition model where the state doesn't change,
    # so we are estimating a constant.

    def transition_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-11]), extra

    def observation_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(loc=particles, scale_diag=[1e-2]), extra

    # Initialize the ensemble.
    particles = self.evaluate(tf.random.normal(
        shape=[100, 1], seed=test_util.test_seed()))

    state = tfs.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={'unchanged': 1})

    predicted_state = tfs.ensemble_kalman_filter_predict(
        state, transition_fn=transition_fn, inflate_fn=None)

    # Check that extra is correctly propagated.
    self.assertIn('unchanged', predicted_state.extra)
    self.assertEqual(1, predicted_state.extra['unchanged'])

    # Check that the state respected the constant dynamics.
    self.assertAllClose(state.particles, predicted_state.particles)

    updated_state = tfs.ensemble_kalman_filter_update(
        predicted_state,
        # The observation is the constant 0.
        observation=[0.],
        seed=test_util.test_seed(),
        observation_fn=observation_fn)

    # Check that extra is correctly propagated.
    self.assertIn('unchanged', updated_state.extra)
    self.assertEqual(1, updated_state.extra['unchanged'])
    self.assertAllEqual(state.particles.shape, updated_state.particles.shape)

  def test_ensemble_kalman_filter_linear_model(self):
    # Simple transition model where the state doesn't change,
    # so we are estimating a constant.

    def transition_fn(_, particles, extra):
      particles = {'x': particles['x'] + particles['xdot'],
                   'xdot': particles['xdot']}
      extra['transition_count'] += 1
      return tfd.JointDistributionNamed(dict(
          x=tfd.MultivariateNormalDiag(loc=particles['x'], scale_diag=[1e-11]),
          xdot=tfd.MultivariateNormalDiag(
              particles['xdot'], scale_diag=[1e-11]))), extra

    def observation_fn(_, particles, extra):
      extra['observation_count'] += 1
      return tfd.MultivariateNormalDiag(
          loc=particles['x'], scale_diag=[1e-2]), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles = {
        'x': self.evaluate(tf.random.normal(
            shape=[300, 5, 1], seed=seed_stream())),
        'xdot': self.evaluate(tf.random.normal(
            shape=[300, 5, 1], seed=seed_stream()))
    }

    state = tfs.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={
            'observation_count': 0, 'transition_count': 0})

    for i in range(5):
      state = tfs.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)
      self.assertIn('transition_count', state.extra)
      self.assertEqual(i + 1, state.extra['transition_count'])

      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[1. * i],
          observation_fn=observation_fn,
          seed=seed_stream())

      self.assertIn('observation_count', state.extra)
      self.assertEqual(i + 1, state.extra['observation_count'])

    self.assertAllClose([4.] * 5, self.evaluate(
        tf.reduce_mean(state.particles['x'], axis=[0, -1])), rtol=0.05)

  def test_ensemble_kalman_filter_constant_model_multivariate(self):

    def transition_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-11] * 2), extra

    def observation_fn(_, particles, extra):
      return tfd.MultivariateNormalDiag(
          loc=particles, scale_diag=[1e-1] * 2), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles = self.evaluate(tf.random.normal(
        shape=[300, 3, 2], seed=seed_stream(), dtype=tf.float64))

    state = tfs.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={'unchanged': 1})

    for _ in range(8):
      state = tfs.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)

      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[0., 0.],
          observation_fn=observation_fn,
          seed=seed_stream())

    self.assertAllClose([[0., 0.]] * 3, self.evaluate(
        tf.reduce_mean(state.particles, axis=0)), atol=1e-2)

  def test_ensemble_kalman_filter_linear_model_multivariate(self):

    def transition_fn(_, particles, extra):
      particles = {'x': particles['x'] + particles['xdot'],
                   'xdot': particles['xdot']}
      extra['transition_count'] += 1
      return tfd.JointDistributionNamed(dict(
          x=tfd.MultivariateNormalDiag(particles['x'], scale_diag=[1e-11] * 2),
          xdot=tfd.MultivariateNormalDiag(
              particles['xdot'], scale_diag=[1e-11] * 2))), extra

    def observation_fn(_, particles, extra):
      extra['observation_count'] += 1
      return tfd.MultivariateNormalDiag(
          loc=particles['x'], scale_diag=[1e-2] * 2), extra

    seed_stream = test_util.test_seed_stream()

    # Initialize the ensemble.
    particles = {
        'x': self.evaluate(tf.random.normal(
            shape=[300, 3, 2], seed=seed_stream(), dtype=tf.float64)),
        'xdot': self.evaluate(tf.random.normal(
            shape=[300, 3, 2], seed=seed_stream(), dtype=tf.float64))
    }

    state = tfs.EnsembleKalmanFilterState(
        step=0, particles=particles, extra={
            'observation_count': 0, 'transition_count': 0})

    for i in range(10):
      state = tfs.ensemble_kalman_filter_predict(
          state,
          transition_fn=transition_fn,
          seed=seed_stream(),
          inflate_fn=None)
      self.assertIn('transition_count', state.extra)
      self.assertEqual(i + 1, state.extra['transition_count'])

      state = tfs.ensemble_kalman_filter_update(
          state,
          observation=[1. * i, 2. * i],
          observation_fn=observation_fn,
          seed=seed_stream())
      print(self.evaluate(
          tf.reduce_mean(state.particles['x'], axis=0)))

      self.assertIn('observation_count', state.extra)
      self.assertEqual(i + 1, state.extra['observation_count'])

    self.assertAllClose([[9., 18.]] * 3, self.evaluate(
        tf.reduce_mean(state.particles['x'], axis=0)), rtol=0.05)


if __name__ == '__main__':
  tf.test.main()
