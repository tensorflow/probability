# Copyright 2024 The TensorFlow Probability Authors.
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
"""Tests for estimators.py."""

import jax
import numpy as np
from tensorflow_probability.python.experimental.autobnn import estimators
from tensorflow_probability.python.experimental.autobnn import util
from tensorflow_probability.python.internal import test_util


class AutoBNNTest(test_util.TestCase):

  def test_train_map(self):
    seed = jax.random.PRNGKey(20231018)
    x_train, y_train = util.load_fake_dataset()

    autobnn = estimators.AutoBnnMapEstimator(
        model_name='linear_plus_periodic',
        likelihood_model='normal_likelihood_logistic_noise',
        seed=seed,
        width=5,
        num_particles=8,
        num_iters=100,
    )
    self.assertFalse(autobnn.check_is_fitted())
    autobnn.fit(x_train, y_train)
    self.assertTrue(autobnn.check_is_fitted())
    self.assertEqual(autobnn.diagnostics_['loss'].shape, (8, 100))
    lo, mid, hi = autobnn.predict_quantiles(x_train)
    np.testing.assert_array_less(lo, mid)
    np.testing.assert_array_less(mid, hi)
    self.assertEqual(
        autobnn.summary(), '\n'.join(['(Periodic(period=12.00)#Linear)'] * 8)
    )

  def test_train_mcmc(self):
    seed = jax.random.PRNGKey(20231018)
    x_train, y_train = util.load_fake_dataset()

    autobnn = estimators.AutoBnnMCMCEstimator(
        model_name='linear_plus_periodic',
        likelihood_model='normal_likelihood_logistic_noise',
        seed=seed,
        width=5,
        num_chains=2,
        num_draws=10,
    )
    self.assertFalse(autobnn.check_is_fitted())
    autobnn.fit(x_train, y_train)
    self.assertTrue(autobnn.check_is_fitted())
    self.assertEqual(autobnn.diagnostics_['noise_scale'].shape, (2 * 10, 1))
    lo, mid, hi = autobnn.predict_quantiles(x_train)
    np.testing.assert_array_less(lo, mid)
    np.testing.assert_array_less(mid, hi)

  # TODO(colcarroll): Add test for AutoBnnVIEstimator.

  def test_summary(self):
    seed = jax.random.PRNGKey(20231018)
    x_train, y_train = util.load_fake_dataset()

    autobnn = estimators.AutoBnnMapEstimator(
        model_name='sum_of_stumps',
        likelihood_model='normal_likelihood_logistic_noise',
        seed=seed,
        width=5,
        num_particles=8,
        num_iters=100,
    )
    autobnn.fit(x_train, y_train)
    summary_lines = autobnn.summary().split('\n')

    self.assertEqual(len(summary_lines), 8, f'Unexpected {len(summary_lines)=}')

    for line in summary_lines:
      self.assertRegex(
          line[1:-1],
          r'\d\.\d{3}\s(RBF|Matern\(2\.5\)|Linear|Quadratic|'
          r'Periodic\(period\=12\.00\)|OneLayer)(\+?)+',
          f'Unexpected summary {line}',
      )


if __name__ == '__main__':
  test_util.main()
