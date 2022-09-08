# Copyright 2022 The TensorFlow Probability Authors.
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
"""Tests for tensorflow_probability.python.internal.empirical_statistical_testing."""

from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import empirical_statistical_testing as emp
from tensorflow_probability.python.internal import test_util


class EmpiricalStatisticalTestingTest(test_util.TestCase):

  def test_ok(self):
    # True mean is 0, true stddev is 0.001
    samples = normal.Normal(0., 1.).sample(1000000, seed=test_util.test_seed())
    # TODO(axch): Add seeds for the randomness inside the bootstrap this does.
    results = emp._evaluate_means_assertion(
        self.evaluate(samples), expected=0, axis=0, atol=0.003, rtol=1e-6)
    self.assertEqual(len(results), 1)  # One batch member
    result = results[0]
    self.assertIsInstance(result, emp._Result)

    # True failure probability should be 0.0027, which is the
    # two-sided 3-sigma failure rate.
    # Test that the 1e-9 confidence interval contains it.
    self.assertLess(result.bootstrap.for_error_rate(1e-9).low_p, 0.0027)
    self.assertGreater(result.bootstrap.for_error_rate(1e-9).high_p, 0.0027)

    # If I want a (two-sided) failure rate that's more like 5.6e-7, I
    # should go to 5 sigma.
    suggestion = result.gaussian.for_error_rate(5.6e-7)
    self.assertFalse(suggestion.out_of_bounds)
    self.assertFalse(suggestion.too_tight)
    self.assertFalse(suggestion.too_broad)
    # However, randomness in the original sample makes the suggestion imprecise.
    self.assertAllClose(suggestion.new_atol, 0.005, rtol=0.3)

    # The report computation functions don't crash
    self.assertIsInstance(emp._format_brief_report(results), str)
    self.assertIsInstance(emp._format_full_report(results), str)

  def test_out_of_bounds(self):
    samples = normal.Normal(10., 0.1).sample(10000, seed=test_util.test_seed())
    results = emp._evaluate_means_assertion(
        self.evaluate(samples), expected=0, axis=0, atol=0.003, rtol=1e-6)
    result = results[0]

    # The true failure probabiliy should be almost 1
    self.assertGreater(result.bootstrap.for_error_rate(1e-9).low_p, 0.9)

    # Fixing the test by moving the tolerance should be fairly extreme
    suggestion = result.gaussian.for_error_rate(5.6e-7)
    self.assertGreater(suggestion.new_atol, 9.)

    # It shouldn't be possible to fix this by increasing the number of
    # samples, because the empirical mean is out of bounds
    self.assertTrue(suggestion.out_of_bounds)

  def test_batch(self):
    samples = normal.Normal(0., 0.1).sample(100, seed=test_util.test_seed())
    results = emp._evaluate_means_assertion(
        self.evaluate(samples), expected=[[0], [1], [2]], axis=0,
        atol=[0.003, 0.004, 0.005], rtol=1e-6)
    self.assertEqual(len(results), 9)

if __name__ == '__main__':
  test_util.main()
