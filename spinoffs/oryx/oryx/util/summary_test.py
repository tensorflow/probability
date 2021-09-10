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
"""Tests for tensorflow_probability.spinoffs.oryx.util.summary."""

from absl.testing import absltest

from jax import lax
import jax.numpy as jnp
import numpy as np

from oryx.internal import test_util
from oryx.util import summary


class SummaryTest(test_util.TestCase):

  def test_can_pull_out_summarized_values_in_strict_mode(self):
    def f(x):
      return summary.summary(x, name='x')
    _, summaries = summary.get_summaries(f)(1.)
    self.assertDictEqual(dict(x=1.), summaries)

  def test_can_pull_out_non_dependent_values(self):
    def f(x):
      summary.summary(x ** 2, name='y')
      return x
    _, summaries = summary.get_summaries(f)(2.)
    self.assertDictEqual(dict(y=4.), summaries)

  def test_duplicate_names_error_in_strict_mode(self):
    def f(x):
      summary.summary(x, name='x')
      summary.summary(x, name='x')
      return x
    with self.assertRaisesRegex(ValueError, 'has already been reaped: x'):
      summary.get_summaries(f)(2.)

  def test_can_pull_summaries_out_of_scan_in_append_mode(self):
    def f(x):
      def body(x, _):
        summary.summary(x, name='x', mode='append')
        return x + 1, ()
      return lax.scan(body, x, jnp.arange(10.))[0]
    value, summaries = summary.get_summaries(f)(0.)
    self.assertEqual(value, 10.)
    np.testing.assert_allclose(summaries['x'], np.arange(10.))


if __name__ == '__main__':
  absltest.main()
