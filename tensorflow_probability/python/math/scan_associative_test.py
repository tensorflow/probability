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
"""Tests for parallel calculation of prefix sums."""

import collections
import functools
import operator

from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.scan_associative import scan_associative


@test_util.test_graph_and_eager_modes()
class _ScanAssociativeTest(test_util.TestCase):

  def test_cumulative_sum_size_zero(self):
    elems = tf.range(0, dtype=tf.int64)
    self.assertAllEqual(
        self.evaluate(scan_associative(operator.add, elems, max_num_levels=8)),
        self.evaluate(tf.cumsum(elems)))

  def test_cumulative_sum_size_one(self):
    elems = self._maybe_static(tf.range(1, dtype=tf.int64))
    self.assertAllEqual(
        self.evaluate(scan_associative(operator.add, elems, max_num_levels=8)),
        self.evaluate(tf.cumsum(elems)))

  def test_cumulative_sum_power_of_two(self):
    elems = self._maybe_static(tf.range(0, 2**4, dtype=tf.int64))
    self.assertAllEqual(
        self.evaluate(scan_associative(operator.add, elems, max_num_levels=8)),
        self.evaluate(tf.cumsum(elems)))

  def test_cumulative_sum_maximally_odd(self):
    # A size that is one less than a power of two ensures that
    # every reduction results in an odd size tensor.
    # This makes a good test for the logic to handle
    # odd sizes
    elems = self._maybe_static(tf.range(0, 2**4 - 1, dtype=tf.int64))
    self.assertAllEqual(
        self.evaluate(scan_associative(operator.add, elems, max_num_levels=8)),
        self.evaluate(tf.cumsum(elems)))

  @parameterized.parameters((0,), (1,), (2,), (-2,), (-1,))
  def test_cumulative_sum_custom_axis(self, axis):
    elems = self._maybe_static(
        tf.random.stateless_normal(
            [4, 32, 31, 1], seed=test_util.test_seed(sampler_type='stateless')))

    axis = self._maybe_static(axis)
    expected_result = self.evaluate(tf.cumsum(elems, axis=axis))
    result = scan_associative(operator.add, elems, axis=axis, max_num_levels=8)
    self.assertAllClose(self.evaluate(result), expected_result, rtol=1e-5)

  def test_counting_by_matmul_example(self):
    num_elems = 2**4 + 1
    upper_row = tf.stack([tf.ones(num_elems, dtype=tf.int64),
                          tf.range(0, num_elems, dtype=tf.int64)], axis=1)
    lower_row = tf.stack([tf.zeros(num_elems, dtype=tf.int64),
                          tf.ones(num_elems, dtype=tf.int64)], axis=1)
    m = self._maybe_static(tf.stack([upper_row, lower_row], axis=1))
    result = self.evaluate(scan_associative(tf.matmul, m, max_num_levels=8))
    self.assertAllEqual(result[:, 0, 1], np.cumsum(np.arange(num_elems)))

  def test_supports_structured_elems_odd_base_case(self):
    pair = collections.namedtuple('pair', ('first', 'second'))
    data = pair(first=self._maybe_static(tf.constant([0., 1., 2.])),
                second=self._maybe_static(tf.constant([0., 10., 20.])))

    def fn(a, b):
      return pair(first=a.first + b.first,
                  second=a.second + b.second)

    result = self.evaluate(scan_associative(fn, elems=data, max_num_levels=8))
    self.assertAllClose(result.first, [0., 1., 3.])
    self.assertAllClose(result.second, [0., 10., 30.])

  def test_supports_structured_elems_complex(self):
    data = self.evaluate(
        uniform.Uniform(-1., 1.).sample(2**4, seed=test_util.test_seed()))
    mean_, variance_ = self.evaluate((
        tf.reduce_mean(data),
        tf.math.reduce_variance(data)))

    # Compute means and variances in a single pass by merging local statistics.
    accumulated_stats = collections.namedtuple(
        'accumulated_stats', ('count', 'mean', 'unscaled_variance'))
    def fn(a, b):
      total_count = a.count + b.count
      return accumulated_stats(
          count=total_count,
          mean=(a.count * a.mean + b.count * b.mean) / total_count,
          unscaled_variance=(
              a.unscaled_variance + b.unscaled_variance +
              (b.mean - a.mean)**2 * a.count * b.count / total_count))

    initial_stats = accumulated_stats(
        count=self._maybe_static(tf.ones_like(data)),
        mean=self._maybe_static(data),
        unscaled_variance=self._maybe_static(tf.zeros_like(data)))
    result = self.evaluate(
        scan_associative(fn, elems=initial_stats, max_num_levels=8))
    self.assertAllClose(mean_, result.mean[-1])
    self.assertAllClose(variance_,
                        result.unscaled_variance[-1] / result.count[-1])

  def test_can_scan_tensors_of_different_rank(self):
    num_elems = 2**4
    elems0 = self.evaluate(
        uniform.Uniform(-1., 1.).sample(
            sample_shape=[num_elems], seed=test_util.test_seed()))
    elems1 = self.evaluate(
        uniform.Uniform(-1., 1.).sample(
            sample_shape=[num_elems, 1], seed=test_util.test_seed()))

    def extended_add(a, b):
      return (a[0] + b[0], a[1] + b[1])

    result = self.evaluate(
        scan_associative(
            extended_add,
            (self._maybe_static(elems0), self._maybe_static(elems1)),
            max_num_levels=8))

    self.assertAllClose(
        result[0],
        self.evaluate(tf.cumsum(elems0)))
    self.assertAllClose(
        result[1],
        self.evaluate(tf.cumsum(elems1, axis=0)))

  @test_util.numpy_disable_gradient_test
  def test_can_differentiate_scan(self):
    n = 2**4 - 1
    x = self._maybe_static(tf.ones(n, dtype=tf.float64))

    def fn(x):
      y = scan_associative(operator.add, x, max_num_levels=8)
      return tf.tensordot(y, y, 1)

    _, dz_dx = gradient.value_and_gradient(fn, x)

    k = tf.range(n, dtype=tf.float64)
    # Exact result (n + k + 1) * (n - k) computed in Mathematica.
    self.assertAllClose(dz_dx, (n + k + 1) * (n - k))

  def test_inconsistent_lengths_raise_error(self):
    elems0 = self.evaluate(
        uniform.Uniform(-1., 1.).sample([10], seed=test_util.test_seed()))
    elems1 = self.evaluate(
        uniform.Uniform(-1., 1.).sample([9], seed=test_util.test_seed()))

    def extended_add(a, b):
      return (a[0] + b[0], a[1] + b[1])

    with self.assertRaisesRegex(
        Exception, 'Inputs must have the same size along the given axis'):
      self.evaluate(
          scan_associative(
              extended_add,
              (self._maybe_static(elems0), self._maybe_static(elems1)),
              max_num_levels=8,
              validate_args=True))

  def test_max_allowed_size(self):
    elems = self.evaluate(
        uniform.Uniform(-1., 1.).sample([511], seed=test_util.test_seed()))

    result = self.evaluate(
        scan_associative(
            operator.add,
            self._maybe_static(elems),
            max_num_levels=8,
            validate_args=True))
    self.assertAllClose(
        result,
        self.evaluate(tf.cumsum(elems)),
        atol=1e-4)

  def test_min_disallowed_size(self):
    elems = self.evaluate(
        uniform.Uniform(-1., 1.).sample([512], seed=test_util.test_seed()))

    with self.assertRaisesRegex(
        Exception, 'Input `Tensor`s must have dimension less than'):
      self.evaluate(
          scan_associative(
              operator.add,
              self._maybe_static(elems),
              max_num_levels=8,
              validate_args=True))


class ScanAssociativeTestStatic(_ScanAssociativeTest):

  # XLA requires static shapes.
  @test_util.disable_test_for_backend(
      disable_numpy=True,
      reason='No compilation in numpy backend.')
  @test_util.test_graph_and_eager_modes
  def test_cumulative_sum_with_xla(self):
    elems = self._maybe_static(tf.range(0, 2**4 - 1, dtype=tf.int64))

    # JAX jit expects arguments to functions to be DeviceArrays. Thus we
    # curry `scan_associative` so that the resulting function takes in `Tensors`
    # or `DeviceArrays`.

    xla_scan = tf.function(jit_compile=True)(
        functools.partial(scan_associative, operator.add))
    result = xla_scan(elems)

    self.assertAllEqual(
        self.evaluate(result),
        self.evaluate(tf.cumsum(elems)))

  def _maybe_static(self, x):
    return x


# Dynamic-shape tests are only meaningful in graph mode.
class ScanAssociativeTestDynamic(_ScanAssociativeTest):

  def _maybe_static(self, x):
    return tf1.placeholder_with_default(x, shape=None)


del _ScanAssociativeTest


if __name__ == '__main__':
  test_util.main()
