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
"""Tests for Special Math Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import importlib

# Dependency imports
import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.math.gradient import value_and_gradient

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


def try_import(name):  # pylint: disable=invalid-name
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf.compat.v1.logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")
stats = try_import("scipy.stats")


def _check_strictly_increasing(array_1d):
  diff = np.diff(array_1d)
  np.testing.assert_array_less(0, diff)


def _make_grid(dtype, grid_spec):
  """Returns a uniform grid + noise, reshaped to shape argument."""
  rng = np.random.RandomState(0)
  num_points = np.prod(grid_spec.shape)
  grid = np.linspace(grid_spec.min, grid_spec.max, num=num_points).astype(dtype)
  grid_spacing = (grid_spec.max - grid_spec.min) / num_points
  grid += 0.1 * grid_spacing * rng.randn(*grid.shape)
  # More useful if it's sorted (e.g. for testing monotonicity, or debugging).
  grid = np.sort(grid)
  return np.reshape(grid, grid_spec.shape)


GridSpec = collections.namedtuple("GridSpec", ["min", "max", "shape"])

ErrorSpec = collections.namedtuple("ErrorSpec", ["rtol", "atol"])


@test_util.run_all_in_graph_and_eager_modes
class NdtriTest(tf.test.TestCase):

  def assertAllFinite(self, x):
    is_finite = np.isfinite(x)
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def testNdtri(self):
    """Verifies that ndtri computation is correct."""
    if not special:
      return

    p = np.linspace(0., 1., 50).astype(np.float64)
    # Quantile performs piecewise rational approximation so adding some
    # special input values to make sure we hit all the pieces.
    p = np.hstack((p, np.exp(-32), 1. - np.exp(-32), np.exp(-2),
                   1. - np.exp(-2)))
    expected_x = special.ndtri(p)
    x = special_math.ndtri(p)
    self.assertAllClose(expected_x, self.evaluate(x), atol=0.)

  def testNdtriDynamicShape(self):
    """Verifies that ndtri computation is correct."""
    if not special:
      return
    p_ = np.linspace(0., 1., 50).astype(np.float32)
    p = tf.compat.v1.placeholder_with_default(p_, shape=None)
    self.assertAllClose(special.ndtri(p_),
                        self.evaluate(special_math.ndtri(p)),
                        atol=0.)

  def _baseNdtriFiniteGradientTest(self, dtype):
    """Verifies that ndtri has finite gradients at interesting points."""
    # Tests gradients at 0, 1, and piece-wise boundaries.
    p = tf.constant(
        np.array([
            0.,
            np.exp(-32.),
            np.exp(-2.),
            1. - np.exp(-2.),
            1. - np.exp(-32.),
            1.,
        ]).astype(dtype))
    # Not having the lambda sanitzer means we'd get an `IndexError` whenever
    # the user supplied function has default args.
    _, grads = value_and_gradient(special_math.ndtri, p)
    self.assertAllFinite(self.evaluate(grads[0]))

  def testNdtriFiniteGradientFloat32(self):
    self._baseNdtriFiniteGradientTest(np.float32)

  def testNdtriFiniteGradientFloat64(self):
    self._baseNdtriFiniteGradientTest(np.float64)


@test_util.run_all_in_graph_and_eager_modes
class NdtrTest(tf.test.TestCase):
  _use_log = False
  # Grid min/max chosen to ensure 0 < cdf(x) < 1.
  _grid32 = GridSpec(min=-12.9, max=5., shape=[100])
  _grid64 = GridSpec(min=-37.5, max=8., shape=[100])
  _error32 = ErrorSpec(rtol=1e-4, atol=0.)
  _error64 = ErrorSpec(rtol=1e-6, atol=0.)

  def _test_grid(self, dtype, grid_spec, error_spec):
    if self._use_log:
      self._test_grid_log(dtype, grid_spec, error_spec)
    else:
      self._test_grid_no_log(dtype, grid_spec, error_spec)

  def _test_grid_log(self, dtype, grid_spec, error_spec):
    if not special:
      return

    grid = _make_grid(dtype, grid_spec)
    actual = self.evaluate(special_math.log_ndtr(grid))

    # Basic tests.
    # isfinite checks for NaN and Inf.
    self.assertTrue(np.isfinite(actual).all())
    # On the grid, -inf < log_cdf(x) < 0.  In this case, we should be able
    # to use a huge grid because we have used tricks to escape numerical
    # difficulties.
    self.assertTrue((actual < 0).all())
    _check_strictly_increasing(actual)

    # Versus scipy.
    expected = special.log_ndtr(grid)
    # Scipy prematurely goes to zero at some places that we don't.  So don't
    # include these in the comparison.
    self.assertAllClose(
        expected.astype(np.float64)[expected < 0],
        actual.astype(np.float64)[expected < 0],
        rtol=error_spec.rtol,
        atol=error_spec.atol)

  def _test_grid_no_log(self, dtype, grid_spec, error_spec):
    if not special:
      return

    grid = _make_grid(dtype, grid_spec)
    actual = self.evaluate(special_math.ndtr(grid))

    # Basic tests.
    # isfinite checks for NaN and Inf.
    self.assertTrue(np.isfinite(actual).all())
    # On the grid, 0 < cdf(x) < 1.  The grid cannot contain everything due
    # to numerical limitations of cdf.
    self.assertTrue((actual > 0).all())
    self.assertTrue((actual < 1).all())
    _check_strictly_increasing(actual)

    # Versus scipy.
    expected = special.ndtr(grid)
    # Scipy prematurely goes to zero at some places that we don't.  So don't
    # include these in the comparison.
    self.assertAllClose(
        expected.astype(np.float64)[expected < 0],
        actual.astype(np.float64)[expected < 0],
        rtol=error_spec.rtol,
        atol=error_spec.atol)

  def test_float32(self):
    self._test_grid(np.float32, self._grid32, self._error32)

  def test_float64(self):
    self._test_grid(np.float64, self._grid64, self._error64)


@test_util.run_all_in_graph_and_eager_modes
class LogNdtrTestLower(NdtrTest):
  _use_log = True
  _grid32 = GridSpec(
      min=-100.,
      max=special_math.LOGNDTR_FLOAT32_LOWER, shape=[100])
  _grid64 = GridSpec(
      min=-100.,
      max=special_math.LOGNDTR_FLOAT64_LOWER, shape=[100])
  _error32 = ErrorSpec(rtol=1e-4, atol=0.)
  _error64 = ErrorSpec(rtol=1e-4, atol=0.)


# The errors are quite large when the input is > 6 or so.  Also,
# scipy.special.log_ndtr becomes zero very early, before 10,
# (due to ndtr becoming 1).  We approximate Log[1 + epsilon] as epsilon, and
# avoid this issue.
@test_util.run_all_in_graph_and_eager_modes
class LogNdtrTestMid(NdtrTest):
  _use_log = True
  _grid32 = GridSpec(
      min=special_math.LOGNDTR_FLOAT32_LOWER,
      max=special_math.LOGNDTR_FLOAT32_UPPER, shape=[100])
  _grid64 = GridSpec(
      min=special_math.LOGNDTR_FLOAT64_LOWER,
      max=special_math.LOGNDTR_FLOAT64_UPPER, shape=[100])
  # Differences show up as soon as we're in the tail, so add some atol.
  _error32 = ErrorSpec(rtol=0.1, atol=1e-7)
  _error64 = ErrorSpec(rtol=0.1, atol=1e-7)


@test_util.run_all_in_graph_and_eager_modes
class LogNdtrTestUpper(NdtrTest):
  _use_log = True
  _grid32 = GridSpec(
      min=special_math.LOGNDTR_FLOAT32_UPPER,
      max=12.,  # Beyond this, log_cdf(x) may be zero.
      shape=[100])
  _grid64 = GridSpec(
      min=special_math.LOGNDTR_FLOAT64_UPPER,
      max=35.,  # Beyond this, log_cdf(x) may be zero.
      shape=[100])
  _error32 = ErrorSpec(rtol=1e-6, atol=1e-14)
  _error64 = ErrorSpec(rtol=1e-6, atol=1e-14)


@test_util.run_all_in_graph_and_eager_modes
class NdtrGradientTest(tf.test.TestCase):
  _use_log = False
  _grid = GridSpec(min=-100., max=100., shape=[1, 2, 3, 8])
  _error32 = ErrorSpec(rtol=1e-4, atol=0)
  _error64 = ErrorSpec(rtol=1e-7, atol=0)

  def assert_all_true(self, v):
    self.assertAllEqual(np.ones_like(v, dtype=np.bool), v)

  def assert_all_false(self, v):
    self.assertAllEqual(np.zeros_like(v, dtype=np.bool), v)

  def _test_grad_finite(self, dtype):
    x = tf.constant([-100., 0., 100.], dtype=dtype)
    output = (special_math.log_ndtr(x) if self._use_log
              else special_math.ndtr(x))
    fn = special_math.log_ndtr if self._use_log else special_math.ndtr
    # Not having the lambda sanitzer means we'd get an `IndexError` whenever
    # the user supplied function has default args.
    output, grad_output = value_and_gradient(fn, x)
    # isfinite checks for NaN and Inf.
    output_, grad_output_ = self.evaluate([output, grad_output])
    self.assert_all_true(np.isfinite(output_))
    self.assert_all_true(np.isfinite(grad_output_[0]))

  def _test_grad_accuracy(self, dtype, grid_spec, error_spec):
    grid = _make_grid(dtype, grid_spec)
    _, actual_grad = self.evaluate(value_and_gradient(
        special_math.log_ndtr if self._use_log else special_math.ndtr, grid))

    # Check for NaN separately in order to get informative failures.
    self.assert_all_false(np.isnan(actual_grad))
    if self._use_log:
      g = np.reshape(actual_grad, [-1])
      half = np.ceil(len(g) / 2)
      self.assert_all_true(g[:int(half)] > 0.)
      self.assert_all_true(g[int(half):] >= 0.)
    else:
      # The ndtr gradient will only be non-zero in the range [-14, 14] for
      # float32 and [-38, 38] for float64.
      self.assert_all_true(actual_grad >= 0.)
    # isfinite checks for NaN and Inf.
    self.assert_all_true(np.isfinite(actual_grad))

    # Versus scipy.
    if not (special and stats):
      return

    expected_grad = stats.norm.pdf(grid)
    if self._use_log:
      expected_grad /= special.ndtr(grid)
      expected_grad[np.isnan(expected_grad)] = 0.
    # Scipy prematurely goes to zero at some places that we don't.  So don't
    # include these in the comparison.
    self.assertAllClose(
        expected_grad.astype(np.float64)[expected_grad < 0],
        actual_grad.astype(np.float64)[expected_grad < 0],
        rtol=error_spec.rtol,
        atol=error_spec.atol)

  def test_float32(self):
    self._test_grad_accuracy(np.float32, self._grid, self._error32)
    self._test_grad_finite(np.float32)

  def test_float64(self):
    self._test_grad_accuracy(np.float64, self._grid, self._error64)
    self._test_grad_finite(np.float64)


@test_util.run_all_in_graph_and_eager_modes
class LogNdtrGradientTest(NdtrGradientTest):
  _use_log = True


@test_util.run_all_in_graph_and_eager_modes
class ErfInvTest(tf.test.TestCase):

  def testErfInvValues(self):
    if not special:
      return
    x = np.linspace(0., 1., 50).astype(np.float64)
    self.assertAllClose(special.erfinv(x),
                        self.evaluate(special_math.erfinv(x)),
                        atol=0)

  def testErfInvIntegerInput(self):
    with self.assertRaises(TypeError):
      x = np.array([1, 2, 3]).astype(np.int32)
      special_math.erfinv(x)

    with self.assertRaises(TypeError):
      x = np.array([1, 2, 3]).astype(np.int64)
      special_math.erfinv(x)


@test_util.run_all_in_graph_and_eager_modes
class LogCDFLaplaceTest(tf.test.TestCase):
  # Note that scipy.stats.laplace does not have a stable Log CDF, so we cannot
  # rely on scipy to cross check the extreme values.

  # Test will be done differently over different ranges.  These are the values
  # such that when exceeded by x, produce output that causes the naive (scipy)
  # implementation to have numerical issues.
  #
  # If x = log(1 / (2 * eps)), then 0.5 * exp{-x} = eps.
  # With inserting eps = np.finfo(dtype).eps, we see that log(1 / (2 * eps)) is
  # the value of x such that any larger value will result in
  # 1 - 0.5 * exp{-x} = 0, which will cause the log_cdf_laplace code to take a
  # log # of zero.  We therefore choose these as our cutoffs for testing.
  CUTOFF_FLOAT64_UPPER = np.log(1. / (2. * np.finfo(np.float64).eps)) - 1.
  CUTOFF_FLOAT32_UPPER = np.log(1. / (2. * np.finfo(np.float32).eps)) - 1.

  def assertAllTrue(self, x):
    self.assertAllEqual(np.ones_like(x, dtype=np.bool), x)

  def _test_grid_log(self, dtype, scipy_dtype, grid_spec, error_spec):
    grid = _make_grid(dtype, grid_spec)
    actual = self.evaluate(special_math.log_cdf_laplace(grid))

    # Basic tests.
    # isfinite checks for NaN and Inf.
    self.assertAllTrue(np.isfinite(actual))
    self.assertAllTrue((actual < 0))
    _check_strictly_increasing(actual)

    # Versus scipy.
    if not stats:
      return

    scipy_dist = stats.laplace(loc=0., scale=1.)
    expected = scipy_dist.logcdf(grid.astype(scipy_dtype))
    self.assertAllClose(
        expected.astype(np.float64),
        actual.astype(np.float64),
        rtol=error_spec.rtol,
        atol=error_spec.atol)

  def test_float32_lower_and_mid_segment_scipy_float32_ok(self):
    # Choose values mild enough that we can use scipy in float32, which will
    # allow for a high accuracy match to scipy (since we both use float32).
    self._test_grid_log(
        np.float32,  # dtype
        np.float32,  # scipy_dtype
        GridSpec(min=-10, max=self.CUTOFF_FLOAT32_UPPER - 5, shape=[100]),
        ErrorSpec(rtol=5e-4, atol=0))

  def test_float32_all_segments_with_scipy_float64_ok(self):
    # Choose values outside the range where scipy float32 works.
    # Let scipy use float64.  This means we
    # won't be exactly the same since we are in float32.
    self._test_grid_log(
        np.float32,  # dtype
        np.float64,  # scipy_dtype
        GridSpec(min=-50, max=self.CUTOFF_FLOAT32_UPPER + 5, shape=[100]),
        ErrorSpec(rtol=0.05, atol=0))

  def test_float32_extreme_values_result_and_gradient_finite_and_nonzero(self):
    # On the lower branch, log_cdf_laplace(x) = x, so we know this will be
    # fine, but test to -200 anyways.
    grid = _make_grid(
        np.float32, GridSpec(min=-200, max=80, shape=[20, 100]))
    grid = tf.convert_to_tensor(value=grid)

    actual, grad = value_and_gradient(special_math.log_cdf_laplace, grid)
    actual_, grad_ = self.evaluate([actual, grad])

    # isfinite checks for NaN and Inf.
    self.assertAllTrue(np.isfinite(actual_))
    self.assertAllTrue(np.isfinite(grad_))
    self.assertFalse(np.any(actual_ == 0))
    self.assertFalse(np.any(grad_ == 0))

  def test_float64_extreme_values_result_and_gradient_finite_and_nonzero(self):
    # On the lower branch, log_cdf_laplace(x) = x, so we know this will be
    # fine, but test to -200 anyways.
    grid = _make_grid(
        np.float64, GridSpec(min=-200, max=700, shape=[20, 100]))
    grid = tf.convert_to_tensor(value=grid)

    actual, grad = value_and_gradient(special_math.log_cdf_laplace, grid)
    actual_, grad_ = self.evaluate([actual, grad])

    # isfinite checks for NaN and Inf.
    self.assertAllTrue(np.isfinite(actual_))
    self.assertAllTrue(np.isfinite(grad_))
    self.assertFalse(np.any(actual_ == 0))
    self.assertFalse(np.any(grad_ == 0))


if __name__ == "__main__":
  tf.test.main()
