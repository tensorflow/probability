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
"""Tests for RQ Spline bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging
import hypothesis as hp
from hypothesis import strategies as hps
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python.bijectors import bijector_test_util
from tensorflow_probability.python.bijectors import hypothesis_testlib as bijector_hps
from tensorflow_probability.python.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.python.internal import test_util

# pylint: disable=no-value-for-parameter


@hps.composite
def rq_splines(draw, batch_shape=None, dtype=tf.float32):
  if batch_shape is None:
    batch_shape = draw(tfp_hps.shapes())

  lo = draw(hps.floats(min_value=-5, max_value=.5))
  hi = draw(hps.floats(min_value=-.5, max_value=5))
  lo, hi = min(lo, hi), max(lo, hi) + .2

  constraints = dict(
      bin_widths=functools.partial(
          bijector_hps.spline_bin_size_constraint, hi=hi, lo=lo, dtype=dtype),
      bin_heights=functools.partial(
          bijector_hps.spline_bin_size_constraint, hi=hi, lo=lo, dtype=dtype),
      knot_slopes=functools.partial(
          bijector_hps.spline_slope_constraint, dtype=dtype))
  params = draw(
      tfp_hps.broadcasting_params(
          batch_shape,
          params_event_ndims=dict(bin_widths=1, bin_heights=1, knot_slopes=1),
          constraint_fn_for=constraints.get))
  return tfb.RationalQuadraticSpline(
      range_min=lo, validate_args=draw(hps.booleans()), **params)


@test_util.test_all_tf_execution_regimes
class RationalQuadraticSplineTest(test_util.TestCase):

  def testDocExample(self):

    nsplits = 3

    class SplineParams(tf.Module):

      def __init__(self, nbins=32):
        self._nbins = nbins
        self._built = False
        self._bin_widths = None
        self._bin_heights = None
        self._knot_slopes = None

      def _bin_positions(self, x):
        x = tf.reshape(x, [-1, self._nbins])
        return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

      def _slopes(self, x):
        x = tf.reshape(x, [-1, self._nbins - 1])
        return tf.math.softplus(x) + 1e-2

      def __call__(self, x, nunits):
        if not self._built:
          self._bin_widths = tf.keras.layers.Dense(
              nunits * self._nbins, activation=self._bin_positions, name='w')
          self._bin_heights = tf.keras.layers.Dense(
              nunits * self._nbins, activation=self._bin_positions, name='h')
          self._knot_slopes = tf.keras.layers.Dense(
              nunits * (self._nbins - 1), activation=self._slopes, name='s')
          self._built = True
        return tfb.RationalQuadraticSpline(
            bin_widths=self._bin_widths(x),
            bin_heights=self._bin_heights(x),
            knot_slopes=self._knot_slopes(x))

    xs = np.random.randn(1, 15).astype(np.float32)  # Keras won't Dense(.)(vec).
    splines = [SplineParams() for _ in range(nsplits)]

    def spline_flow():
      stack = tfb.Identity()
      for i in range(nsplits):
        stack = tfb.RealNVP(5 * i, bijector_fn=splines[i])(stack)
      return stack

    ys = spline_flow().forward(xs)
    ys_inv = spline_flow().inverse(ys)  # reconstruct ensures no cache hit.

    init_vars = []
    for s in splines:
      init_vars += [v.initializer for v in s.variables]
    self.evaluate(init_vars)
    self.assertAllClose(xs, self.evaluate(ys_inv))

  def testDegenerateSplines(self):
    bijector = tfb.RationalQuadraticSpline([], [], 1, validate_args=True)
    xs = np.linspace(-2, 2, 20, dtype=np.float32)
    self.assertAllClose(xs, self.evaluate(bijector.forward(xs)))
    self.assertAllClose(
        0, self.evaluate(bijector.forward_log_det_jacobian(xs, event_ndims=1)))
    self.assertAllClose(
        np.zeros_like(xs),
        self.evaluate(bijector.forward_log_det_jacobian(xs, event_ndims=0)))

    bijector = tfb.RationalQuadraticSpline([2.], [2.], [], validate_args=True)
    xs = np.linspace(-2, 2, 20, dtype=np.float32)
    self.assertAllClose(xs, self.evaluate(bijector.forward(xs)))
    self.assertAllClose(
        0, self.evaluate(bijector.forward_log_det_jacobian(xs, event_ndims=1)))
    self.assertAllClose(
        np.zeros_like(xs),
        self.evaluate(bijector.forward_log_det_jacobian(xs, event_ndims=0)))

  def testTheoreticalFldjSimple(self):
    bijector = tfb.RationalQuadraticSpline(
        bin_widths=[1., 1],
        bin_heights=[np.sqrt(.5), 2 - np.sqrt(.5)],
        knot_slopes=1)
    self.assertEqual(tf.float64, bijector.dtype)
    dim = 5
    x = np.linspace(-1.05, 1.05, num=2 * dim, dtype=np.float64).reshape(2, dim)
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=0,
        inverse_event_ndims=0,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=0)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector, x, event_ndims=0)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=5)
  def testTheoreticalFldj(self, data):
    # get_fldj_theoretical test rig requires 1-d batches.
    batch_shape = data.draw(tfp_hps.shapes(min_ndims=1, max_ndims=1))
    bijector = data.draw(rq_splines(batch_shape=batch_shape, dtype=tf.float64))
    self.assertEqual(tf.float64, bijector.dtype)
    bw, bh, kd = self.evaluate(
        [bijector.bin_widths, bijector.bin_heights, bijector.knot_slopes])
    logging.info('bw: %s\nbh: %s\nkd: %s', bw, bh, kd)
    x_shp = ((bw + bh)[..., :-1] + kd).shape[:-1]
    if x_shp[-1] == 1:  # Possibly broadcast the x dim.
      dim = data.draw(hps.integers(min_value=1, max_value=7))
      x_shp = x_shp[:-1] + (dim,)
    x = np.linspace(-5, 5, np.prod(x_shp), dtype=np.float64).reshape(*x_shp)
    y = self.evaluate(bijector.forward(x))
    bijector_test_util.assert_bijective_and_finite(
        bijector,
        x,
        y,
        eval_func=self.evaluate,
        event_ndims=0,
        inverse_event_ndims=0,
        rtol=1e-5)
    fldj = bijector.forward_log_det_jacobian(x, event_ndims=0)
    fldj_theoretical = bijector_test_util.get_fldj_theoretical(
        bijector, x, event_ndims=0)
    self.assertAllClose(
        self.evaluate(fldj_theoretical),
        self.evaluate(fldj),
        atol=1e-5,
        rtol=1e-5)

  def testVerifiesBroadcastingStatic(self):
    with self.assertRaisesRegex(ValueError, '`bin_heights` must broadcast'):
      tfb.RationalQuadraticSpline([[2, 1, .5]] * 2, [[.5, 2, 1]] * 3, [.3, 2])

    with self.assertRaisesRegex(ValueError,
                                'non-scalar `knot_slopes` must broadcast'):
      tfb.RationalQuadraticSpline([2, 1, .5], [.5, 2, 1], [.3, 2, .5])

  def testVerifiesBroadcastingDynamic(self):

    @tf.function
    def f(bin_sizes, slopes):
      return tfb.RationalQuadraticSpline(
          bin_sizes, bin_sizes, slopes, validate_args=True).forward(bin_sizes)

    f = f.get_concrete_function(
        tf.TensorSpec((None,), dtype=tf.float32),
        tf.TensorSpec((None,), dtype=tf.float32))

    with self.assertRaisesOpError('Incompatible shapes'):
      self.evaluate(f(tf.constant([1., 1, 1, 1]), tf.constant([2., 3])))

  def testAssertsMismatchedSums(self):
    with self.assertRaisesOpError(r'`sum\(bin_widths, axis=-1\)` must equal '
                                  r'`sum\(bin_heights, axis=-1\)`'):
      bijector = tfb.RationalQuadraticSpline(
          bin_widths=[.2, .1, .5],
          bin_heights=[.1, .3, 5.4],
          knot_slopes=[.3, .5],
          validate_args=True)
      self.evaluate(bijector.forward([.3]))

  def testAssertsNonPositiveBinSizes(self):
    with self.assertRaisesOpError('`bin_widths` must be positive'):
      bijector = tfb.RationalQuadraticSpline(
          bin_widths=[.3, .2, -.1],
          bin_heights=[.1, .2, .1],
          knot_slopes=[.4, .5],
          validate_args=True)
      self.evaluate(bijector.forward([.3]))

    with self.assertRaisesOpError('`bin_heights` must be positive'):
      bijector = tfb.RationalQuadraticSpline(
          bin_widths=[.3, .2, .1],
          bin_heights=[.5, 0, .1],
          knot_slopes=[.3, .7],
          validate_args=True)
      self.evaluate(bijector.forward([.3]))

  def testAssertsNonPositiveSlope(self):
    with self.assertRaisesOpError('`knot_slopes` must be positive'):
      bijector = tfb.RationalQuadraticSpline(
          bin_widths=[.1, .2, 1],
          bin_heights=[1, .2, .1],
          knot_slopes=[-.5, 1],
          validate_args=True)
      self.evaluate(bijector.forward([.3]))

    with self.assertRaisesOpError('`knot_slopes` must be positive'):
      bijector = tfb.RationalQuadraticSpline(
          bin_widths=[.1, .2, 1],
          bin_heights=[1, .2, .1],
          knot_slopes=[1, 0.],
          validate_args=True)
      self.evaluate(bijector.forward([.3]))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
