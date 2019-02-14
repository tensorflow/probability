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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from scipy import interpolate as scipy_interpolate
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top


@test_util.run_all_in_graph_and_eager_modes
class InterpRegular1DGridTest(tf.test.TestCase):
  """Test for tfp.math.interp_regular_1d_grid."""

  def setUp(self):
    self.rng = np.random.RandomState(42)

  def test_on_1d_array_nan_fill_value(self):
    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = np.nan  # Will fill this value for points < x_min.
    y_expected[-1] = np.nan  # Will fill this value for points > x_max.

    with self.test_session():
      y = tfp.math.interp_regular_1d_grid(
          x, x_min, x_max, y_ref, fill_value=np.nan)
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_different_below_above_fill_values(self):
    # Simple debuggable test
    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = -111  # Will fill this value for points < x_min.
    y_expected[-1] = 111  # Will fill this value for points > x_max.

    with self.test_session():
      y = tfp.math.interp_regular_1d_grid(
          x, x_min, x_max, y_ref, fill_value_below=-111., fill_value_above=111.)
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_constant_extension_fill_value(self):
    # Simple debuggable test
    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = y_ref[0]  # Will fill this value for points < x_min.
    y_expected[-1] = y_ref[-1]  # Will fill this value for points > x_max.

    with self.test_session():
      y = tfp.math.interp_regular_1d_grid(
          x, x_min, x_max, y_ref, fill_value='constant_extension')
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_extrapolate_fill_value(self):
    # Simple debuggable test
    x_min = 0.
    x_max = 2.
    dtype = np.float32
    num_pts = 3

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = -2.  # Linear extrapolation.
    y_expected[-1] = 4.  # Linear extrapolation.

    with self.test_session():
      y = tfp.math.interp_regular_1d_grid(
          x, x_min, x_max, y_ref, fill_value='extrapolate')
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_log_spaced_grid(self):
    x_min = 1.
    x_max = 100000.
    num_pts = 10

    # With only 10 interpolating points between x_ref = 1 and 100000,
    # and y_ref = log(x_ref), we better use a log-spaced grid, or else error
    # will be very bad.
    implied_x_ref = tf.exp(
        tf.linspace(tf.math.log(x_min), tf.math.log(x_max), num_pts))
    y_ref = tf.math.log(implied_x_ref)

    x = tf.linspace(x_min + 0.123, x_max, 20)
    y_expected = tf.math.log(x)

    with self.test_session():
      y = tfp.math.interp_regular_1d_grid(
          x, x_min, x_max, y_ref, grid_regularizing_transform=tf.math.log)
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      # Super duper accuracy!  Note accuracy was not good if I did not use the
      # grid_regularizing_transform.
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_matrix_interpolation(self):
    # Interpolate a matrix-valued function of one variable.
    # Use only two interpolation points.
    mat_0 = np.array([[1., 0.], [0., 1.]])
    mat_1 = np.array([[0., -1], [1, 0]])
    y_ref = np.array([mat_0, mat_1])

    # Get three output matrices at once.
    x = np.array([0., 0.5, 1.])
    y = tfp.math.interp_regular_1d_grid(
        x, x_ref_min=0., x_ref_max=1., y_ref=y_ref, axis=0)
    self.assertAllEqual((3, 2, 2), y.shape)
    y_ = self.evaluate(y)
    self.assertAllClose(y_[0], mat_0)
    self.assertAllClose(y_[1], 0.5 * mat_0 + 0.5 * mat_1)
    self.assertAllClose(y_[2], mat_1)

    # Get one single output matrix
    y_0p5 = tfp.math.interp_regular_1d_grid(
        x=0.5, x_ref_min=0., x_ref_max=1., y_ref=y_ref, axis=0)
    self.assertAllClose(self.evaluate(y_0p5), 0.5 * mat_0 + 0.5 * mat_1)

  def test_scalar_valued_function_and_get_matrix_of_results(self):
    y_ref = tf.exp(tf.linspace(start=0., stop=10., num=200))
    x = [[1.1, 1.2], [2.1, 2.2]]
    y = tfp.math.interp_regular_1d_grid(
        x, x_ref_min=0., x_ref_max=10., y_ref=y_ref)
    self.assertAllEqual((2, 2), y.shape)
    self.assertAllClose(np.exp(x), self.evaluate(y), rtol=1e-3)

  def _check_sinusoid(self,
                      x_shape,
                      y_ref_shape,
                      axis,
                      fill_value,
                      dtype=np.float32):
    """Check correctness on a sinusoidal function."""
    positive_axis = axis if axis >= 0 else axis + len(y_ref_shape)

    x_ref_min = 0.
    x_ref_max = 2 * np.pi

    # Create implied_x_ref, which has same shape = y_ref_shape.
    implied_x_ref_1d = np.linspace(
        x_ref_min, x_ref_max, y_ref_shape[axis], dtype=dtype)
    implied_x_ref = implied_x_ref_1d.copy()
    for _ in y_ref_shape[positive_axis + 1:]:
      implied_x_ref = implied_x_ref[..., np.newaxis]
    for _ in y_ref_shape[:positive_axis]:
      implied_x_ref = implied_x_ref[np.newaxis, ...]
    implied_x_ref = implied_x_ref + np.zeros(y_ref_shape, dtype=dtype)

    y_ref = np.sin(implied_x_ref)

    # make sure some values are not in [x_ref_min, x_ref_max]
    x = np.linspace(
        x_ref_min - 0.5, x_ref_max + 0.5, np.prod(x_shape),
        dtype=dtype).reshape(x_shape)

    with self.cached_session():
      y = tfp.math.interp_regular_1d_grid(
          x,
          x_ref_min,
          x_ref_max,
          y_ref,
          fill_value=fill_value,
          axis=axis)

      expected_out_shape = (
          y_ref.shape[:positive_axis] + x_shape +
          y_ref.shape[positive_axis + 1:])
      self.assertAllEqual(expected_out_shape, y.shape)

      sp_func = scipy_interpolate.interp1d(
          implied_x_ref_1d,
          y_ref,
          kind='linear',
          axis=axis,
          bounds_error=False,  # Allow values outside range.
          fill_value=fill_value)
      y_expected = sp_func(x)
      self.assertAllClose(self.evaluate(y), y_expected, atol=1e-5, rtol=0)

  def test_axis_1_shape_2_50_3_4_fvextrapolate_32bit(self):
    self._check_sinusoid(
        x_shape=(2, 7),
        y_ref_shape=(2, 50, 3, 4),
        axis=1,
        fill_value='extrapolate',
        dtype=np.float32)

  def test_axis_1_shape_2_50_3_4_fvscalar_64bit(self):
    self._check_sinusoid(
        x_shape=(),
        y_ref_shape=(2, 50, 3, 4),
        axis=1,
        fill_value=np.nan,
        dtype=np.float32)

  def test_axis_n2_shape_2_3_50_4_fvextrapolate_64bit(self):
    self._check_sinusoid(
        x_shape=(10,),
        y_ref_shape=(2, 3, 50, 4),
        axis=-2,
        fill_value='extrapolate',
        dtype=np.float32)

  def test_axis_n1_shape_2_3_4_50_fvextrapolate_32bit(self):
    self._check_sinusoid(
        x_shape=(1, 8),
        y_ref_shape=(2, 3, 4, 50),
        axis=-1,
        fill_value='extrapolate',
        dtype=np.float32)

  def test_axis_3_shape_2_3_4_50_fvscalar_32bit(self):
    self._check_sinusoid(
        x_shape=(5,),
        y_ref_shape=(2, 3, 4, 50),
        axis=3,
        fill_value=np.inf,
        dtype=np.float32)

  def test_gradients_and_propagation_of_nan_in_x(self):
    # If x contains NaN, this should propagate through to y, and not mess up the
    # gradients associated with finite members of x.
    # In fact, even NaN members of x result in finite (zero) gradients.

    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    x_ = np.array([0., 0.1, np.nan, 0.4, 1.]).astype(dtype)
    y_expected = 2 * x_

    x = tf.constant(x_)

    with self.test_session():
      y = tfp.math.interp_regular_1d_grid(x, x_min, x_max, y_ref)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)
      if not tf.executing_eagerly():
        dy_dx_ = self.evaluate(tf.gradients(ys=y, xs=x)[0])
        self.assertAllClose([2., 2., 0., 2., 2.], dy_dx_)


@test_util.run_all_in_graph_and_eager_modes
class BatchInterpRegular1DGridTest(tf.test.TestCase):
  """Test for 1-D usage of tfp.math.interp_regular_1d_grid."""

  def setUp(self):
    self.rng = np.random.RandomState(42)

  def test_on_1d_array_nan_fill_value(self):
    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = np.nan  # Will fill this value for points < x_min.
    y_expected[-1] = np.nan  # Will fill this value for points > x_max.

    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x, x_min, x_max, y_ref, fill_value=np.nan)
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_different_below_above_fill_values(self):
    # Simple debuggable test
    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = -111  # Will fill this value for points < x_min.
    y_expected[-1] = 111  # Will fill this value for points > x_max.

    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x,
          x_min,
          x_max,
          y_ref,
          fill_value_below=-111.,
          fill_value_above=111.)
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_constant_extension_fill_value(self):
    # Simple debuggable test
    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = y_ref[0]  # Will fill this value for points < x_min.
    y_expected[-1] = y_ref[-1]  # Will fill this value for points > x_max.

    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x,
          x_min,
          x_max,
          y_ref,
          fill_value='constant_extension')
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_extrapolate_fill_value(self):
    # Simple debuggable test
    x_min = 0.
    x_max = 2.
    dtype = np.float32
    num_pts = 3

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    # First and last values of x are outside the [x_min, x_max] range.
    x = np.array([-1., 0.1, 0.2, 0.4, 2.]).astype(dtype)
    y_expected = 2 * x
    y_expected[0] = -2.  # Linear extrapolation.
    y_expected[-1] = 4.  # Linear extrapolation.

    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x, x_min, x_max, y_ref, fill_value='extrapolate')
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_on_1d_array_log_spaced_grid(self):
    x_min = 1.
    x_max = 100000.
    num_pts = 10

    # With only 10 interpolating points between x_ref = 1 and 100000,
    # and y_ref = log(x_ref), we better use a log-spaced grid, or else error
    # will be very bad.
    implied_x_ref = tf.exp(
        tf.linspace(tf.math.log(x_min), tf.math.log(x_max), num_pts))
    y_ref = tf.math.log(implied_x_ref)

    x = tf.linspace(x_min + 0.123, x_max, 20)
    y_expected = tf.math.log(x)

    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x, x_min, x_max, y_ref, grid_regularizing_transform=tf.math.log)
      self.assertAllEqual(y_expected.shape, y.shape)
      y_ = self.evaluate(y)
      # Super duper accuracy!  Note accuracy was not good if I did not use the
      # grid_regularizing_transform.
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)

  def test_scalar_valued_interpolation_with_1_batch_dim(self):
    # First batch member is an exponential function, second is a log.
    implied_x_ref = [tf.linspace(-3., 3.2, 200), tf.linspace(0.5, 3., 200)]
    y_ref = tf.stack(  # Shape [2, 200]
        [tf.exp(implied_x_ref[0]),
         tf.math.log(implied_x_ref[1])],
        axis=0)

    x = [[-1., 1., 0.],  # Shape [2, 3].  Batch=2, 3 values per batch.
         [1., 2., 3.]]
    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x,
          x_ref_min=[-3., 0.5],
          x_ref_max=[3.2, 3.],
          y_ref=y_ref,
          axis=-1)
      self.assertAllEqual((2, 3), y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(np.exp(x[0]), y_[0], rtol=1e-3)
      self.assertAllClose(np.log(x[1]), y_[1], rtol=1e-3, atol=1e-3)

  def test_scalar_valued_interpolation_with_1_batch_dim_x_has_empty_batch(self):
    # First batch member is an exponential function, second is a log.
    implied_x_ref = [tf.linspace(-3., 3.2, 200), tf.linspace(0.5, 3., 200)]
    y_ref = tf.stack(  # Shape [2, 200]
        [tf.exp(implied_x_ref[0]),
         tf.math.log(implied_x_ref[1])],
        axis=0)

    # x.shape = [3] ==> No batch dim.
    # Values of x will be broadcast so effectively x.shape = [2, 3]
    x = [1., 1.12, 2.11]
    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x,
          x_ref_min=[-3., 0.5],
          x_ref_max=[3.2, 3.],
          y_ref=y_ref,
          axis=-1)
      self.assertAllEqual((2, 3), y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(np.exp(x), y_[0], rtol=1e-3)
      self.assertAllClose(np.log(x), y_[1], rtol=1e-3, atol=1e-3)

  def test_scalar_valued_with_1_batch_dim_x_and_x_minmax_have_empty_batch(self):
    implied_x_ref = tf.linspace(-3., 3.2, 200)
    y_ref = tf.stack(  # Shape [2, 200]
        [tf.exp(implied_x_ref), tf.exp(2 * implied_x_ref)], axis=0)

    # x.shape = [3] ==> No batch dim.
    # Values of x will be broadcast so effectively x.shape = [2, 3]
    x = np.array([1., 1.12, 2.11], dtype=np.float32)
    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x,
          x_ref_min=-3.,
          x_ref_max=3.2,
          y_ref=y_ref,
          axis=-1)
      self.assertAllEqual((2, 3), y.shape)
      y_ = self.evaluate(y)
      self.assertAllClose(np.exp(x), y_[0], rtol=1e-3)
      self.assertAllClose(np.exp(2 * x), y_[1], rtol=1e-3)

  def test_vector_valued_interpolation_with_empty_batch_shape(self):
    implied_x_ref = tf.linspace(1., 3., 200)  # Shape [200]
    y_ref = tf.stack(  # Shape [200, 2]
        [tf.exp(implied_x_ref),
         tf.math.log(implied_x_ref)],
        axis=-1)

    # x has no batch dims, which it shouldn't since axis=0.
    x = [1., 2., 3.]
    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(
          x, x_ref_min=1., x_ref_max=3., y_ref=y_ref, axis=0)
      self.assertAllEqual((3, 2), y.shape)
      y_ = self.evaluate(y)
      expected_y = np.stack([np.exp(x), np.log(x)], axis=-1)
      self.assertAllClose(y_, expected_y, rtol=1e-3, atol=1e-3)

  def test_vector_valued_interpolation_with_1_batch_dim_scalar_minmax(self):
    # implied_x_ref.shape = [3, 200], with all 3 batch members having the same
    # min and max (hence, "scalar minmax").
    implied_x_ref = np.array([np.linspace(1., 3., 200)] * 3)
    y_ref = np.stack(  # Shape [3, 200, 2]
        [np.exp(implied_x_ref), np.exp(2 * implied_x_ref)], axis=-1)

    x = 1 + 2 * self.rng.rand(3, 5)
    with self.test_session():
      # Interpolate along axis 1, the axis with 200 elements.
      y = tfp.math.batch_interp_regular_1d_grid(
          x, x_ref_min=1., x_ref_max=3., y_ref=y_ref, axis=1)
      self.assertAllEqual((3, 5, 2), y.shape)
      y_ = self.evaluate(y)
      expected_y = np.stack([np.exp(x), np.exp(2 * x)], axis=-1)
      self.assertAllClose(y_, expected_y, rtol=1e-3, atol=1e-3)

  def test_vector_valued_interpolation_with_1_batch_dim_batch_minmax(self):
    # implied_x_ref.shape = [2, 200], with the 2 batch members having different
    # min/max (hence, "batch minmax").
    # The first batch member will be used for exponentials,
    # The second batch member will be used for logs.
    implied_x_ref = np.stack(  # Shape [2, 200]
        [np.linspace(-1., 2, 200), np.linspace(1., 3., 200)], axis=0)
    y_exp = np.stack(  # Shape [200, 2]
        [np.exp(implied_x_ref[0]), np.exp(2 * implied_x_ref[0])], axis=-1)
    y_log = np.stack(  # Shape [200, 2]
        [np.log(implied_x_ref[1]), np.log(2 * implied_x_ref[1])], axis=-1)
    y_ref = np.stack([y_exp, y_log], axis=0)  # Shape [2, 200, 2]

    x = np.stack(  # Shape [2, 3]
        [[-0.5, 0., 1.], [1.12, 2.34, 2.5]], axis=0)
    with self.test_session():
      # Interpolate along axis 1, the axis with 200 elements.
      y = tfp.math.batch_interp_regular_1d_grid(
          x,
          x_ref_min=[-1., 1],
          x_ref_max=[2., 3.],
          y_ref=y_ref,
          axis=1)
      self.assertAllEqual((2, 3, 2), y.shape)
      y_ = self.evaluate(y)
      expected_y_exp = np.stack(  # Shape [3, 2]
          [np.exp(x[0]), np.exp(2 * x[0])], axis=-1)
      expected_y_log = np.stack(  # Shape [3, 2]
          [np.log(x[1]), np.log(2 * x[1])], axis=-1)
      expected_y = np.stack(  # Shape [2, 3, 2]
          [expected_y_exp, expected_y_log], axis=0)
      self.assertAllClose(y_, expected_y, rtol=1e-3, atol=1e-3)

  def test_gradients_and_propagation_of_nan_in_x(self):
    # If x contains NaN, this should propagate through to y, and not mess up the
    # gradients associated with finite members of x.
    # In fact, even NaN members of x result in finite (zero) gradients.

    x_min = 0.
    x_max = 1.
    dtype = np.float32
    num_pts = 4

    implied_x_ref = np.linspace(x_min, x_max, num_pts, dtype=dtype)
    y_ref = 2 * implied_x_ref

    x_ = np.array([0., 0.1, np.nan, 0.4, 1.]).astype(dtype)
    y_expected = 2 * x_

    x = tf.constant(x_)

    with self.test_session():
      y = tfp.math.batch_interp_regular_1d_grid(x, x_min, x_max, y_ref)
      y_ = self.evaluate(y)
      self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)
      if not tf.executing_eagerly():
        dy_dx_ = self.evaluate(tf.gradients(ys=y, xs=x)[0])
        self.assertAllClose([2., 2., 0., 2., 2.], dy_dx_)


if __name__ == '__main__':
  tf.test.main()
