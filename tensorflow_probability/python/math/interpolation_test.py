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
import numpy as onp  # pylint: disable=reimported
from scipy import interpolate as scipy_interpolate

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from tensorflow_probability.python.internal import test_util


@test_util.test_all_tf_execution_regimes
class InterpRegular1DGridTest(test_util.TestCase):
  """Test for tfp.math.interp_regular_1d_grid."""

  def setUp(self):
    super(InterpRegular1DGridTest, self).setUp()
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
        x_ref_min - 0.5, x_ref_max + 0.5, np.int32(np.prod(x_shape)),
        dtype=dtype).reshape(x_shape)

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

    y = tfp.math.interp_regular_1d_grid(x, x_min, x_max, y_ref)
    y_ = self.evaluate(y)
    self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)
    if not tf.executing_eagerly():
      dy_dx_ = self.evaluate(tf.gradients(ys=y, xs=x)[0])
      self.assertAllClose([2., 2., 0., 2., 2.], dy_dx_)


@test_util.test_all_tf_execution_regimes
class BatchInterpRegular1DGridTest(test_util.TestCase):
  """Test for 1-D usage of tfp.math.interp_regular_1d_grid."""

  def setUp(self):
    super(BatchInterpRegular1DGridTest, self).setUp()
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
    implied_x_ref = tf.linspace(np.array(-3., dtype=np.float32),
                                3.2, 200)
    y_ref = tf.stack(  # Shape [2, 200]
        [tf.exp(implied_x_ref), tf.exp(2 * implied_x_ref)], axis=0)

    # x.shape = [3] ==> No batch dim.
    # Values of x will be broadcast so effectively x.shape = [2, 3]
    x = np.array([1., 1.12, 2.11], dtype=np.float32)
    y = tfp.math.batch_interp_regular_1d_grid(
        x,
        x_ref_min=np.array(-3., dtype=np.float32),
        x_ref_max=np.array(3.2, dtype=np.float32),
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

    y = tfp.math.batch_interp_regular_1d_grid(x, x_min, x_max, y_ref)
    y_ = self.evaluate(y)
    self.assertAllClose(y_, y_expected, atol=0, rtol=1e-6)
    if not tf.executing_eagerly():
      dy_dx_ = self.evaluate(tf.gradients(ys=y, xs=x)[0])
      self.assertAllClose([2., 2., 0., 2., 2.], dy_dx_)


@test_util.test_all_tf_execution_regimes
class BatchInterpRegularNDGridTest(test_util.TestCase):

  def test_2d_scalar_valued_no_leading_dims(self):
    y_ref = [[0., 1.], [2., 3.]]
    y = tfp.math.batch_interp_regular_nd_grid(
        # Interpolate at one single point.
        x=[[0., 0.]],
        x_ref_min=[0., 0.],
        x_ref_max=[1., 1.],
        y_ref=y_ref,
        axis=-2)
    # Test x at the upper left grid point.
    self.assertAllClose([0.0], self.evaluate(y))

    # Test x at all grid points
    y = tfp.math.batch_interp_regular_nd_grid(
        x=[[0., 0.], [0., 1.], [1., 0.], [1., 1.]],
        x_ref_min=[0., 0.],
        x_ref_max=[1., 1.],
        y_ref=y_ref,
        axis=-2)
    self.assertAllClose([0.0, 1.0, 2.0, 3.0], self.evaluate(y))

    # Test x at intermediate grid points, outside points, and NaN.
    y = tfp.math.batch_interp_regular_nd_grid(
        x=[
            [0.0, 0.5],
            [0.5, 0.0],
            [1., np.nan],
            [np.nan, 1.],
            # [0, 1] is the same as [0, 2], since we snap values outside the
            # grid back on to the grid.
            [0.0, 1.0],
            [0.0, 2.0]
        ],
        x_ref_min=[0., 0.],
        x_ref_max=[1., 1.],
        y_ref=y_ref,
        axis=-2)
    self.assertAllClose([0.5, 1.0, np.nan, np.nan, 1., 1.], self.evaluate(y))

  def test_2d_scalar_valued_no_leading_dims_fill_value_provided(self):
    y_ref = [[0., 1.], [2., 3.]]

    # Test x at intermediate grid points, outside points, and NaN.
    y = tfp.math.batch_interp_regular_nd_grid(
        x=[
            [-1.0, 0.5],  # Outside the grid.
            [10.0, 0.5],  # Outside the grid.
            [0.5, -0.5],  # Outside the grid.
            [0.5, 3.5],  # Outside the grid.
            [0.0, 0.5],
            [0.5, 0.0],
            [1., np.nan],
            [np.nan, 1.],
            [0.0, 1.0],
        ],
        x_ref_min=[0., 0.],
        x_ref_max=[1., 1.],
        y_ref=y_ref,
        axis=-2,
        fill_value=-42.)
    self.assertAllClose([-42, -42, -42, -42, 0.5, 1.0, np.nan, np.nan, 1.],
                        self.evaluate(y))

  def test_1d_scalar_valued_function(self):
    x_ref_min = np.array([-2.], dtype=np.float32)
    x_ref_max = np.array([1.3], dtype=np.float32)
    ny = [100]

    # Build y_ref.
    x0s = tf.linspace(x_ref_min[0], x_ref_max[0], ny[0])

    def func(x0):
      return tf.sin(x0)**2

    # Shape ny
    y_ref = self.evaluate(func(x0s))

    # Shape [10, 1]
    x = tf.random.uniform(
        shape=(10, 1), minval=x_ref_min[0], maxval=x_ref_max[0],
        seed=test_util.test_seed())

    x = self.evaluate(x)

    expected_y = func(x[:, 0])
    actual_y = tfp.math.batch_interp_regular_nd_grid(
        x=x, x_ref_min=x_ref_min, x_ref_max=x_ref_max, y_ref=y_ref, axis=-1)

    self.assertAllClose(
        self.evaluate(expected_y), self.evaluate(actual_y), atol=0.02)

  def test_1d_scalar_valued_function_with_batch_dims(self):
    # Shape [2, 1], [2]is the batch shape.
    x_ref_min = np.array([[-2.], [-3.]], dtype=np.float32)

    # Shape [1] -- will have to be broadcast!
    x_ref_max = np.array([1.3], dtype=np.float32)
    ny = [100]

    # Build y_ref

    # x0s.shape = [2, ny]
    x0s_batch_0 = tf.linspace(x_ref_min[0][0], x_ref_max[0], ny[0])
    x0s_batch_1 = tf.linspace(x_ref_min[1][0], x_ref_max[0], ny[0])
    x0s = tf.stack([x0s_batch_0, x0s_batch_1], axis=0)

    def func(x0):
      return tf.sin(x0)**2

    # Shape [2, ny]
    y_ref = self.evaluate(func(x0s))

    # x's batch shape is [3, 2], which is the largest of the inputs, so it will
    # determine the output batch shape.
    x = tf.random.uniform(
        shape=(3, 2, 10, 1), minval=x_ref_min[0], maxval=x_ref_max[0],
        seed=test_util.test_seed())

    x = self.evaluate(x)

    expected_y = func(x[..., 0])
    actual_y = tfp.math.batch_interp_regular_nd_grid(
        x=x, x_ref_min=x_ref_min, x_ref_max=x_ref_max, y_ref=y_ref, axis=-1)

    self.assertAllClose(
        self.evaluate(expected_y), self.evaluate(actual_y), atol=0.02)

  def test_2d_scalar_valued_function(self):
    x_ref_min = np.array([0., 1.], dtype=np.float32)
    x_ref_max = np.array([1.3, 2.], dtype=np.float32)
    ny = [100, 110]

    # Build y_ref.
    x0s, x1s = tf.meshgrid(
        tf.linspace(x_ref_min[0], x_ref_max[0], ny[0]),
        tf.linspace(x_ref_min[1], x_ref_max[1], ny[1]),
        indexing='ij')

    def func(x0, x1):
      return tf.sin(x0) * tf.cos(x1)

    # Shape ny
    y_ref = self.evaluate(func(x0s, x1s))

    # Shape [10, 2]
    seed = test_util.test_seed_stream()
    x = tf.stack([
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[0], maxval=x_ref_max[0], seed=seed()),
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[1], maxval=x_ref_max[1], seed=seed()),
    ],
                 axis=-1)

    x = self.evaluate(x)

    expected_y = func(x[:, 0], x[:, 1])
    actual_y = tfp.math.batch_interp_regular_nd_grid(
        x=x, x_ref_min=x_ref_min, x_ref_max=x_ref_max, y_ref=y_ref, axis=-2)

    self.assertAllClose(
        self.evaluate(expected_y), self.evaluate(actual_y), atol=0.02)

  def test_2d_vector_valued_function(self):
    x_ref_min = np.array([1., 0.], dtype=np.float32)
    x_ref_max = np.array([2.3, 1.], dtype=np.float32)
    ny = [200, 210]

    # Build y_ref.
    x0s, x1s = tf.meshgrid(
        tf.linspace(x_ref_min[0], x_ref_max[0], ny[0]),
        tf.linspace(x_ref_min[1], x_ref_max[1], ny[1]),
        indexing='ij')

    def func(x0, x1):
      # Shape [..., 2] output.
      return tf.stack([tf.sin(x0 * x1), tf.cos(x0 * x1)], axis=-1)

    # Shape ny + [2]
    y_ref = self.evaluate(func(x0s, x1s))

    # Shape [10, 2]
    seed = test_util.test_seed_stream()
    x = tf.stack([
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[0], maxval=x_ref_max[0], seed=seed()),
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[1], maxval=x_ref_max[1], seed=seed()),
    ],
                 axis=-1)

    x = self.evaluate(x)

    expected_y = func(x[:, 0], x[:, 1])
    actual_y = tfp.math.batch_interp_regular_nd_grid(
        x=x, x_ref_min=x_ref_min, x_ref_max=x_ref_max, y_ref=y_ref, axis=-3)

    self.assertAllClose(
        self.evaluate(expected_y), self.evaluate(actual_y), atol=0.02)

  def test_2d_vector_valued_function_with_batch_dims(self):
    # No batch dims, will broadcast.
    x_ref_min = np.array([0., 0.], dtype=np.float32)

    # No batch dims, will broadcast.
    x_ref_max = np.array([1., 1.], dtype=np.float32)
    ny = [200, 210]

    # Build y_ref.

    # First step is to build up two batches of x0 and x1.
    x0s, x1s = tf.meshgrid(
        tf.linspace(x_ref_min[0], x_ref_max[0], ny[0]),
        tf.linspace(x_ref_min[1], x_ref_max[1], ny[1]),
        indexing='ij')
    x0s = tf.stack([x0s, x0s], axis=0)
    x1s = tf.stack([x1s, x1s], axis=0)

    def func(batch_of_x0, batch_of_x1):
      """Function that does something different for batch 0 and batch 1."""
      # batch_0_result.shape = [..., 2].
      x0, x1 = batch_of_x0[0, ...], batch_of_x1[0, ...]
      batch_0_result = tf.stack([tf.sin(x0 * x1), tf.cos(x0 * x1)], axis=-1)

      x0, x1 = batch_of_x0[1, ...], batch_of_x1[1, ...]
      batch_1_result = tf.stack([tf.sin(2 * x0), tf.cos(2 * x1)], axis=-1)

      return tf.stack([batch_0_result, batch_1_result], axis=0)

    # Shape [2] + ny + [2]
    y_ref = self.evaluate(func(x0s, x1s))

    # Shape [2, 10, 2].  The batch shape is [2], the [10] is the number of
    # interpolants per batch.
    x = tf.random.uniform(shape=[2, 10, 2], seed=test_util.test_seed())

    x = self.evaluate(x)

    expected_y = func(x[..., 0], x[..., 1])
    actual_y = tfp.math.batch_interp_regular_nd_grid(
        x=x, x_ref_min=x_ref_min, x_ref_max=x_ref_max, y_ref=y_ref, axis=-3)

    self.assertAllClose(
        self.evaluate(expected_y), self.evaluate(actual_y), atol=0.02)

  def test_3d_vector_valued_function_and_fill_value(self):
    x_ref_min = np.array([1.0, 0.0, -1.2], dtype=np.float32)
    x_ref_max = np.array([2.3, 3.0, 1.0], dtype=np.float32)
    ny = [200, 210, 211]

    # Build y_ref.
    x0s, x1s, x2s = tf.meshgrid(
        tf.linspace(x_ref_min[0], x_ref_max[0], ny[0]),
        tf.linspace(x_ref_min[1], x_ref_max[1], ny[1]),
        tf.linspace(x_ref_min[2], x_ref_max[2], ny[2]),
        indexing='ij')

    def func(x0, x1, x2):
      # Shape [..., 2] output.
      return tf.stack([tf.sin(x0 * x1 * x2), tf.cos(x0 * x1 * x2)], axis=-1)

    # Shape ny + [2]
    y_ref = self.evaluate(func(x0s, x1s, x2s))

    seed = test_util.test_seed_stream()
    # Shape [10, 3]
    x = tf.stack([
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[0], maxval=x_ref_max[0], seed=seed()),
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[1], maxval=x_ref_max[1], seed=seed()),
        tf.random.uniform(
            shape=(10,), minval=x_ref_min[2], maxval=x_ref_max[2], seed=seed()),
    ],
                 axis=-1)

    x = onp.array(self.evaluate(x))
    x[0, 0] = -3  # Outside the grid, so `fill_value` will be imputed.

    expected_y = onp.array(self.evaluate(func(x[:, 0], x[:, 1], x[:, 2])))
    fill_value = -42
    expected_y[0, :] = fill_value

    actual_y = tfp.math.batch_interp_regular_nd_grid(
        x=x,
        x_ref_min=x_ref_min,
        x_ref_max=x_ref_max,
        y_ref=y_ref,
        axis=-4,
        fill_value=fill_value)

    self.assertAllClose(expected_y, self.evaluate(actual_y), atol=0.02)

  def test_axis_set_too_large_raises(self):
    with self.assertRaisesRegexp(ValueError, 'Since dims'):
      tfp.math.batch_interp_regular_nd_grid(
          x=[[1.]], x_ref_min=[0.], x_ref_max=[1.], y_ref=[0., 1.], axis=3)

  @test_util.numpy_disable_gradient_test
  @test_util.jax_disable_test_missing_functionality(
      'Bug in gradient of gather/prod: '
      'https://github.com/google/jax/issues/1888')
  def test_gradients_nonzero_at_reference_points(self):
    x_ref_min = [0.]
    x_ref_max = [1.]

    # Build y_ref.
    x0s = [0., 0.5, 1.]

    def y_func(x0):
      return tf.convert_to_tensor(value=x0) * 2.

    # Shape [3]
    y_ref = self.evaluate(y_func(x0s))

    x = [
        # Outside the grid
        # Grad will be 0, since outside the grid is in a "flat" region.
        [-1.],
        # 0 is at the left reference pt.
        # Grad will be 2.
        [0.],
        # 0.25 is in between reference pts.
        # Grad will be 2.
        [0.25],
        # Middle reference pt.
        # Grad will be 2.
        [0.5],
        # Right reference pt.
        # Grad will be 2.
        [1.],
        # Outside the grid
        # Grad will be 0, since outside the grid is in a "flat" region.
        [2.],
    ]
    x = tf.convert_to_tensor(value=x)

    def func(xx):
      return tfp.math.batch_interp_regular_nd_grid(
          x=xx, x_ref_min=x_ref_min, x_ref_max=x_ref_max, y_ref=y_ref, axis=-1)

    _, dy_dx_ = self.evaluate(tfp.math.value_and_gradient(func, x))
    self.assertAllEqual([0., 2., 2., 2., 2., 0.], dy_dx_[..., 0])

  def test_float64(self):
    y_ref = tf.convert_to_tensor([[0., 1.], [2., 3.]], dtype=tf.float64)
    y = tfp.math.batch_interp_regular_nd_grid(
        # Interpolate at one single point.
        x=tf.convert_to_tensor([[0., 0.]], dtype=tf.float64),
        x_ref_min=tf.convert_to_tensor([0., 0.], dtype=tf.float64),
        x_ref_max=tf.convert_to_tensor([1., 1.], dtype=tf.float64),
        y_ref=y_ref,
        axis=-2)
    # Test x at the upper left grid point.
    self.assertEqual(y.dtype, tf.float64)
    self.assertAllClose([0.0], self.evaluate(y))

if __name__ == '__main__':
  tf.test.main()
