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
"""Bijector unit-test utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

# Dependency imports
from absl import logging
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import reshape as reshape_bijector
from tensorflow_probability.python.distributions import uniform as uniform_distribution
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import test_util as tfp_test_util
from tensorflow_probability.python.math.gradient import batch_jacobian


JAX_MODE = False


def assert_finite(array):
  if not np.isfinite(array).all():
    raise AssertionError('array was not all finite. %s' % array[:15])


def assert_strictly_increasing(array):
  np.testing.assert_array_less(0., np.diff(array))


def assert_strictly_decreasing(array):
  np.testing.assert_array_less(np.diff(array), 0.)


def assert_strictly_monotonic(array):
  if array[0] < array[-1]:
    assert_strictly_increasing(array)
  else:
    assert_strictly_decreasing(array)


def assert_scalar_congruency(bijector,
                             lower_x,
                             upper_x,
                             eval_func,
                             n=int(10e3),
                             rtol=0.01):
  """Assert `bijector`'s forward/inverse/inverse_log_det_jacobian are congruent.

  We draw samples `X ~ U(lower_x, upper_x)`, then feed these through the
  `bijector` in order to check that:

  1. the forward is strictly monotonic.
  2. the forward/inverse methods are inverses of each other.
  3. the jacobian is the correct change of measure.

  This can only be used for a Bijector mapping open subsets of the real line
  to themselves.  This is due to the fact that this test compares the `prob`
  before/after transformation with the Lebesgue measure on the line.

  Args:
    bijector:  Instance of Bijector
    lower_x:  Python scalar.
    upper_x:  Python scalar.  Must have `lower_x < upper_x`, and both must be in
      the domain of the `bijector`.  The `bijector` should probably not produce
      huge variation in values in the interval `(lower_x, upper_x)`, or else the
      variance based check of the Jacobian will require small `rtol` or huge
      `n`.
    eval_func: Function to evaluate any intermediate results.
    n:  Number of samples to draw for the checks.
    rtol:  Positive number.  Used for the Jacobian check.

  Raises:
    AssertionError:  If tests fail.
  """
  # Should be monotonic over this interval
  ten_x_pts = np.linspace(lower_x, upper_x, num=10).astype(np.float32)
  if bijector.dtype is not None:
    ten_x_pts = ten_x_pts.astype(dtype_util.as_numpy_dtype(bijector.dtype))
    lower_x = np.cast[dtype_util.as_numpy_dtype(bijector.dtype)](lower_x)
    upper_x = np.cast[dtype_util.as_numpy_dtype(bijector.dtype)](upper_x)
  forward_on_10_pts = bijector.forward(ten_x_pts)

  # Set the lower/upper limits in the range of the bijector.
  lower_y, upper_y = eval_func(
      [bijector.forward(lower_x),
       bijector.forward(upper_x)])
  if upper_y < lower_y:  # If bijector.forward is a decreasing function.
    lower_y, upper_y = upper_y, lower_y

  # Uniform samples from the domain, range.
  seed_stream = tfp_test_util.test_seed_stream(salt='assert_scalar_congruency')
  uniform_x_samps = uniform_distribution.Uniform(
      low=lower_x, high=upper_x).sample(n, seed=seed_stream())
  uniform_y_samps = uniform_distribution.Uniform(
      low=lower_y, high=upper_y).sample(n, seed=seed_stream())

  # These compositions should be the identity.
  inverse_forward_x = bijector.inverse(bijector.forward(uniform_x_samps))
  forward_inverse_y = bijector.forward(bijector.inverse(uniform_y_samps))

  # For a < b, and transformation y = y(x),
  # (b - a) = \int_a^b dx = \int_{y(a)}^{y(b)} |dx/dy| dy
  # "change_measure_dy_dx" below is a Monte Carlo approximation to the right
  # hand side, which should then be close to the left, which is (b - a).
  # We assume event_ndims=0 because we assume scalar -> scalar. The log_det
  # methods will handle whether they expect event_ndims > 0.
  dy_dx = tf.exp(
      bijector.inverse_log_det_jacobian(uniform_y_samps, event_ndims=0))
  # E[|dx/dy|] under Uniform[lower_y, upper_y]
  # = \int_{y(a)}^{y(b)} |dx/dy| dP(u), where dP(u) is the uniform measure
  expectation_of_dy_dx_under_uniform = tf.reduce_mean(dy_dx)
  # dy = dP(u) * (upper_y - lower_y)
  change_measure_dy_dx = (
      (upper_y - lower_y) * expectation_of_dy_dx_under_uniform)

  # We'll also check that dy_dx = 1 / dx_dy.
  dx_dy = tf.exp(
      bijector.forward_log_det_jacobian(
          bijector.inverse(uniform_y_samps), event_ndims=0))

  [
      forward_on_10_pts_v,
      dy_dx_v,
      dx_dy_v,
      change_measure_dy_dx_v,
      uniform_x_samps_v,
      uniform_y_samps_v,
      inverse_forward_x_v,
      forward_inverse_y_v,
  ] = eval_func([
      forward_on_10_pts,
      dy_dx,
      dx_dy,
      change_measure_dy_dx,
      uniform_x_samps,
      uniform_y_samps,
      inverse_forward_x,
      forward_inverse_y,
  ])

  assert_strictly_monotonic(forward_on_10_pts_v)
  # Composition of forward/inverse should be the identity.
  np.testing.assert_allclose(
      inverse_forward_x_v, uniform_x_samps_v, atol=1e-5, rtol=1e-3)
  np.testing.assert_allclose(
      forward_inverse_y_v, uniform_y_samps_v, atol=1e-5, rtol=1e-3)
  # Change of measure should be correct.
  np.testing.assert_allclose(
      desired=upper_x - lower_x, actual=change_measure_dy_dx_v,
      atol=0, rtol=rtol)
  # Inverse Jacobian should be equivalent to the reciprocal of the forward
  # Jacobian.
  np.testing.assert_allclose(
      desired=dy_dx_v, actual=np.reciprocal(dx_dy_v), atol=1e-5, rtol=1e-3)


def assert_bijective_and_finite(bijector,
                                x,
                                y,
                                event_ndims,
                                eval_func,
                                inverse_event_ndims=None,
                                atol=0,
                                rtol=1e-5):
  """Assert that forward/inverse (along with jacobians) are inverses and finite.

  It is recommended to use x and y values that are very very close to the edge
  of the Bijector's domain.

  Args:
    bijector:  A Bijector instance.
    x:  np.array of values in the domain of bijector.forward.
    y:  np.array of values in the domain of bijector.inverse.
    event_ndims: Integer describing the number of event dimensions this bijector
      operates on.
    eval_func: Function to evaluate any intermediate results.
    inverse_event_ndims: Integer describing the number of event dimensions for
      the bijector codomain. If None, then the value of `event_ndims` is used.
    atol:  Absolute tolerance.
    rtol:  Relative tolerance.

  Raises:
    AssertionError:  If tests fail.
  """
  if inverse_event_ndims is None:
    inverse_event_ndims = event_ndims
  # These are the incoming points, but people often create a crazy range of
  # values for which these end up being bad, especially in 16bit.
  assert_finite(x)
  assert_finite(y)

  f_x = bijector.forward(x)
  g_y = bijector.inverse(y)

  [
      x_from_x,
      y_from_y,
      ildj_f_x,
      fldj_x,
      ildj_y,
      fldj_g_y,
      f_x_v,
      g_y_v,
  ] = eval_func([
      bijector.inverse(f_x),
      bijector.forward(g_y),
      bijector.inverse_log_det_jacobian(f_x, event_ndims=inverse_event_ndims),
      bijector.forward_log_det_jacobian(x, event_ndims=event_ndims),
      bijector.inverse_log_det_jacobian(y, event_ndims=inverse_event_ndims),
      bijector.forward_log_det_jacobian(g_y, event_ndims=event_ndims),
      f_x,
      g_y,
  ])

  assert_finite(x_from_x)
  assert_finite(y_from_y)
  assert_finite(ildj_f_x)
  assert_finite(fldj_x)
  assert_finite(ildj_y)
  assert_finite(fldj_g_y)
  assert_finite(f_x_v)
  assert_finite(g_y_v)

  np.testing.assert_allclose(x_from_x, x, atol=atol, rtol=rtol)
  np.testing.assert_allclose(y_from_y, y, atol=atol, rtol=rtol)
  np.testing.assert_allclose(-ildj_f_x, fldj_x, atol=atol, rtol=rtol)
  np.testing.assert_allclose(-ildj_y, fldj_g_y, atol=atol, rtol=rtol)


def get_fldj_theoretical(bijector,
                         x,
                         event_ndims,
                         inverse_event_ndims=None,
                         input_to_unconstrained=None,
                         output_to_unconstrained=None):
  """Numerically approximate the forward log det Jacobian of a bijector.

  We compute the Jacobian of the chain
  output_to_unconst_vec(bijector(inverse(input_to_unconst_vec))) so that
  we're working with a full rank matrix.  We then adjust the resulting Jacobian
  for the unconstraining bijectors.

  Bijectors that constrain / unconstrain their inputs/outputs may not be
  testable with this method, since the composition above may reduce the test
  to something trivial.  However, bijectors that map within constrained spaces
  should be fine.

  Args:
    bijector: the bijector whose Jacobian we wish to approximate
    x: the value for which we want to approximate the Jacobian. Must have rank
      at least `event_ndims`.
    event_ndims: number of dimensions in an event
    inverse_event_ndims: Integer describing the number of event dimensions for
      the bijector codomain. If None, then the value of `event_ndims` is used.
    input_to_unconstrained: bijector that maps the input to the above bijector
      to an unconstrained 1-D vector.  If unspecified, flatten the input into
      a 1-D vector according to its event_ndims.
    output_to_unconstrained: bijector that maps the output of the above bijector
      to an unconstrained 1-D vector.  If unspecified, flatten the input into
      a 1-D vector according to its event_ndims.

  Returns:
    fldj: A gradient-based evaluation of the log det Jacobian of
      `bijector.forward` at `x`.
  """
  if inverse_event_ndims is None:
    inverse_event_ndims = event_ndims
  if input_to_unconstrained is None:
    input_to_unconstrained = reshape_bijector.Reshape(
        event_shape_in=x.shape[tensorshape_util.rank(x.shape) - event_ndims:],
        event_shape_out=[-1])
  if output_to_unconstrained is None:
    f_x_shape = bijector.forward_event_shape(x.shape)
    output_to_unconstrained = reshape_bijector.Reshape(
        event_shape_in=f_x_shape[tensorshape_util.rank(f_x_shape) -
                                 inverse_event_ndims:],
        event_shape_out=[-1])

  x = tf.convert_to_tensor(x)
  x_unconstrained = 1 * input_to_unconstrained.forward(x)
  # Collapse any batch dimensions (including scalar) to a single axis.
  batch_shape = x_unconstrained.shape[:-1]
  x_unconstrained = tf.reshape(
      x_unconstrained, [int(np.prod(batch_shape)), x_unconstrained.shape[-1]])

  def f(x_unconstrained, batch_shape=batch_shape):
    # Unflatten any batch dimensions now under the tape.
    unflattened_x_unconstrained = tf.reshape(
        x_unconstrained,
        tensorshape_util.concatenate(batch_shape, x_unconstrained.shape[-1:]))
    f_x = bijector.forward(input_to_unconstrained.inverse(
        unflattened_x_unconstrained))
    return f_x

  def f_unconstrained(x_unconstrained, batch_shape=batch_shape):
    f_x_unconstrained = output_to_unconstrained.forward(
        f(x_unconstrained, batch_shape=batch_shape))
    # Flatten any batch dimensions to a single axis.
    return tf.reshape(
        f_x_unconstrained,
        [int(np.prod(batch_shape)), f_x_unconstrained.shape[-1]])

  if JAX_MODE:
    f_unconstrained = functools.partial(f_unconstrained, batch_shape=[])
  jacobian = batch_jacobian(f_unconstrained, x_unconstrained)
  jacobian = tf.reshape(
      jacobian, tensorshape_util.concatenate(batch_shape, jacobian.shape[-2:]))
  logging.vlog(1, 'Jacobian: %s', jacobian)

  log_det_jacobian = 0.5 * tf.linalg.slogdet(
      tf.matmul(jacobian, jacobian, adjoint_a=True)).log_abs_determinant

  input_correction = input_to_unconstrained.forward_log_det_jacobian(
      x, event_ndims=event_ndims)
  output_correction = output_to_unconstrained.forward_log_det_jacobian(
      f(x_unconstrained), event_ndims=inverse_event_ndims)
  return (log_det_jacobian + tf.cast(input_correction, log_det_jacobian.dtype) -
          tf.cast(output_correction, log_det_jacobian.dtype))
