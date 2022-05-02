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
"""Implements bijectors based on heavy-tail Lambert W transformations.

The heavy-tail Lambert W transformation can be used to add heavy-tails to a
(location-scale) distribution.  It is based on the bijective transformation of
a (location-scale) input variable X to a heavy-tailed output variable Y:

  Y = (U * exp (delta/2 * U^2)) * scale + shift,

where U = (X - shift) / scale and delta >= 0 controls the degree of heavy-tails.
Clearly, if delta == 0, Y = X.

More interestingly, this transformation is bijective (for delta >= 0) and thus
can be inverted, which leads to a Gaussianization transformation of data y:

  X = loc + scale * W_delta((Y - shift) / scale),

where

  W_delta(z) = sign(z) * sqrt(W(0.5 * delta * z^2)),

and W(z) is the Lambert W function.

It is thus a generalization of the standard scaling function,
u = (x - mu) / sigma, which adds a step to also remove heavy-tails with a
bijective transformation.

Lambert W bijectors are the basis of Lambert W x F distributions, which are a
generalization of any X ~ F random variable into its heavy-tailed version
Y ~ Lambert W x F. See `LambertWDistribution` class for details.


### References:
[1]: Goerg, G.M., 2011. Lambert W random variables - a new family of generalized
skewed distributions with applications to risk estimation. The Annals of Applied
Statistics, 5(3), pp.2197-2230.
[2]: Goerg, G.M., 2015. The Lambert way to Gaussianize heavy-tailed data with
the inverse of Tukey's h transformation as a special case. The Scientific World
Journal.
"""
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import scale as tfb_scale
from tensorflow_probability.python.bijectors import shift as tfb_shift
from tensorflow_probability.python.bijectors import softplus as tfb_softplus
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    "LambertWTail",
]


def _xexp_delta_squared(u, delta):
  """Applies `u * exp(u^2 * delta / 2)` transformation to the input `u`.

  Args:
    u: Input of the transformation.
    delta: Parameter delta of the transformation. For `delta=0` it reduces to
      the identity function.

  Returns:
    The transformed tensor with same shape and same dtype as `u`.
  """
  delta = tf.convert_to_tensor(delta, dtype=u.dtype)
  u = tf.broadcast_to(u, ps.broadcast_shape(ps.shape(u), ps.shape(delta)))
  return u * tf.math.exp(0.5 * delta * u**2)


def _w_delta_squared(z, delta):
  """Applies W_delta transformation to the input.

  For a given z, `W_delta(z) = sign(z) * (W(delta * z^2)/delta)^0.5`. This
  transformation is defined in Equation (9) of [1].

  Args:
    z: Input of the transformation.
    delta: Parameter delta of the transformation.

  Returns:
    The transformed Tensor with same shape and same dtype as `z`.
  """
  delta = tf.convert_to_tensor(delta, dtype=z.dtype)
  z = tf.broadcast_to(z,
                      ps.broadcast_shape(ps.shape(z), ps.shape(delta)))
  wd = tf.sign(z) * tf.sqrt(tfp_math.lambertw(delta * z**2) / delta)
  return tf.where(tf.equal(delta, 0.0), z, wd)


# Private class that implements the heavy tail transformation.
class _HeavyTailOnly(bijector.AutoCompositeTensorBijector):
  """Heavy tail transformation for Lambert W x F distributions.

  This bijector defines the transformation z = u * exp(0.5 * delta * u**2)
  and its inverse, where it is assumed that `u` is already an appropriately
  shifted & scaled input. It is the basis of the location-scale heavy-tail
  Lambert W x F distributions / transformations.
  The effect of this transformation is that it adds heavy tails to input.

  Attributes:
    tailweight: Tail parameter `delta` of the transformation(s).
  """

  def __init__(self,
               tailweight,
               validate_args=False,
               name="HeavyTailOnly"):
    """Construct heavy-tail Lambert W bijector.

    Args:
      tailweight: Floating point tensor; specifies the excess tail behaviors of
        the distribution(s). Must contain only non-negative values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([tailweight], tf.float32)
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, name="tailweight", dtype=dtype)

      super(_HeavyTailOnly, self).__init__(validate_args=validate_args,
                                           forward_min_event_ndims=0,
                                           parameters=parameters,
                                           name=name)

  def _is_increasing(self):
    """Heavy-tail Lambert W x F transformations are increasing."""
    return True

  def _forward(self, x):
    """Turns input vector x into its heavy-tail version using Lambert W."""
    return _xexp_delta_squared(x, delta=self._tailweight)

  def _inverse(self, y):
    """Reverses the _forward transformation."""
    return _w_delta_squared(y, delta=self._tailweight)

  def _inverse_log_det_jacobian(self, y):
    """Returns log of the Jacobian determinant of the inverse function."""
    # Jacobian is log(W_delta(y, delta) / y) - log(1 + W(delta * y^2)).
    # Note that W_delta(y, delta) / y >= 0, since they share the same sign.
    # For numerical stability use log(abs()) - log(abs()).
    # See also Eq (11) and (31) of
    # https://www.hindawi.com/journals/tswj/2015/909231/
    log_jacobian_term_nonzero = (
        tf.math.log(tf.abs(_w_delta_squared(y, delta=self._tailweight))) -
        tf.math.log(tf.abs(y)) -
        tf.math.log(1. + tfp_math.lambertw(self._tailweight * y**2)))
    # If y = 0 the expression becomes log(0/0) - log(1 + 0), and the first term
    # equals log(1) = 0.  Hence, for y = 0 the whole expression equals 0.
    return tf.where(tf.equal(y, 0.0),
                    tf.zeros_like(y),
                    log_jacobian_term_nonzero)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: tfb_softplus.Softplus(low=dtype_util.eps(dtype)))))


class LambertWTail(chain.Chain):
  """LambertWTail transformation for heavy-tail Lambert W x F random variables.

  A random variable Y has a Lambert W x F distribution if W_tau(Y) = X has
  distribution F, where tau = (shift, scale, tail) parameterizes the inverse
  transformation.

  This bijector defines the transformation underlying Lambert W x F
  distributions that transform an input random variable to an output
  random variable with heavier tails. It is defined as

    Y = (U * exp(0.5 * tail * U^2)) * scale + shift,  tail >= 0

  where U = (X - shift) / scale is a shifted/scaled input random variable, and
  tail >= 0 is the tail parameter.

  Attributes:
    shift: shift to center (uncenter) the input data.
    scale: scale to normalize (de-normalize) the input data.
    tailweight: Tail parameter `delta` of heavy-tail transformation; must be
      >= 0.
  """

  def __init__(self,
               shift,
               scale,
               tailweight,
               validate_args=False,
               name="lambertw_tail"):
    """Construct a location scale heavy-tail Lambert W bijector.

    The parameters `shift`, `scale`, and `tail` must be shaped in a way that
    supports broadcasting (e.g. `shift + scale + tail` is a valid operation).

    Args:
      shift: Floating point tensor; the shift for centering (uncentering) the
        input (output) random variable(s).
      scale: Floating point tensor; the scaling (unscaling) of the input
        (output) random variable(s). Must contain only positive values.
      tailweight: Floating point tensor; the tail behaviors of the output random
        variable(s).  Must contain only non-negative values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `shift` and `scale` and `tail` have different `dtype`.
    """
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([tailweight, shift, scale], tf.float32)
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, name="tailweight", dtype=dtype)
      self._shift = tensor_util.convert_nonref_to_tensor(
          shift, name="shift", dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name="scale", dtype=dtype)
      dtype_util.assert_same_float_dtype((self._tailweight, self._shift,
                                          self._scale))

      self._shift_and_scale = chain.Chain([tfb_shift.Shift(self._shift),
                                           tfb_scale.Scale(self._scale)])
      # 'bijectors' argument in tfb.Chain super class are executed in reverse(!)
      # order.  Hence the ordering in the list must be (3,2,1), not (1,2,3).
      super(LambertWTail, self).__init__(
          bijectors=[self._shift_and_scale,
                     _HeavyTailOnly(tailweight=self._tailweight),
                     invert.Invert(self._shift_and_scale)],
          validate_args=validate_args)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        shift=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: tfb_softplus.Softplus(low=dtype_util.eps(dtype)))),
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: tfb_softplus.Softplus(low=dtype_util.eps(dtype)))))
