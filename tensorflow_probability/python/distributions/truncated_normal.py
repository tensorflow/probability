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
"""The Truncated Normal distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import special_math


__all__ = [
    "TruncatedNormal",
]


class TruncatedNormal(tf.distributions.Distribution):
  """The Truncated Normal distribution.

  #### Mathematical details

  The truncated normal is a normal distribution bounded between `low`
  and `high` (the pdf is 0 outside these bounds and renormalized).

  Samples from this distribution are differentiable with respect to `loc`,
  `scale` as well as the bounds, `low` and `high`, i.e., this
  implementation is fully reparameterizeable.

  For more details, see [here](
  https://en.wikipedia.org/wiki/Truncated_normal_distribution).

  ### Mathematical Details

  The probability density function (pdf) of this distribution is:
  ```none
    pdf(x; loc, scale, low, high) =
        { (2 pi)**(-0.5) exp(-0.5 y**2) / (scale * z) for low <= x <= high
        { 0                                    otherwise
    y = (x - loc)/scale
    z = NormalCDF((high - loc) / scale) - NormalCDF((lower - loc) / scale)
  ```

  where:

  * `NormalCDF` is the cumulative density function of the Normal distribution
    with 0 mean and unit variance.

  This is a scalar distribution so the event shape is always scalar and the
  dimensions of the parameters defined the batch_shape.

  #### Examples
  ```python

  tfd = tfp.distributions
  # Define a batch of two scalar TruncatedNormals which modes at 0. and 1.0
  dist = tfd.TruncatedNormal(loc=[0., 1.], scale=1.0,
                             low=[-1., 0.],
                             high=[1., 1.])

  # Evaluate the pdf of the distributions at 0.5 and 0.8 respectively returning
  # a 2-vector tensor.
  dist.prob([0.5, 0.8])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```
  """

  def __init__(self,
               loc,
               scale,
               low,
               high,
               validate_args=False,
               allow_nan_stats=True,
               name="TruncatedNormal"):
    """Construct TruncatedNormal.

    All parameters of the distribution will be broadcast to the same shape,
    so the resulting distribution will have a batch_shape of the broadcast
    shape of all parameters.

    Args:
      loc: Floating point tensor; the mean of the normal distribution(s) (
        note that the mean of the resulting distribution will be different
        since it is modified by the bounds).
      scale: Floating point tensor; the std deviation of the normal
        distribution(s).
      low: `float` `Tensor` representing lower bound of the distribution's
        support. Must be such that `low < high`.
      high: `float` `Tensor` representing upper bound of the distribution's
        support. Must be such that `low < high`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked at run-time.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name, values=[scale]) as name:
      loc = tf.convert_to_tensor(loc, name="loc")
      dtype = loc.dtype
      scale = tf.convert_to_tensor(scale, name="scale", dtype=dtype)
      low = tf.convert_to_tensor(low, name="low", dtype=dtype)
      high = tf.convert_to_tensor(high, name="high", dtype=dtype)
      tf.assert_same_float_dtype([loc, scale, low, high])

      self._broadcast_batch_shape = distribution_util.get_broadcast_shape(
          loc, scale, low, high)

      # Broadcast all parameters to the same shape
      broadcast_ones = tf.ones(shape=self._broadcast_batch_shape,
                               dtype=scale.dtype)
      self._scale = scale * broadcast_ones
      self._loc = loc * broadcast_ones
      self._low = low * broadcast_ones
      self._high = high * broadcast_ones

      with tf.control_dependencies([self._validate()] if validate_args else []):
        self._loc = tf.identity(self._loc)

    super(TruncatedNormal, self).__init__(
        dtype=dtype,
        # This distribution is fully reparameterized. loc, scale have straight
        # through gradients. The gradients for the bounds are implemented using
        # custom derived expressions based on implicit gradients.
        # For the special case of lower bound zero and a positive upper bound
        # an equivalent expression can also be found in Sec 9.1.1.
        # of https://arxiv.org/pdf/1806.01851.pdf. The implementation here
        # handles arbitrary bounds.
        reparameterization_type=tf.distributions.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[loc, scale, low, high],
        name=name)

  def _validate(self):
    vops = [tf.assert_positive(self._scale),
            tf.assert_positive(self._high - self._low),
            tf.verify_tensor_all_finite(self._high,
                                        "Upper bound not finite"),
            tf.verify_tensor_all_finite(self._low,
                                        "Lower bound not finite"),
            tf.verify_tensor_all_finite(self._loc,
                                        "Loc not finite"),
            tf.verify_tensor_all_finite(self._scale,
                                        "Scale not finite"),
           ]
    return tf.group(*vops, name="ValidationOps")

  @property
  def _standardized_low(self):
    return (self._low - self._loc) / self._scale

  @property
  def _standardized_high(self):
    return (self._high - self._loc) / self._scale

  @property
  def _normalizer(self):
    return (special_math.ndtr(self._standardized_high) -
            special_math.ndtr(self._standardized_low))

  def _normal_pdf(self, x):
    return 1. / np.sqrt(2 * np.pi) * tf.exp(-0.5 * tf.square(x))

  @staticmethod
  def _param_shapes(sample_shape):
    # All parameters are of the same shape
    shape = tf.convert_to_tensor(sample_shape, dtype=tf.int32)
    return {"loc": shape,
            "scale": shape,
            "high": shape,
            "low": shape}

  @property
  def loc(self):
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  def _batch_shape_tensor(self):
    # All the parameters are broadcast the same shape during construction.
    return tf.shape(self.loc)

  def _batch_shape(self):
    # All the parameters are broadcast the same shape during construction.
    return self.loc.shape

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    sample_and_batch_shape = tf.concat([[n], self.batch_shape_tensor()], 0)
    flat_batch_and_sample_shape = tf.stack([
        tf.reduce_prod(self.batch_shape_tensor()), n])

    # In order to be reparameterizable we sample on the truncated_normal of
    # unit variance and mean and scale (but with the standardized
    # truncation bounds).

    @tf.custom_gradient
    def _std_samples_with_gradients(lower, upper):
      """Standard truncated Normal with gradient support for low, high."""
      # Note: Unlike the convention in tf_probability,
      # parameterized_truncated_normal returns a tensor with the final dimension
      # being the sample dimension.
      std_samples = random_ops.parameterized_truncated_normal(
          shape=flat_batch_and_sample_shape,
          means=0.0,
          stddevs=1.0,
          minvals=lower,
          maxvals=upper,
          dtype=self.dtype,
          seed=seed)

      def grad(dy):
        """Computes a derivative for the min and max parameters.

        This function implements the derivative wrt the truncation bounds, which
        get blocked by the sampler. We use a custom expression for numerical
        stability instead of automatic differentiation on CDF for implicit
        gradients.

        Args:
          dy: output gradients

        Returns:
           The standard normal samples and the gradients wrt the upper
           bound and lower bound.
        """
        # std_samples has an extra dimension (the sample dimension), expand
        # lower and upper so they broadcast along this dimension.
        # See note above regarding parameterized_truncated_normal, the sample
        # dimension is the final dimension.
        lower_broadcast = lower[..., tf.newaxis]
        upper_broadcast = upper[..., tf.newaxis]

        cdf_samples = ((special_math.ndtr(std_samples) -
                        special_math.ndtr(lower_broadcast)) /
                       (special_math.ndtr(upper_broadcast)
                        - special_math.ndtr(lower_broadcast)))

        # tiny, eps are tolerance parameters to ensure we stay away from giving
        # a zero arg to the log CDF expression.

        tiny = np.finfo(self.dtype.as_numpy_dtype).tiny
        eps = np.finfo(self.dtype.as_numpy_dtype).eps
        cdf_samples = tf.clip_by_value(cdf_samples, tiny, 1 - eps)

        du = tf.exp(0.5 * (std_samples**2 - upper_broadcast**2)
                    + tf.log(cdf_samples))
        dl = tf.exp(0.5 * (std_samples**2 - lower_broadcast**2)
                    + tf.log1p(-cdf_samples))

        # Reduce the gradient across the samples
        grad_u = tf.reduce_sum(dy * du, axis=-1)
        grad_l = tf.reduce_sum(dy * dl, axis=-1)
        return [grad_l, grad_u]

      return std_samples, grad

    std_samples = _std_samples_with_gradients(
        tf.reshape(self._standardized_low, [-1]),
        tf.reshape(self._standardized_high, [-1]))

    # The returned shape is [flat_batch x n]
    std_samples = tf.transpose(std_samples, [1, 0])

    std_samples = tf.reshape(std_samples, sample_and_batch_shape)
    samples = (std_samples * tf.expand_dims(self._scale, axis=0) +
               tf.expand_dims(self._loc, axis=0))

    return samples

  def _log_prob(self, x):
    log_prob = -(0.5 * ((x - self.loc) / self.scale) ** 2 +
                 0.5 * np.log(2. * np.pi)
                 + tf.log(self.scale * self._normalizer))
    # p(x) is 0 outside the bounds.
    neg_inf = tf.log(tf.zeros_like(log_prob))
    bounded_log_prob = tf.where(
        tf.logical_or(x > self._high, x < self._low),
        neg_inf,
        log_prob)
    return bounded_log_prob

  def _cdf(self, x):
    cdf_in_support = ((special_math.ndtr((x - self.loc) / self.scale)
                       -  special_math.ndtr(self._standardized_low))
                      / self._normalizer)
    return tf.clip_by_value(cdf_in_support, 0., 1.)

  def _entropy(self):
    return (tf.log(np.sqrt(2. * np.pi * np.e) *
                   self.scale * self._normalizer) +
            (self._standardized_low * self._normal_pdf(
                self._standardized_low) -
             self._standardized_high * self._normal_pdf(
                 self._standardized_high)) / (2. * self._normalizer))

  def _mean(self):
    return (self.loc +
            self._scale * ((self._normal_pdf(self._standardized_low) -
                            self._normal_pdf(self._standardized_high))
                           / self._normalizer))

  def _mode(self):
    # mode = { loc:         for low <= loc <= high
    #          low: for loc < low
    #          high: for loc > high
    #        }
    return tf.clip_by_value(self.loc, self.low, self.high)

  def _variance(self):
    var = (tf.square(self.scale) *
           (1. + (self._standardized_low * self._normal_pdf(
               self._standardized_low) -
                  self._standardized_high * self._normal_pdf(
                      self._standardized_high)) / self._normalizer -
            tf.square((self._normal_pdf(self._standardized_low) -
                       self._normal_pdf(self._standardized_high))
                      / self._normalizer)))
    return var
