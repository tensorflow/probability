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
"""The Pareto distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor


class Pareto(distribution.Distribution):
  """Pareto distribution.

  The Pareto distribution is parameterized by a `scale` and a
  `concentration` parameter.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; alpha, scale, x >= scale) = alpha * scale ** alpha / x ** (alpha + 1)
  ```

  where `concentration = alpha`.

  Note that `scale` acts as a scaling parameter, since
  `Pareto(c, scale).pdf(x) == Pareto(c, 1.).pdf(x / scale)`.

  The support of the distribution is defined on `[scale, infinity)`.
  """

  def __init__(self,
               concentration,
               scale=1.,
               validate_args=False,
               allow_nan_stats=True,
               name='Pareto'):
    """Construct Pareto distribution with `concentration` and `scale`.

    Args:
      concentration: Floating point tensor. Must contain only positive values.
      scale: Floating point tensor, equivalent to `mode`. `scale` also
        restricts the domain of this distribution to be in `[scale, inf)`.
        Must contain only positive values. Default value: `1`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs. Default value: `False` (i.e. do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'Pareto'.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([concentration, scale],
                                      dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      super(Pareto, self).__init__(
          dtype=self._concentration.dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(concentration=0, scale=0)

  @property
  def scale(self):
    """Scale parameter and also the lower bound of the support."""
    return self._scale

  @property
  def concentration(self):
    """Concentration parameter for this distribution."""
    return self._concentration

  def _batch_shape_tensor(self, concentration=None, scale=None):
    return ps.broadcast_shape(
        ps.shape(
            self.concentration if concentration is None else concentration),
        ps.shape(self.scale if scale is None else scale))

  def _batch_shape(self):
    return tf.broadcast_static_shape(self.concentration.shape, self.scale.shape)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    shape = ps.concat(
        [[n],
         self._batch_shape_tensor(concentration=concentration, scale=scale)],
        axis=0)
    sampled = samplers.uniform(shape, maxval=1., seed=seed, dtype=self.dtype)
    log_sample = tf.math.log(scale) - tf.math.log1p(-sampled) / concentration
    return tf.exp(log_sample)

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    with tf.control_dependencies([
        assert_util.assert_greater_equal(
            x, scale, message='`x` is not in the support of the distribution.')
    ] if self.validate_args else []):

      def log_prob_on_support(z):
        return (tf.math.log(concentration) +
                concentration * tf.math.log(scale) -
                (concentration + 1.) * tf.math.log(z))

      return self._extend_support(
          x, scale, log_prob_on_support, alt=-np.inf)

  def _prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    scale = tf.convert_to_tensor(self.scale)
    with tf.control_dependencies([
        assert_util.assert_greater_equal(
            x, scale, message='`x` is not in the support of the distribution.')
    ] if self.validate_args else []):

      def prob_on_support(z):
        return concentration * (scale**concentration) / (z**(concentration + 1))

      return self._extend_support(x, scale, prob_on_support, alt=0.)

  def _log_cdf(self, x):
    scale = tf.convert_to_tensor(self.scale)
    return self._extend_support(
        x, scale,
        lambda x: tf.math.log1p(-(scale / x)**self.concentration),
        alt=-np.inf)

  def _cdf(self, x):
    scale = tf.convert_to_tensor(self.scale)
    return self._extend_support(
        x, scale,
        lambda x: -tf.math.expm1(self.concentration * tf.math.log(scale / x)),
        alt=0.)

  def _log_survival_function(self, x):
    scale = tf.convert_to_tensor(self.scale)
    return self._extend_support(
        x, scale,
        lambda x: self.concentration * tf.math.log(scale / x),
        alt=np.inf)

  @distribution_util.AppendDocstring(
      """The mean of Pareto is defined` if `concentration > 1.`, otherwise it
      is `Inf`.""")
  def _mean(self):
    concentration = tf.convert_to_tensor(self.concentration)
    return tf.where(concentration > 1.,
                    concentration * self.scale / (concentration - 1),
                    dtype_util.as_numpy_dtype(self.dtype)(np.inf))

  @distribution_util.AppendDocstring(
      """The variance of Pareto is defined` if `concentration > 2.`, otherwise
      it is `Inf`.""")
  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    valid_variance = (self.scale**2 * concentration /
                      ((concentration - 1.)**2 * (concentration - 2.)))
    return tf.where(concentration > 2.,
                    valid_variance,
                    dtype_util.as_numpy_dtype(self.dtype)(np.inf))

  def _mode(self):
    scale = tf.convert_to_tensor(self.scale)
    return tf.broadcast_to(scale, self._batch_shape_tensor(scale=scale))

  def _extend_support(self, x, scale, f, alt):
    """Returns `f(x)` if x is in the support, and `alt` otherwise.

    Given `f` which is defined on the support of this distribution
    (e.g. x > scale), extend the function definition to the real line
    by defining `f(x) = alt` for `x < scale`.

    Args:
      x: Floating-point Tensor to evaluate `f` at.
      scale: Floating-point Tensor by which to verify `x` validity.
      f: Lambda that takes in a tensor and returns a tensor. This represents the
        function who we want to extend the domain of definition.
      alt: Python or numpy literal representing the value to use for extending
        the domain.

    Returns:
      Tensor representing an extension of `f(x)`.
    """
    if self.validate_args:
      return f(x)
    scale = tf.convert_to_tensor(self.scale) if scale is None else scale
    is_invalid = x < scale
    # We need to do this to ensure gradients are sound.
    y = f(tf.where(is_invalid, scale, x))
    if alt == 0.:
      alt = tf.zeros([], dtype=y.dtype)
    elif alt == 1.:
      alt = tf.ones([], dtype=y.dtype)
    else:
      alt = dtype_util.as_numpy_dtype(self.dtype)(alt)
    return tf.where(is_invalid, alt, y)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(
          assert_util.assert_positive(
              self.concentration, message='`concentration` must be positive.'))
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(
          assert_util.assert_positive(
              self.scale, message='`scale` must be positive.'))
    return assertions

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    deferred_scale = DeferredTensor(self.scale, lambda x: x)
    return chain_bijector.Chain([
        shift_bijector.Shift(
            shift=deferred_scale, validate_args=self.validate_args),
        softplus_bijector.Softplus(validate_args=self.validate_args)
    ], validate_args=self.validate_args)


@kullback_leibler.RegisterKL(Pareto, Pareto)
def _kl_pareto_pareto(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Pareto.

  Args:
    a: instance of a Pareto distribution object.
    b: instance of a Pareto distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_pareto_pareto'.

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_pareto_pareto'):
    # Consistent with
    # http://www.mast.queensu.ca/~communications/Papers/gil-msc11.pdf, page 55
    # Terminology is different from source to source for Pareto distributions.
    # The 'concentration' parameter corresponds to 'a' in that source, and the
    # 'scale' parameter corresponds to 'm'.
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)
    a_concentration = tf.convert_to_tensor(a.concentration)
    b_concentration = tf.convert_to_tensor(b.concentration)
    return tf.where(
        a_scale >= b_scale,
        (b_concentration * (tf.math.log(a_scale) - tf.math.log(b_scale)) +
         tf.math.log(a_concentration) - tf.math.log(b_concentration) +
         b_concentration / a_concentration - 1.),
        dtype_util.as_numpy_dtype(a.dtype)(np.inf))
