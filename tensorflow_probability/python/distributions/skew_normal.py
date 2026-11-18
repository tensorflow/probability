# Copyright 2026 The TensorFlow Probability Authors.
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
"""The Skew-Normal distribution class."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import special as tfp_math_special

__all__ = [
    'SkewNormal',
]

class SkewNormal(distribution.AutoCompositeTensorDistribution):
  """The Skew-Normal distribution.

  The Skew-Normal distribution is a generalization of the Normal distribution
  that allows for non-zero skewness.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale, skewness) = 2 / scale * phi(z) * Phi(skewness * z)
  z = (x - loc) / scale
  ```

  where:
  * `loc` is the location parameter,
  * `scale` is the scale parameter,
  * `skewness` (often denoted as `alpha`) is the skewness parameter,
  * `phi(z)` is the standard normal pdf, and
  * `Phi(z)` is the standard normal cdf.

  The cumulative distribution function (cdf) is,

  ```none
  cdf(x; loc, scale, skewness) = Phi(z) - 2 * T(z, skewness)
  ```

  where `T` is Owen's T function.
  """

  def __init__(self,
               loc,
               scale,
               skewness,
               validate_args=False,
               allow_nan_stats=True,
               name='SkewNormal'):
    """Construct Skew-Normal distribution with parameters `loc`, `scale`, and `skewness`.

    Args:
      loc: Floating point tensor; the location of the distribution(s).
      scale: Floating point tensor; the scale of the distribution(s).
        Must contain only positive values.
      skewness: Floating point tensor; the skewness of the distribution(s).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, scale, skewness], dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, dtype=dtype, name='skewness')
      
      super(SkewNormal, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        skewness=parameter_properties.ParameterProperties())
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  @property
  def skewness(self):
    """Distribution parameter for skewness."""
    return self._skewness

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    
    shape = ps.concat([[n], self._batch_shape_tensor(
        loc=loc, scale=scale, skewness=skewness)], axis=0)
    
    seed1, seed2 = samplers.split_seed(seed, salt='skew_normal')
    
    u0 = samplers.normal(shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed1)
    v = samplers.normal(shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed2)
    
    delta = skewness / tf.math.sqrt(1. + tf.math.square(skewness))
    z = delta * tf.math.abs(u0) + tf.math.sqrt(1. - tf.math.square(delta)) * v
    
    return z * scale + loc

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    z = (x - self.loc) / scale
    
    log_unnormalized = -0.5 * tf.math.square(z)
    log_normalization = tf.constant(
        0.5 * np.log(2. * np.pi), dtype=self.dtype) + tf.math.log(scale)
    log_pdf_normal = log_unnormalized - log_normalization
    
    log_cdf_skew = special_math.log_ndtr(skewness * z)
    
    return tf.constant(np.log(2.), dtype=self.dtype) + log_pdf_normal + log_cdf_skew

  def _cdf(self, x):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    z = (x - self.loc) / scale
    
    return special_math.ndtr(z) - 2. * tfp_math_special.owens_t(z, skewness)

  def _mean(self):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    delta = skewness / tf.math.sqrt(1. + tf.math.square(skewness))
    return self.loc + scale * delta * tf.constant(np.sqrt(2. / np.pi), dtype=self.dtype)

  def _variance(self):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    delta = skewness / tf.math.sqrt(1. + tf.math.square(skewness))
    
    c = tf.constant(2. / np.pi, dtype=self.dtype)
    return tf.math.square(scale) * (1. - c * tf.math.square(delta))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init:
      try:
        self._batch_shape()
      except ValueError:
        raise ValueError(
            'Arguments `loc`, `scale`, and `skewness` must have compatible '
            'shapes; loc.shape={}, scale.shape={}, skewness.shape={}.'.format(
                self.loc.shape, self.scale.shape, self.skewness.shape))
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access both arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))

    return assertions
