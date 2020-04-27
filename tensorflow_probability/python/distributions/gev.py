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
"""The GeneralizedExtremeValue distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import gev_cdf as gev_cdf_bijector
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


class GeneralizedExtremeValue(transformed_distribution.TransformedDistribution):
  """The scalar GeneralizedExtremeValue distribution
  with location `loc`, `scale` and `concentration` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; loc, scale, conc) = t(x; loc, scale, conc) ** (1 + conc) * exp(
  - t(x; loc, scale, conc) ) / scale
  where t(x) =
    * (1 + conc * (x - loc) / scale) ) ** (-1 / conc) when conc != 0;
    * exp(-(x - loc) / scale) when conc = 0.
  ```

  where `concentration = conc`.

  The cumulative density function of this distribution is,

  ```cdf(x; mu, sigma) = exp(-t(x))```

  The generalized extreme value distribution is a member of the
  [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family),
  i.e., it can be constructed as,

  ```none
  X ~ GeneralizedExtremeValue(loc=0, scale=1, concentration=conc)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tfp.distributions

  # Define a single scalar generalized extreme values distribution.
  dist = tfd.GeneralizedExtremeValue(loc=0., scale=3., concentration=0.9)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued generalized extreme values.
  # The first has mean 1 and scale 11, the second 2 and 22.
  dist = tfd.GeneralizedExtremeValue(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two scalar valued Logistics.
  # Both have mean 1, but different scales.
  dist = tfd.GeneralizedExtremeValue(loc=1., scale=1, concentration=[0, 0.9])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               scale,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='GeneralizedExtremeValue'):
    """Construct generalized extreme value distributions with location `loc`,
    scale `scale`, and concentration `concentration`.

    The parameters `loc`, `scale`, and `concentration` must be shaped in a way
    that supports broadcasting (e.g. `loc + scale` + `concentration` is valid).

    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s).
        scale must contain only positive values.
      concentration: Floating point tensor, the concentration of
        the distribution(s).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False`.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value `NaN` to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: `'GeneralizedExtremeValue'`.

    Raises:
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)
      dtype_util.assert_same_float_dtype([loc, scale, concentration])
      # Positive scale is asserted by the incorporated GEV bijector.
      self._gev_bijector = gev_cdf_bijector.GeneralizedExtremeValueCDF(
          loc=loc, scale=scale, concentration=concentration,
          validate_args=validate_args)

      # Because the uniform sampler generates samples in `[0, 1)` this would
      # cause samples to lie in `(inf, -inf]` instead of `(inf, -inf)`. To fix
      # this, we use `np.finfo(dtype_util.as_numpy_dtype(self.dtype).tiny`
      # because it is the smallest, positive, 'normal' number.
      batch_shape = distribution_util.get_broadcast_shape(loc, scale,
                                                          concentration)
      super(GeneralizedExtremeValue, self).__init__(
          # TODO(b/137665504): Use batch-adding meta-distribution to set the
          # batch shape instead of tf.ones.
          distribution=uniform.Uniform(
              low=np.finfo(dtype_util.as_numpy_dtype(dtype)).tiny,
              high=tf.ones(batch_shape, dtype=dtype),
              allow_nan_stats=allow_nan_stats),
          # The GEV bijector encodes the CDF function as the forward,
          # and hence needs to be inverted.
          bijector=invert_bijector.Invert(
              self._gev_bijector, validate_args=validate_args),
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('loc', 'scale', 'concentration'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, scale=0, concentration=0)

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._gev_bijector.loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._gev_bijector.scale

  @property
  def concentration(self):
    """Distribution parameter for shape."""
    return self._gev_bijector.concentration

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * tf.ones_like(self.loc)
    return 1. + tf.math.log(scale) + np.euler_gamma*(1. + self.concentration)

  def _log_prob(self, x):
    with tf.control_dependencies(self._gev_bijector._maybe_assert_valid_x(x)):
      scale = tf.convert_to_tensor(self.scale)
      z = (x - self.loc) / scale

      conc = tf.convert_to_tensor(self.concentration)
      equal_zero = tf.equal(conc, 0.)
      log_t = tf.where(equal_zero, -z,
                       -tf.math.log1p(z * conc) / conc)

      return (conc + 1) * log_t - tf.exp(log_t) - tf.math.log(scale)

  def _mean(self):
    conc = tf.convert_to_tensor(self.concentration)
    equal_zero = tf.equal(conc, 0.)
    less_than_one = tf.less(conc, 1.)

    mean_z = tf.where(equal_zero,
                      np.euler_gamma*tf.ones_like(conc),
                      tf.where(less_than_one,
                               tf.math.expm1(tf.math.lgamma(1. - conc)) / conc,
                               np.inf*tf.ones_like(conc))
                      )

    return self.loc + self.scale * mean_z

  def _stddev(self):
    conc = tf.convert_to_tensor(self.concentration)
    equal_zero = tf.equal(conc, 0.)
    less_than_half = tf.less(conc, 0.5)

    g1_square = tf.exp(tf.math.lgamma(1. - conc)) ** 2
    g2 = tf.exp(tf.math.lgamma(1. - 2.*conc))

    std_z = tf.where(equal_zero,
                     tf.ones_like(conc) * np.pi / np.sqrt(6),
                     tf.where(less_than_half,
                              tf.math.sqrt(g2 - g1_square) / tf.abs(conc),
                              np.inf*tf.ones_like(conc))
                     )

    return self.scale * tf.ones_like(self.loc) * std_z

  def _mode(self):
    conc = tf.convert_to_tensor(self.concentration)
    equal_zero = tf.equal(conc, 0.)

    # Use broadcasting rules to calculate the full broadcast sigma.
    mode_z = tf.where(equal_zero,
                      tf.zeros_like(conc),
                      tf.math.expm1(-conc * tf.math.log1p(conc)) /conc)

    return self.loc + self.scale * mode_z

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector. Consider switching to
    # Chain([Softplus(), Log()]) to lighten the doubly-exponential right tail.
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return self._gev_bijector._parameter_control_dependencies(is_init)  # pylint: disable=protected-access
