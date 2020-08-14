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
"""The InverseGaussian distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.bijectors import scale as scale_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import special_math
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'InverseGaussian',
]


class InverseGaussian(distribution.Distribution):
  """Inverse Gaussian distribution.

  The [inverse Gaussian distribution]
  (https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)
  is parameterized by a `loc` and a `concentration` parameter. It's also known
  as the Wald distribution. Some, e.g., the Python scipy package, refer to the
  special case when `loc` is 1 as the Wald distribution.

  The "inverse" in the name does not refer to the distribution associated to
  the multiplicative inverse of a random variable. Rather, the cumulant
  generating function of this distribution is the inverse to that of a Gaussian
  random variable.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, lambda) = [lambda / (2 pi x ** 3)] ** 0.5
                       exp{-lambda(x - mu) ** 2 / (2 mu ** 2 x)}
  ```

  where
  * `loc = mu`
  * `concentration = lambda`.

  The support of the distribution is defined on `(0, infinity)`.

  Mapping to R and Python scipy's parameterization:
  * R: statmod::invgauss
     - mean = loc
     - shape = concentration
     - dispersion = 1 / concentration. Used only if shape is NULL.
  * Python: scipy.stats.invgauss
     - mu = loc / concentration
     - scale = concentration
  """

  def __init__(self,
               loc,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='InverseGaussian'):
    """Constructs inverse Gaussian distribution with `loc` and `concentration`.

    Args:
      loc: Floating-point `Tensor`, the loc params. Must contain only positive
        values.
      concentration: Floating-point `Tensor`, the concentration params.
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
        Default value: `False` (i.e. do not validate args).
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
        Default value: `True`.
      name: Python `str` name prefixed to Ops created by this class.
        Default value: 'InverseGaussian'.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([loc, concentration],
                                      dtype_hint=tf.float32)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, dtype=dtype, name='concentration')
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')

      super(InverseGaussian, self).__init__(
          dtype=self._loc.dtype,
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(loc=0, concentration=0)

  @property
  def loc(self):
    """Location parameter."""
    return self._loc

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  def _batch_shape_tensor(self, loc=None, concentration=None):
    return ps.broadcast_shape(
        ps.shape(self.loc if loc is None else loc),
        ps.shape(
            self.concentration if concentration is None else concentration))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.concentration.shape)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    # See https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution or
    # https://www.jstor.org/stable/2683801
    concentration = tf.convert_to_tensor(self.concentration)
    loc = tf.convert_to_tensor(self.loc)
    chi2_seed, unif_seed = samplers.split_seed(seed, salt='inverse_gaussian')
    shape = ps.concat([[n], self._batch_shape_tensor(
        loc=loc, concentration=concentration)], axis=0)
    sampled_chi2 = tf.square(samplers.normal(
        shape, seed=chi2_seed, dtype=self.dtype))
    sampled_uniform = samplers.uniform(
        shape, seed=unif_seed, dtype=self.dtype)
    sampled = (
        loc + tf.square(loc) * sampled_chi2 / (2. * concentration) -
        loc / (2. * concentration) *
        tf.sqrt(4. * loc * concentration * sampled_chi2 +
                tf.square(loc * sampled_chi2)))
    return tf.where(sampled_uniform <= loc / (loc + sampled),
                    sampled, tf.square(loc) / sampled)

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    loc = tf.convert_to_tensor(self.loc)
    return (0.5 * (tf.math.log(concentration) - np.log(2. * np.pi) -
                   3. * tf.math.log(x)) +
            (-concentration * tf.math.squared_difference(x, loc)) /
            (2. * tf.square(loc) * x))

  def _cdf(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    loc = tf.convert_to_tensor(self.loc)
    return (
        special_math.ndtr((tf.math.rsqrt(x / concentration) * (x / loc - 1.))) +
        tf.exp(2. * concentration / loc) *
        special_math.ndtr(-tf.math.rsqrt(x / concentration) * (x / loc + 1)))

  @distribution_util.AppendDocstring(
      """The mean of inverse Gaussian is the `loc` parameter.""")
  def _mean(self):
    # Shape is broadcasted with + tf.zeros_like().
    return self.loc + tf.zeros_like(self.concentration)

  @distribution_util.AppendDocstring(
      """The variance of inverse Gaussian is `loc` ** 3 / `concentration`.""")
  def _variance(self):
    return self.loc ** 3 / self.concentration

  def _default_event_space_bijector(self):
    return chain_bijector.Chain([
        softplus_bijector.Softplus(validate_args=self.validate_args),
        scale_bijector.Scale(scale=-1., validate_args=self.validate_args),
        exp_bijector.Log(validate_args=self.validate_args),
        softplus_bijector.Softplus(validate_args=self.validate_args)
    ], validate_args=self.validate_args)

  def _sample_control_dependencies(self, x):
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_non_negative(
        x, message='Sample must be non-negative.'))
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.concentration):
      assertions.append(assert_util.assert_positive(
          self.concentration,
          message='Argument `concentration` must be positive.'))
    if is_init != tensor_util.is_ref(self.loc):
      assertions.append(assert_util.assert_positive(
          self.loc,
          message='Argument `loc` must be positive.'))
    return assertions
