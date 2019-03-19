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
import tensorflow as tf
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import special_math

__all__ = [
    "InverseGaussian",
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
               name="InverseGaussian"):
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
    with tf.compat.v1.name_scope(name, values=[loc, concentration]):
      dtype = dtype_util.common_dtype([loc, concentration],
                                      preferred_dtype=tf.float32)
      loc = tf.convert_to_tensor(value=loc, name="loc", dtype=dtype)
      concentration = tf.convert_to_tensor(
          value=concentration, name="concentration", dtype=dtype)
      with tf.control_dependencies([
          tf.compat.v1.assert_positive(loc),
          tf.compat.v1.assert_positive(concentration)
      ] if validate_args else []):
        self._loc = tf.identity(loc, name="loc")
        self._concentration = tf.identity(concentration, name="concentration")
      tf.debugging.assert_same_float_dtype([self._loc, self._concentration])
    super(InverseGaussian, self).__init__(
        dtype=self._loc.dtype,
        reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._concentration],
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

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(input=self.loc), tf.shape(input=self.concentration))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.concentration.shape)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    # See https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution or
    # https://www.jstor.org/stable/2683801
    seed = seed_stream.SeedStream(seed, "inverse_gaussian")
    shape = tf.concat([[n], self.batch_shape_tensor()], axis=0)
    sampled_chi2 = (tf.random.normal(
        shape, mean=0., stddev=1., seed=seed(), dtype=self.dtype))**2.
    sampled_uniform = tf.random.uniform(
        shape, minval=0., maxval=1., seed=seed(), dtype=self.dtype)
    sampled = (
        self.loc + self.loc ** 2. * sampled_chi2 / (2. * self.concentration) -
        self.loc / (2. * self.concentration) *
        (4. * self.loc * self.concentration * sampled_chi2 +
         (self.loc * sampled_chi2) ** 2) ** 0.5)
    return tf.where(
        sampled_uniform <= self.loc / (self.loc + sampled),
        sampled,
        self.loc ** 2 / sampled)

  def _log_prob(self, x):
    with tf.control_dependencies([
        tf.compat.v1.assert_greater(
            x, tf.cast(0., x.dtype.base_dtype), message="x must be positive.")
    ] if self.validate_args else []):

      return (0.5 * (tf.math.log(self.concentration) - np.log(2. * np.pi) -
                     3. * tf.math.log(x)) + (-self.concentration *
                                             (x - self.loc)**2.) /
              (2. * self.loc**2. * x))

  def _cdf(self, x):
    with tf.control_dependencies([
        tf.compat.v1.assert_greater(
            x, tf.cast(0., x.dtype.base_dtype), message="x must be positive.")
    ] if self.validate_args else []):

      return (
          special_math.ndtr(
              ((self.concentration / x) ** 0.5 *
               (x / self.loc - 1.))) +
          tf.exp(2. * self.concentration / self.loc) *
          special_math.ndtr(
              - (self.concentration / x) ** 0.5 *
              (x / self.loc + 1)))

  @distribution_util.AppendDocstring(
      """The mean of inverse Gaussian is the `loc` parameter.""")
  def _mean(self):
    # Shape is broadcasted with + tf.zeros_like().
    return self.loc + tf.zeros_like(self.concentration)

  @distribution_util.AppendDocstring(
      """The variance of inverse Gaussian is `loc` ** 3 / `concentration`.""")
  def _variance(self):
    return self.loc ** 3 / self.concentration
