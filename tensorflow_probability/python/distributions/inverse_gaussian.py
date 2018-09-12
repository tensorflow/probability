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
from tensorflow.python.framework import tensor_shape

__all__ = [
    "InverseGaussian",
]


class InverseGaussian(tf.distributions.Distribution):
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

  TODO(jmiao): Add mapping to R and Python scipy's parameterization.
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
    with tf.name_scope(name, values=[loc, concentration]):
      self._loc = tf.convert_to_tensor(loc, name="loc")
      self._concentration = tf.convert_to_tensor(concentration,
                                                 name="concentration")
      with tf.control_dependencies([
          tf.assert_positive(self._loc),
          tf.assert_positive(self._concentration)] if validate_args else []):
        self._loc = tf.identity(self._loc, name="loc")
        self._concentration = tf.identity(self._concentration,
                                          name="concentration")
        tf.assert_same_float_dtype([self._loc, self._concentration])
    super(InverseGaussian, self).__init__(
        dtype=self._loc.dtype,
        reparameterization_type=tf.distributions.NOT_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._concentration],
        name=name)

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
        tf.shape(self.loc), tf.shape(self.concentration))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.loc.shape, self.concentration.shape)

  def _event_shape(self):
    return tensor_shape.scalar()

  # TODO(b/112596766): Add  _sample_n(), _mean(), _variance(), _cdf().

  def _log_prob(self, x):
    with tf.control_dependencies([
        tf.assert_greater(
            x, tf.cast(0., x.dtype.base_dtype),
            message="x must be positive."
        )] if self.validate_args else []):

      return (0.5 * (tf.log(self.concentration) -
                     np.log(2. * np.pi) -
                     3. * tf.log(x))  +
              (-self.concentration * (x - self.loc) ** 2.) /
              (2. * self.loc ** 2. * x))

