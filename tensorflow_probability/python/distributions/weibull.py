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
"""The Weibull distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.bijectors import weibull_cdf as weibull_cdf_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util


class Weibull(transformed_distribution.TransformedDistribution,
              distribution.AutoCompositeTensorDistribution):
  """The Weibull distribution with 'concentration' and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; lambda, k) =
    k / lambda * (x / lambda) ** (k - 1) * exp(-(x / lambda) ** k)
  ```

  where `concentration = k` and `scale = lambda`.

  The cumulative density function of this distribution is,

  ```none
  cdf(x; lambda, k) = 1 - exp(-(x / lambda) ** k)
  ```

  The Weibull distribution includes the Exponential and Rayleigh distributions
  as special cases:

  ```none
  Exponential(rate) = Weibull(concentration=1., 1. / rate)
  ```

  ```none
  Rayleigh(scale) = Weibull(concentration=2., sqrt(2.) * scale)
  ```

  #### Examples

  Example of initialization of one distribution.

  ```python
  tfd = tfp.distributions

  # Define a single scalar Weibull distribution.
  dist = tfd.Weibull(concentration=1., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)
  ```

  Example of initialization of a 3-batch of distributions with varying scales
  and concentrations.

  ```python
  tfd = tfp.distributions

  # Define a 3-batch of Weibull distributions.
  scale = [1., 3., 45.]
  concentration = [2.5, 22., 7.]
  dist = tfd.Weibull(concentration=concentration, scale=scale)

  # Evaluate the cdfs at 1.
  dist.cdf(1.)    # shape: [3]
  ```
  """

  def __init__(self,
               concentration,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Weibull'):
    """Construct Weibull distributions.

    The parameters `concentration` and `scale` must be shaped in a way that
    supports broadcasting (e.g. `concentration + scale` is a valid operation).

    Args:
     concentration: Positive Float-type `Tensor`, the concentration param of the
       distribution. Must contain only positive values.
     scale: Positive Float-type `Tensor`, the scale param of the distribution.
       Must contain only positive values.
     validate_args: Python `bool` indicating whether arguments should be checked
       for correctness.
     allow_nan_stats: Python `bool` indicating whether nan values should be
       allowed.
     name: Python `str` name given to ops managed by this class.
       Default value: `'Weibull'`.

    Raises:
      TypeError: if concentration and scale are different dtypes.

    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([concentration, scale],
                                      dtype_hint=tf.float32)
      concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)
      scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      # Positive scale and concentration is asserted by the incorporated
      # Weibull bijector.
      self._weibull_bijector = weibull_cdf_bijector.WeibullCDF(
          scale=scale, concentration=concentration, validate_args=validate_args)

      super(Weibull, self).__init__(
          distribution=uniform.Uniform(
              low=tf.zeros([], dtype=dtype),
              high=tf.ones([], dtype=dtype),
              allow_nan_stats=allow_nan_stats),
          # The Weibull bijector encodes the CDF function as the forward,
          # and hence needs to be inverted.
          bijector=invert_bijector.Invert(
              self._weibull_bijector, validate_args=validate_args),
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        concentration=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def concentration(self):
    """Distribution parameter for the concentration."""
    return self._weibull_bijector.concentration

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._weibull_bijector.scale

  experimental_is_sharded = False

  def _entropy(self):
    one = tf.constant(1., dtype=self.dtype)
    concentration = tf.convert_to_tensor(self.concentration)
    euler_gamma = tf.constant(np.euler_gamma, dtype=self.dtype)
    return (euler_gamma * (one - tf.math.reciprocal(concentration)) +
            tf.math.log(self.scale) - tf.math.log(concentration) + one)

  def _log_prob(self, x):
    return self._weibull_bijector.forward_log_det_jacobian(x, event_ndims=0)

  def _mean(self):
    one = tf.constant(1., dtype=self.dtype)
    return self.scale * tf.exp(
        tf.math.lgamma(one + tf.math.reciprocal(self.concentration)))

  def _variance(self):
    concentration = tf.convert_to_tensor(self.concentration)
    one = tf.constant(1., dtype=self.dtype)
    two = tf.constant(2., dtype=self.dtype)
    return (
        tf.square(self.scale) *
        (tf.exp(tf.math.lgamma(one + two / concentration)) -
         tf.exp(two * tf.math.lgamma(one + tf.math.reciprocal(concentration)))))

  def _stddev(self):
    concentration = tf.convert_to_tensor(self.concentration)
    one = tf.constant(1., dtype=self.dtype)
    two = tf.constant(2., dtype=self.dtype)
    return (self.scale * tf.math.sqrt(
        tf.exp(tf.math.lgamma(one + two / concentration)) -
        tf.exp(two * tf.math.lgamma(one + tf.math.reciprocal(concentration)))))

  def _mode(self):
    one = tf.constant(1., dtype=self.dtype)
    concentration = tf.convert_to_tensor(self.concentration)
    return (((concentration - one) / concentration)**
            tf.math.reciprocal(concentration) * self.scale)

  def _parameter_control_dependencies(self, is_init):
    return self._weibull_bijector._parameter_control_dependencies(is_init)  # pylint: disable=protected-access

  def _sample_control_dependencies(self, x):
    """Checks the validity of a sample."""
    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(
        assert_util.assert_non_negative(
            x, message='Sample must be non-negative.'))
    return assertions

  def _default_event_space_bijector(self):
    return softplus_bijector.Softplus(validate_args=self.validate_args)


@kullback_leibler.RegisterKL(Weibull, Weibull)
def _kl_weibull_weibull(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Weibull.

  Args:
    a: instance of a Weibull distribution object.
    b: instance of a Weibull distribution object.
    name: (optional) Name to use for created operations. default is
      'kl_weibull_weibull'.

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_weibull_weibull'):
    # Result from https://arxiv.org/pdf/1310.3713.pdf
    a_concentration = tf.convert_to_tensor(a.concentration)
    b_concentration = tf.convert_to_tensor(b.concentration)
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)

    return ((tf.math.log(a_concentration) -
             a_concentration * tf.math.log(a_scale)) -
            (tf.math.log(b_concentration) -
             b_concentration * tf.math.log(b_scale)) +
            ((a_concentration - b_concentration) *
             (tf.math.log(a_scale) - np.euler_gamma / a_concentration)) +
            ((a_scale / b_scale)**b_concentration *
             tf.exp(tf.math.lgamma(b_concentration / a_concentration + 1.))) -
            1.)
