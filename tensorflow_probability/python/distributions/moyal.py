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
"""The Moyal distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import moyal_cdf as moyal_cdf_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'Moyal',
]


# TODO(b/182603117): Remove `AutoCompositeTensor` subclass when
# `TransformedDistribution` is converted to `CompositeTensor`.
class Moyal(transformed_distribution.TransformedDistribution,
            distribution.AutoCompositeTensorDistribution):
  """The Moyal distribution with location `loc` and `scale` parameters.

  #### Mathematical details

  The probability density function (pdf) of this distribution is,

  ```none
  pdf(x; mu, sigma) = exp(- 1/2 * ((x - mu) / sigma + exp(-(x - mu) / sigma))) /
    (sigma * sqrt(2 pi))
  ```

  where `loc = mu` and `scale = sigma`.

  The cumulative density function of this distribution is,

  ```none
  cdf(x; mu, sigma) = erfc(exp(- 1/2 * (x - mu) / sigma) / sqrt(2))
  ```

  The Moyal distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ Moyal(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tfp.distributions

  # Define a single scalar Moyal distribution.
  dist = tfd.Moyal(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Moyals.
  # The first has mean 1 and scale 11, the second 2 and 22.
  dist = tfd.Moyal(loc=[1, 2.], scale=[11, 22.])

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
  dist = tfd.Moyal(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Moyal'):
    """Construct Moyal distributions with location and scale `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor, the means of the distribution(s).
      scale: Floating point tensor, the scales of the distribution(s).
        scale must contain only positive values.
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
        Default value: `'Moyal'`.

    Raises:
      TypeError: if loc and scale are different dtypes.


    #### References

    [1] J.E. Moyal, "XXX. Theory of ionization fluctuations",
       The London, Edinburgh, and Dublin Philosophical Magazine
       and Journal of Science.
       https://www.tandfonline.com/doi/abs/10.1080/14786440308521076
    [2] G. Cordeiro, J. Nobre, R. Pescim, E. Ortega,
        "The beta Moyal: a useful skew distribution",
        https://www.arpapress.com/Volumes/Vol10Issue2/IJRRAS_10_2_02.pdf
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype([loc, scale])
      # Positive scale is asserted by the incorporated Moyal bijector.
      self._moyal_bijector = moyal_cdf_bijector.MoyalCDF(
          loc=loc, scale=scale, validate_args=validate_args)

      # Because the uniform sampler generates samples in `[0, 1)` this would
      # cause samples to lie in `(inf, -inf]` instead of `(inf, -inf)`. To fix
      # this, we use `np.finfo(dtype_util.as_numpy_dtype(self.dtype).tiny`
      # because it is the smallest, positive, 'normal' number.
      super(Moyal, self).__init__(
          # TODO(b/137665504): Use batch-adding meta-distribution to set the
          # batch shape instead of tf.ones.
          distribution=uniform.Uniform(
              low=np.finfo(dtype_util.as_numpy_dtype(dtype)).tiny,
              high=tf.ones([], dtype=dtype),
              allow_nan_stats=allow_nan_stats),
          # The Moyal bijector encodes the CDF function as the forward,
          # and hence needs to be inverted.
          bijector=invert_bijector.Invert(
              self._moyal_bijector, validate_args=validate_args),
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._moyal_bijector.loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._moyal_bijector.scale

  experimental_is_sharded = False

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast sigma.
    scale = self.scale * tf.ones_like(self.loc)
    return (0.5 * (1. + 2 * tf.math.log(scale) +
                   np.euler_gamma + np.log(4. * np.pi)))

  def _log_prob(self, x):
    scale = tf.convert_to_tensor(self.scale)
    z = (x - self.loc) / scale
    return (- 0.5 * (z + tf.exp(-z)) - 0.5 * np.log(2. * np.pi)
            - tf.math.log(scale))

  def _mean(self):
    return self.loc + self.scale * (np.euler_gamma + np.log(2.))

  def _stddev(self):
    return self.scale * tf.ones_like(self.loc) * np.pi / np.sqrt(2.)

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector. Consider switching to
    # Chain([Softplus(), Log()]) to lighten the doubly-exponential right tail.
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    return self._moyal_bijector._parameter_control_dependencies(is_init)  # pylint: disable=protected-access


@kullback_leibler.RegisterKL(Moyal, Moyal)
def _kl_moyal_moyal(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Moyal.

  Args:
    a: instance of a Moyal distribution object.
    b: instance of a Moyal distribution object.
    name: (optional) Name to use for created operations.
      default is 'kl_moyal_moyal'.

  Returns:
    Batchwise KL(a || b)
  """
  with tf.name_scope(name or 'kl_moyal_moyal'):
    # This KL divergence was calculated in Mathematica
    a_loc = tf.convert_to_tensor(a.loc)
    b_loc = tf.convert_to_tensor(b.loc)
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)
    exp_term = tf.math.exp(a_scale * np.log(2.) / b_scale +
                           (b_loc - a_loc) / b_scale +
                           tf.math.lgamma(0.5 + a_scale / b_scale)
                           - 0.5 * np.log(np.pi))
    return 0.5 * (exp_term + (a_loc - b_loc) / b_scale - 1 +
                  (a_scale / b_scale - 1) * (np.euler_gamma + np.log(2.)) +
                  2 * tf.math.log(b_scale / a_scale))
