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
"""The NormalInverseGaussian distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import inverse_gaussian
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import bessel

__all__ = [
    'NormalInverseGaussian',
]


def _log1px2(x):
  """Safely compute log(1 + x ** 2).

  For small x, use log1p(x ** 2). For large x(x >> 1), use 2 * log(x). Also
  avoid nan grad using double-where for x ~= 0.

  Args:
    x: float `Tensor`.

  Returns:
    y: log(1 + x ** 2).
  """
  # The idea with this is to use 2 log(x) when x** 2 >> 1, so that adding 1
  # doesn't matter. This happens when x >> 1 / sqrt(eps). But this causes
  # grad problems for zero input:
  #
  # If x is zero, the log(1 + x**2) is log(1) = 0. But then 2 * log(x) is
  # 2 * log(0) = 2 * -Inf, which causes problems. So for 0 input, we need a safe
  # value for the negative case and use the double-where trick
  # (see, eg, https://github.com/google/jax/issues/1052)
  finfo = np.finfo(dtype_util.as_numpy_dtype(x.dtype))
  is_basically_zero = tf.abs(x) < finfo.tiny
  safe_x = tf.where(is_basically_zero, tf.ones_like(x), x)
  return tf.where(
      is_basically_zero,
      tf.abs(x),
      tf.where(
          tf.abs(x) * np.sqrt(finfo.eps) <= 1.,
          tf.math.log1p(safe_x**2.),
          2 * tf.math.log(tf.math.abs(safe_x))))


class NormalInverseGaussian(distribution.AutoCompositeTensorDistribution):
  """Normal Inverse Gaussian distribution.

  The [Normal-inverse Gaussian distribution]
  (https://en.wikipedia.org/wiki/Normal-inverse_Gaussian_distribution)
  is parameterized by a `loc`, `tailweight`, `skewness` and `scale` parameter.

  #### Mathematical Details

  The name of this distribution refers to it being a variance mean mixture.
  In other words if `x` is sampled via:

  ```none
  z ~ InverseGaussian(1 / gamma, 1.)
  x ~ Normal(loc + skewness * z, scale * z)
  ```
  then `x ~ NormalInverseGaussian(loc, scale, tailweight, skewness)`.

  where `gamma = sqrt(tailweight ** 2 - skewness ** 2)`.


  The probability density function (pdf) is,

  ```none
  pdf(x; mu, sigma, alpha, beta) = [alpha * sigma * K1(alpha * g)] / (pi * g)
                                   exp(sigma * gamma + beta * (x - loc))
  ```

  where
  * `loc = mu`
  * `tailweight = alpha`
  * `skewness = beta`
  * `scale = sigma`
  * `g = sqrt(sigma ** 2 + (x - mu) ** 2)`
  * `gamma = sqrt(alpha ** 2 - beta ** 2)`
  * `K1(x)` is the modified Bessel function of the second kind with
    order parameter 1.

  The support of the distribution is defined on `(-infinity, infinity)`.

  Mapping to R and Python scipy's parameterization:
  * R: GeneralizedHyperbolic.NIG
    - mu = loc
    - delta = scale
    - alpha = tailweight
    - beta = skewness
  * Python: scipy.stats.norminvgauss
    - a = tailweight
    - b = skewness
    - loc = loc
    - Note that `scipy.stats.norminvgauss` implements the distribution as a
      location-scale family. However, in the original paper, and other
      implementations (such as R) do not implement it this way. Thus the
      `scale` parameters here and scipy don't match unless `scale = 1`.

  Warning: As mentioned above, this distribution is __not__ a location-scale
  family. Specifically:

  ```none
  NIG(loc, scale, alpha, beta) != loc + scale * NIG(0, 1, alpha, beta).
  ```

  """

  def __init__(self,
               loc,
               scale,
               tailweight,
               skewness,
               validate_args=False,
               allow_nan_stats=True,
               name='NormalInverseGaussian'):
    """Constructs Normal-inverse Gaussian distribution.

    Args:
      loc: Floating point `Tensor`, the location params of the distribution(s).
      scale: Positive floating point `Tensor`, the scale params of the
        distribution(s).
      tailweight: Positive floating point `Tensor`, the tailweight params of the
        distribution(s). Expect `|tailweight| >= |skewness|`.
      skewness: Floating point `Tensor`, the skewness params of the
        distribution(s). Expect `|tailweight| >= |skewness|`.
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
        Default value: 'NormalInverseGaussian'.
    """
    parameters = dict(locals())
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype([loc, scale, tailweight, skewness],
                                      dtype_hint=tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._tailweight = tensor_util.convert_nonref_to_tensor(
          tailweight, dtype=dtype, name='tailweight')
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, dtype=dtype, name='skewness')

      super(NormalInverseGaussian, self).__init__(
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
        tailweight=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        # TODO(b/169874884): Support decoupled parameterization.
        skewness=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Location parameter."""
    return self._loc

  @property
  def scale(self):
    """Scale parameter."""
    return self._scale

  @property
  def tailweight(self):
    """Tailweight parameter."""
    return self._tailweight

  @property
  def skewness(self):
    """Skewness parameter."""
    return self._skewness

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    tailweight = tf.convert_to_tensor(self.tailweight)
    skewness = tf.convert_to_tensor(self.skewness)
    ig_seed, normal_seed = samplers.split_seed(
        seed, salt='normal_inverse_gaussian')
    batch_shape = self._batch_shape_tensor(
        loc=loc,
        scale=scale,
        tailweight=tailweight,
        skewness=skewness)
    w = tailweight * tf.math.exp(0.5 * tf.math.log1p(
        -tf.math.square(skewness / tailweight)))
    w = tf.broadcast_to(w, batch_shape)
    ig_samples = inverse_gaussian.InverseGaussian(
        scale / w, tf.math.square(scale)).sample(n, seed=ig_seed)

    sample_shape = ps.concat([[n], batch_shape], axis=0)
    normal_samples = samplers.normal(
        shape=ps.convert_to_shape_tensor(sample_shape),
        mean=0., stddev=1., dtype=self.dtype, seed=normal_seed)
    return (loc + tf.math.sqrt(ig_samples) * (
        skewness * tf.math.sqrt(ig_samples) + normal_samples))

  def _log_prob(self, x):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    tailweight = tf.convert_to_tensor(self.tailweight)
    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    y = (x - loc) / scale
    z = _log1px2(y)
    w = tailweight * tf.math.exp(0.5 * tf.math.log1p(
        -tf.math.square(skewness / tailweight)))
    log_unnormalized_prob = (
        bessel.log_bessel_kve(
            numpy_dtype(1.), tailweight * scale * tf.math.exp(0.5 * z)) -
        0.5 * z - tailweight * scale * tf.math.exp(0.5 * z))
    log_unnormalized_prob = log_unnormalized_prob + scale * skewness * y
    log_normalization = np.log(np.pi) - scale * w - tf.math.log(tailweight)
    return log_unnormalized_prob - log_normalization

  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    tailweight = tf.convert_to_tensor(self.tailweight)
    w = tailweight * tf.math.exp(0.5 * tf.math.log1p(
        -tf.math.square(skewness / tailweight)))
    return loc + (skewness * scale / w)

  def _variance(self):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)
    tailweight = tf.convert_to_tensor(self.tailweight)
    scale = tf.broadcast_to(
        scale, self._batch_shape_tensor(
            scale=scale, tailweight=tailweight, skewness=skewness))
    w = tailweight * tf.math.exp(0.5 * tf.math.log1p(
        -tf.math.square(skewness / tailweight)))
    return scale * tf.math.square(tailweight) / w ** 3

  def _default_event_space_bijector(self):
    return identity_bijector.Identity()

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    tailweight_is_ref = tensor_util.is_ref(self.tailweight)
    tailweight = tf.convert_to_tensor(self.tailweight)
    if (is_init != tailweight_is_ref and
        is_init != tensor_util.is_ref(self.skewness)):
      assertions.append(assert_util.assert_less(
          tf.math.abs(self.skewness),
          tailweight,
          message='Expect `tailweight > |skewness|`'))
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale,
          message='Argument `scale` must be positive.'))
    if is_init != tailweight_is_ref:
      assertions.append(assert_util.assert_positive(
          tailweight,
          message='Argument `tailweight` must be positive.'))

    return assertions
