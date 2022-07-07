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
"""The Cauchy distribution class."""

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'Cauchy',
]


class Cauchy(distribution.AutoCompositeTensorDistribution):
  """The Cauchy distribution with location `loc` and scale `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale) = 1 / (pi scale (1 + z**2))
  z = (x - loc) / scale
  ```
  where `loc` is the location, and `scale` is the scale.

  The Cauchy distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e.
  `Y ~ Cauchy(loc, scale)` is equivalent to,

  ```none
  X ~ Cauchy(loc=0, scale=1)
  Y = loc + scale * X
  ```

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  tfd = tfp.distributions

  # Define a single scalar Cauchy distribution.
  dist = tfd.Cauchy(loc=0., scale=3.)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)

  # Define a batch of two scalar valued Cauchy distributions.
  dist = tfd.Cauchy(loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  dist.sample([3])

  # Arguments are broadcast when possible.
  # Define a batch of two scalar valued Cauchy distributions.
  # Both have median 1, but different scales.
  dist = tfd.Cauchy(loc=1., scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.)
  ```

  """

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='Cauchy'):
    """Construct Cauchy distributions.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the modes of the distribution(s).
      scale: Floating point tensor; the half-widths of the distribution(s) at
        their half-maximums. Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc` and `scale` have different `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale], tf.float32)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype([self._loc, self._scale])
      super(Cauchy, self).__init__(
          dtype=self._scale.dtype,
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
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(loc=loc, scale=scale)
    shape = ps.concat([[n], batch_shape], 0)
    probs = samplers.uniform(
        shape=shape, minval=0., maxval=1., dtype=self.dtype, seed=seed)
    return self._quantile(probs, loc=loc, scale=scale)

  def _log_prob(self, x):
    npdt = dtype_util.as_numpy_dtype(self.dtype)
    scale = tf.convert_to_tensor(self.scale)
    log_unnormalized_prob = -tf.math.log1p(tf.square(self._z(x, scale=scale)))
    log_normalization = npdt(np.log(np.pi)) + tf.math.log(scale)
    return log_unnormalized_prob - log_normalization

  def _cdf(self, x):
    return tf.atan(self._z(x)) / np.pi + 0.5

  def _log_cdf(self, x):
    return tf.math.log1p(2 / np.pi * tf.atan(self._z(x))) - np.log(2)

  def _entropy(self):
    h = np.log(4 * np.pi) + tf.math.log(self.scale)
    return h * tf.ones_like(self.loc)

  def _quantile(self, p, loc=None, scale=None):
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    scale = tf.convert_to_tensor(self.scale if scale is None else scale)
    return loc + scale * tf.tan(np.pi * (p - 0.5))

  def _mode(self):
    return self.loc * tf.ones_like(self.scale)

  def _z(self, x, loc=None, scale=None):
    """Standardize input `x`."""
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    scale = tf.convert_to_tensor(self.scale if scale is None else scale)
    with tf.name_scope('standardize'):
      return (x - loc) / scale

  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with tf.name_scope('reconstruct'):
      return z * self.scale + self.loc

  def _mean(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      raise ValueError('`mean` is undefined for Cauchy distribution.')

  def _stddev(self):
    if self.allow_nan_stats:
      return tf.fill(self.batch_shape_tensor(),
                     dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      raise ValueError('`stddev` is undefined for Cauchy distribution.')

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector (consider one that
    # transforms away the heavy tails).
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(assert_util.assert_positive(
          self.scale, message='Argument `scale` must be positive.'))
    return assertions


@kullback_leibler.RegisterKL(Cauchy, Cauchy)
def _kl_cauchy_cauchy(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b) with a and b Cauchy.

  Note that this KL divergence is symmetric in its arguments.

  Args:
    a: instance of a Cauchy distribution object.
    b: instance of a Cauchy distribution object.
    name: Name to use for created operations.
      Default value: `None` (i.e., `'kl_cauchy_cauchy'`).

  Returns:
    kl_div: Batchwise KL(a || b)

  #### References

  [1] Frederic Chyzak and Frank Nielsen. A closed-form formula for the
  Kullback-Leibler divergence between Cauchy distributions.
  https://arxiv.org/abs/1905.10965
  """
  with tf.name_scope(name or 'kl_cauchy_cauchy'):
    a_scale = tf.convert_to_tensor(a.scale)
    b_scale = tf.convert_to_tensor(b.scale)
    b_loc = tf.convert_to_tensor(b.loc)
    scale_sum_square = tf.math.square(a_scale + b_scale)
    loc_diff_square = tf.math.squared_difference(a.loc, b_loc)

    return (tf.math.log(scale_sum_square + loc_diff_square) -
            np.log(4.) - tf.math.log(a_scale) - tf.math.log(b_scale))
