# Copyright 2022 The TensorFlow Probability Authors.
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
"""The Two-Piece Normal distribution class."""

# Dependency imports
import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
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
from tensorflow_probability.python.math.numeric import log1psquare

__all__ = [
    'TwoPieceNormal',
]

NUMPY_MODE = False


def _numpy_cast(x, dtype):
  # TODO(b/223684173): Many special math routines don't respect the input dtype.
  if NUMPY_MODE:
    return tf.cast(x, dtype)
  else:
    return x


def standardize(value, loc, scale, skewness):
  """Apply mean-variance-skewness standardization to input `value`.

  Note that scale and skewness can be negative.

  Args:
    value: Floating-point tensor; the value(s) to be standardized.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  return (value - loc) / tf.math.abs(scale) * tf.math.abs(
      tf.where(value < loc, skewness, tf.math.reciprocal(skewness)))


def cdf(value, loc, scale, skewness):
  """Compute cumulative distribution function of Two-Piece Normal distribution.

  Note that scale and skewness can be negative.

  Args:
    value: Floating-point tensor; where to compute the cdf.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  one = tf.constant(1., dtype=loc.dtype)
  two = tf.constant(2., dtype=loc.dtype)

  z = standardize(value, loc=loc, scale=scale, skewness=skewness)
  normal_cdf = _numpy_cast(special_math.ndtr(z), loc.dtype)

  squared_skewness = tf.math.square(skewness)
  return tf.math.reciprocal(one + squared_skewness) * tf.where(
      z < 0.,
      two * normal_cdf,
      one - squared_skewness + two * squared_skewness * normal_cdf)


def quantile(value, loc, scale, skewness):
  """Compute quantile function (inverse cdf) of Two-Piece Normal distribution.

  Note that scale and skewness can be negative.

  Args:
    value: Floating-point tensor; where to compute the quantile function.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  half = tf.constant(0.5, dtype=loc.dtype)
  one = tf.constant(1., dtype=loc.dtype)
  two = tf.constant(2., dtype=loc.dtype)

  squared_skewness = tf.math.square(skewness)
  cond = value < tf.math.reciprocal(one + squared_skewness)

  # Here we use the following fact:
  # X ~ Normal(loc=0, scale=1) => 2 * X**2 ~ Gamma(alpha=0.5, beta=1)
  probs = (one - value * (one + squared_skewness)) * tf.where(
      cond, one, -tf.math.reciprocal(squared_skewness))
  gamma_quantile = _numpy_cast(tfp_math.igammainv(half, p=probs), loc.dtype)

  abs_skewness = tf.math.abs(skewness)
  adj_scale = tf.math.abs(scale) * tf.where(
      cond, -tf.math.reciprocal(abs_skewness), abs_skewness)

  return loc + adj_scale * tf.math.sqrt(two * gamma_quantile)


class TwoPieceNormal(distribution.AutoCompositeTensorDistribution):
  """The Two-Piece Normal distribution.

  The Two-Piece Normal generalizes the Normal distribution with an additional
  shape parameter. Under the general formulation proposed by [Fernández and
  Steel (1998)][2], it is parameterized by location `loc`, scale `scale`, and
  shape `skewness`. If `skewness` is above one, the distribution becomes right-
  skewed (or positively skewed). If `skewness` is greater than zero and less
  than one, the distribution becomes left-skewed (or negatively skewed). The
  Normal distribution is retrieved when `skewness` is equal to one.

  This distribution is also called the Fernández-Steel Skew Normal distribution
  [(Castillo et al., 2011)][1], the Skew Normal Type 2 distribution [(Rigby et
  al., 2019, Section 18.3.5, p380)][3], and the [Split Normal distribution][4].
  The Fernández and Steel's formulation is mathematically equivalent to the main
  parameterization discussed in the last reference.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale, skewness) =
      k * normal_pdf(y * skewness; 0, 1) when x < loc, and
      k * normal_pdf(y / skewness; 0, 1) when x >= loc
  where
      k = (2 * skewness) / ((1 + skewness**2) * scale)
      y = (x - loc) / scale
  ```

  where `loc` is the location, `scale` is the scale, `skewness` is the shape
  parameter, and `normal_pdf(x; 0, 1)` is the pdf of the Normal distribution
  with zero mean and unit variance.

  The cumulative distribution function (cdf) is,

  ```none
  cdf(x; loc, scale, skewness) =
      k0 * normal_cdf(y * skewness; 0, 1) when x < loc, and
      k1 + k2 * normal_cdf(y / skewness; 0, 1) when x >= loc
  where
      k0 = 2 / (1 + skewness**2)
      k1 = (1 - skewness**2) / (1 + skewness**2)
      k2 = (2 * skewness**2) / (1 + skewness**2)
      y = (x - loc) / scale
  ```

  where `normal_cdf(x; 0, 1)` is the cdf of the Normal distribution with zero
  mean and unit variance.

  The quantile function (inverse cdf) is,

  ```none
  quantile(p; loc, scale, skewness) =
      loc + s0 * normal_quantile(x0) when p <= 1 / (1 + skewness**2), and
      loc + s1 * normal_quantile(x1) when p > 1 / (1 + skewness**2)
  where
      s0 = scale / skewness
      s1 = scale * skewness
      x0 = (p * (1 + skewness**2)) / 2
      x1 = (p * (1 + skewness**2) - 1 + skewness**2) / (2 * skewness**2)
      y = (x - loc) / scale
  ```

  where `normal_quantile(x; 0, 1)` is the quantile function of the Normal
  distribution with zero mean and unit variance.

  The mean and variance are, respectively,

  ```none
  mean(loc, scale, skewness) = loc + scale * E(Y)
  variance(loc, scale, skewness) = scale**2 * (
      skewness**2 + 1 / skewness**2 - 1 - E(Y)**2)
  where
      E(Y) = sqrt(2) / sqrt(pi) * (skewness - 1 / skewness)
  ```

  The Two-Piece Normal distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family): it can be constructed
  as,

  ```none
  Z ~ Normal(loc=0, scale=1)
  W ~ Bernoulli(probs=1 / (1 + skewness**2))
  Y = (1 - W) * |Z| * skewness - W * |Z| / skewness
  X = loc + scale * Y
  ```

  #### Examples

  Example of initialization of one distribution.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Two-Piece Normal distribution.
  dist = tfd.TwoPieceNormal(loc=3., scale=10., skewness=0.75)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)
  ```

  Example of initialization of a batch of distributions. Arguments are
  broadcast when possible.

  ```python
  # Define a batch of three scalar valued Two-Piece Normals.
  # They have mean 3, scale 10, but different skewnesses.
  dist = tfd.TwoPieceNormal(loc=3., scale=10., skewness=[0.75, 1., 1.33])

  # Get 2 samples, returning a 2 x 3 tensor.
  value = dist.sample(2)

  # Evaluate the pdf of the distributions on the same points, value,
  # returning a 2 x 3 tensor.
  dist.prob(value)
  ```

  #### References

  [1]: Nabor O. Castillo et al. On the Fernández-Steel distribution: Inference
       and application. _Computational Statistics & Data Analysis_, 55(11),
       2951-2961, 2011.

  [2]: Carmen Fernández and Mark F. J. Steel. On Bayesian modeling of fat tails
       and skewness. _Journal of the American Statistical Association_, 93(441),
       359-371, 1998.

  [3]: Robert A. Rigby et al. _Distributions for modeling location, scale, and
       shape: Using GAMLSS in R_. Chapman and Hall/CRC, 2019.

  [4]: https://en.wikipedia.org/wiki/Split_normal_distribution

  """

  def __init__(self,
               loc,
               scale,
               skewness,
               validate_args=False,
               allow_nan_stats=True,
               name='TwoPieceNormal'):
    """Construct Two-Piece Normal distributions.

    The Two-Piece Normal is parametrized with location `loc`, scale `scale`,
    and shape parameter `skewness`. The parameters must be shaped in a way that
    supports broadcasting (e.g. `loc + scale` is a valid operation).

    Args:
      loc: Floating point tensor; the location(s) of the distribution(s).
      scale: Floating point tensor; the scale(s) of the distribution(s). Must
        contain only positive values.
      skewness: Floating point tensor; the skewness(es) of the distribution(s).
        Must contain only positive values.
      validate_args: Python `bool`, default `False`. When `True`, distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False`, invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if `loc`, `scale`, and `skewness` have different `dtype`.
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
      super().__init__(
          dtype=dtype,
          # skewness contributes to a discrete choice. The other two variables
          # are fine.
          reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        skewness=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for the scale."""
    return self._scale

  @property
  def skewness(self):
    """Distribution parameter for the skewness."""
    return self._skewness

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    batch_shape = self._batch_shape_tensor(
        loc=loc, scale=scale, skewness=skewness)
    sample_shape = ps.concat([[n], batch_shape], axis=0)

    uniform_seed, normal_seed = samplers.split_seed(
        seed, salt='two_piece_normal')
    uniform_sample = samplers.uniform(
        sample_shape, maxval=1., dtype=self.dtype, seed=uniform_seed)
    normal_sample = samplers.normal(
        sample_shape, dtype=self.dtype, seed=normal_seed)

    sample = tf.abs(normal_sample) * tf.where(
        uniform_sample < tf.math.reciprocal(1. + skewness**2),
        -tf.math.reciprocal(skewness),
        skewness)

    return loc + scale * sample

  def _log_prob(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    half = tf.constant(0.5, dtype=self.dtype)
    two = tf.constant(2., dtype=self.dtype)
    pi = tf.constant(np.pi, dtype=self.dtype)

    z = standardize(value, loc=loc, scale=scale, skewness=skewness)

    log_unnormalized = -half * tf.math.square(z)
    log_normalization = (
        _numpy_cast(log1psquare(skewness), loc.dtype) -
        tf.math.log(two * skewness) +
        tf.math.log(scale) +
        half * tf.math.log(two * pi))

    return log_unnormalized - log_normalization

  def _cdf(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    return cdf(value, loc=loc, scale=scale, skewness=skewness)

  def _survival_function(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    # Here we use the following property of this distribution:
    # sf = 1. - cdf(value; loc, scale, skewness)
    #    = cdf(-value; -loc, scale, 1. / skewness)
    return cdf(-value, loc=-loc, scale=scale,
               skewness=tf.math.reciprocal(skewness))

  def _quantile(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    return quantile(value, loc=loc, scale=scale, skewness=skewness)

  def _mean(self):
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    two = tf.constant(2., dtype=self.dtype)
    pi = tf.constant(np.pi, dtype=self.dtype)

    m = tf.math.sqrt(two / pi) * (skewness - tf.math.reciprocal(skewness))
    mean = loc + scale * m

    batch_shape = self._batch_shape_tensor(
        loc=loc, scale=scale, skewness=skewness)

    return tf.broadcast_to(mean, shape=batch_shape)

  def _variance(self):
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    one = tf.constant(1., dtype=self.dtype)
    two = tf.constant(2., dtype=self.dtype)
    pi = tf.constant(np.pi, dtype=self.dtype)

    m = tf.math.sqrt(two / pi) * (skewness - tf.math.reciprocal(skewness))
    squared_skewness = tf.math.square(skewness)
    v = (squared_skewness + tf.math.reciprocal(squared_skewness) - one -
         tf.math.square(m))
    variance = tf.square(scale) * v

    batch_shape = self._batch_shape_tensor(scale=scale, skewness=skewness)

    return tf.broadcast_to(variance, shape=batch_shape)

  def _mode(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, shape=self._batch_shape_tensor(loc=loc))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if is_init:
      # _batch_shape() will raise error if it can statically prove that `loc`,
      # `scale`, and `skewness` have incompatible shapes.
      try:
        self._batch_shape()
      except ValueError as e:
        raise ValueError('Arguments `loc`, `scale` and `skewness` '
                         'must have compatible shapes; '
                         f'loc.shape={self.loc.shape}, '
                         f'scale.shape={self.scale.shape}, '
                         f'skewness.shape={self.skewness.shape}.') from e
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access the three arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(
          assert_util.assert_positive(
              self.scale, message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.skewness):
      assertions.append(
          assert_util.assert_positive(
              self.skewness, message='Argument `skewness` must be positive.'))

    return assertions
