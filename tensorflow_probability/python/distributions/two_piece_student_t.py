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
"""The Two-Piece Student's t-distribution class."""

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.distributions import two_piece_normal
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import special
from tensorflow_probability.python.math.numeric import log1psquare

__all__ = [
    'TwoPieceStudentT',
]


class TwoPieceStudentT(distribution.AutoCompositeTensorDistribution):
  """The Two-Piece Student's t-distribution.

  This distribution generalizes the Student's t-distribution with an additional
  shape parameter. Under the general formulation proposed by [Fernández and
  Steel (1998)][1], it is parameterized by degree of freedom `df`, location
  `loc`, scale `scale`, and skewness `skewness`. If `skewness` is above one,
  the distribution becomes right-skewed (or positively skewed). If `skewness`
  is greater than zero and less than one, the distribution becomes left-skewed
  (or negatively skewed). The Student's t-distribution is retrieved when
  `skewness` is equal to one.

  This distribution is also called the Skewed Student distribution [Fernández
  and Steel (1998)][1] and the Skew T Type 3 (ST3) distribution [(Rigby et al.,
  2019, Section 18.4.14, p414)][2].

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, loc, scale, skewness) =
      k * t_pdf(y * skewness; df, 0, 1), when x < loc
      k * t_pdf(y / skewness; df, 0, 1), when x >= loc
  where
      k = (2 * skewness) / ((1 + skewness**2) * scale)
      y = (x - loc) / scale
  ```

  where `df` is the degree of freedom, `loc` is the location, `scale` is the
  scale, `skewness` is the shape parameter, and `t_pdf(x; df, 0, 1)` is the pdf
  of the Student's t-distribution with `df` degree of freedom, zero mean, and
  unit scale.

  The cumulative distribution function (cdf) is,

  ```none
  cdf(x; df, loc, scale, skewness) =
      k0 * t_cdf(y * skewness; df, 0, 1), when x < loc
      k1 + k2 * (t_cdf(y / skewness; df, 0, 1) - 0.5), when x >= loc
  where
      k0 = 2 / (1 + skewness**2)
      k1 = 1 / (1 + skewness**2)
      k2 = (2 * skewness**2) / (1 + skewness**2)
      y = (x - loc) / scale
  ```

  where `t_cdf(x; df, 0, 1)` is the cdf of the Student's t-distribution with
  `df` degree of freedom, zero mean, and unit scale.

  The quantile function (inverse cdf) is,

  ```none
  quantile(p; loc, scale, skewness) =
      loc + s0 * t_quantile(x0; df, 0, 1), when p <= 1 / (1 + skewness**2)
      loc + s1 * t_quantile(x1; df, 0, 1), when p > 1 / (1 + skewness**2)
  where
      s0 = scale / skewness
      s1 = scale * skewness
      x0 = (p * (1 + skewness**2)) / 2
      x1 = (p * (1 + skewness**2) - 1 + skewness**2) / (2 * skewness**2)
      y = (x - loc) / scale
  ```

  where `t_quantile(x; df, 0, 1)` is the quantile function of the Student's
  t-distribution with `df` degree of freedom, zero mean, and unit scale.

  The mean and variance are, respectively,

  ```none
  mean(df, loc, scale, skewness) =
      loc + scale * m, when df > 1
      NaN, when df <= 1
  variance(df, loc, scale, skewness) =
      scale**2 * (r * (skewness**2 + 1 / skewness**2 - 1) - m**2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
  where
      r = df / (df - 2)
      m = (2 * sqrt(df) * (skewness - 1 / skewness)) / (
          (df - 1) * Beta(0.5, 0.5 * df))
  ```

  The Two-Piece Student's t-distribution is a member of the [location-scale
  family][https://en.wikipedia.org/wiki/Location-scale_family]: it can be
  constructed as,

  ```none
  Z ~ StudentT(df=df, loc=0, scale=1)
  W ~ Bernoulli(probs=1 / (1 + skewness**2))
  Y = (1 - W) * |Z| * skewness - W * |Z| / skewness
  X = loc + scale * Y
  ```

  #### Examples

  Example of initialization of one distribution.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Two-Piece Student's t-distribution.
  dist = tfd.TwoPieceStudentT(df=6., loc=3., scale=10., skewness=0.75)

  # Evaluate the cdf at 1, returning a scalar.
  dist.cdf(1.)
  ```

  Example of initialization of a batch of distributions. Arguments are
  broadcast when possible.

  ```python
  # Define a batch of three scalar valued Two-Piece Student's t-distribution.
  # They have degree of freedom 6, mean 3, scale 10, but different skewnesses.
  dist = tfd.TwoPieceStudentT(
      df=6., loc=3., scale=10., skewness=[0.75, 1., 1.33])

  # Get 2 samples, returning a 2 x 3 tensor.
  value = dist.sample(2)

  # Evaluate the pdf of the distributions on the same points, value,
  # returning a 2 x 3 tensor.
  dist.prob(value)
  ```

  #### References

  [1]: Carmen Fernández and Mark F. J. Steel. On Bayesian modeling of fat tails
       and skewness. _Journal of the American Statistical Association_, 93(441),
       359-371, 1998.

  [2]: Robert A. Rigby et al. _Distributions for modeling location, scale, and
       shape: Using GAMLSS in R_. Chapman and Hall/CRC, 2019.

  """

  def __init__(self,
               df,
               loc,
               scale,
               skewness,
               validate_args=False,
               allow_nan_stats=True,
               name='TwoPieceStudentT'):
    """Construct Two-Piece Student's t-distributions.

    The Two-Piece Student's t-distribution is parametrized with degree of
    freedom `df`, location `loc`, scale `scale`, and skewness `skewness`.
    The parameters must be shaped in a way that supports broadcasting (e.g.
    `df + loc + scale + skewness` is a valid operation).

    Args:
      df: Floating point tensor; the degree(s) of freedom of the
        distribution(s). Must contain only positive values.
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
      TypeError: if `df`, `loc`, `scale`, and `skewness` have different
        `dtype`.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [df, loc, scale, skewness], dtype_hint=tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, dtype=dtype, name='df')
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, dtype=dtype, name='scale')
      self._skewness = tensor_util.convert_nonref_to_tensor(
          skewness, dtype=dtype, name='skewness')
      super().__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        skewness=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))

  @property
  def df(self):
    """Distribution parameter for the the degree of freedom."""
    return self._df

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
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    batch_shape = self._batch_shape_tensor(
        df=df, loc=loc, scale=scale, skewness=skewness)
    sample_shape = ps.concat([[n], batch_shape], axis=0)

    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    half = numpy_dtype(0.5)
    broadcast_df = tf.broadcast_to(df, sample_shape)

    two_piece_normal_seed, gamma_seed = samplers.split_seed(
        seed, salt='two_piece_student_t_split')
    two_piece_normal_samples = two_piece_normal.random_two_piece_normal(
        sample_shape, skewness=skewness, seed=two_piece_normal_seed)
    gamma_samples = gamma.random_gamma(
        (), concentration=half * broadcast_df, rate=half, seed=gamma_seed)
    samples = two_piece_normal_samples * tf.math.rsqrt(
        gamma_samples / broadcast_df)

    return loc + scale * samples

  def _log_prob(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    return log_prob(value, df=df, loc=loc, scale=scale, skewness=skewness)

  def _cdf(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    return cdf(value, df=df, loc=loc, scale=scale, skewness=skewness)

  def _survival_function(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    # Here we use the following property of this distribution:
    # sf = 1. - cdf(value; df, loc, scale, skewness)
    #    = cdf(-value; df, -loc, scale, 1. / skewness)
    return cdf(-value, df=df, loc=-loc, scale=scale,
               skewness=tf.math.reciprocal(skewness))

  def _quantile(self, value):
    value = tf.convert_to_tensor(value, dtype_hint=self.dtype)
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    return quantile(value, df=df, loc=loc, scale=scale, skewness=skewness)

  @distribution_util.AppendDocstring("""
      The mean of Two-Piece Student's t-distribution equals

      ```
      loc + scale * m, when df > 1
      NaN, when df <= 1

      where
          m = (2 * sqrt(df) * (skewness - 1 / skewness)) / (
              (df - 1) * Beta(0.5, 0.5 * df))
      ```

      If `self.allow_nan_stats=False`, then an exception will be raised rather
      than returning `NaN`.
      """)
  def _mean(self):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    half = numpy_dtype(0.5)
    one = numpy_dtype(1.)
    two = numpy_dtype(2.)

    very_small_df = (df <= one)
    safe_df = tf.where(very_small_df, two, df)
    m = two * tf.math.sqrt(safe_df) * (
        skewness - tf.math.reciprocal(skewness)) / (
            (safe_df - one) * tf.math.exp(special.lbeta(half, half * safe_df)))
    mean = loc + scale * m

    if self.allow_nan_stats:
      return tf.where(very_small_df, numpy_dtype(np.nan), mean)
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_greater(
              df,
              numpy_dtype(1.),
              message='mean not defined for components of df <= 1'),
      ], mean)

  @distribution_util.AppendDocstring("""
      The variance of Two-Piece Student's t-distribution equals

      ```
      scale**2 * (r * (skewness**2 + 1 / skewness**2 - 1) - m**2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1

      where
          r = df / (df - 2)
          m = (2 * sqrt(df) * (skewness - 1 / skewness)) / (
              (df - 1) * Beta(0.5, 0.5 * df))
      ```

      If `self.allow_nan_stats=False`, then an exception will be raised rather
      than returning `NaN`.
      """)
  def _variance(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    skewness = tf.convert_to_tensor(self.skewness)

    numpy_dtype = dtype_util.as_numpy_dtype(self.dtype)
    half = numpy_dtype(0.5)
    one = numpy_dtype(1.)
    two = numpy_dtype(2.)

    batch_shape = self._batch_shape_tensor(
        df=df, scale=scale, skewness=skewness)
    df = tf.broadcast_to(df, shape=batch_shape)

    small_df = (df <= two)
    safe_df = tf.where(small_df, numpy_dtype(3.), df)
    squared_skewness = tf.math.square(skewness)

    m = two * tf.math.sqrt(safe_df) * (
        skewness - tf.math.reciprocal(skewness)) / (
            (safe_df - one) * tf.math.exp(special.lbeta(half, half * safe_df)))
    v = -tf.math.square(m) + safe_df / (safe_df - two) * (
        squared_skewness + tf.math.reciprocal(squared_skewness) - one)
    variance = tf.square(scale) * v

    # When 1 < df <= 2, variance is infinite.
    result_where_defined = tf.where(small_df, numpy_dtype(np.inf), variance)

    if self.allow_nan_stats:
      return tf.where(df <= one, numpy_dtype(np.nan), result_where_defined)
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_greater(
              df,
              numpy_dtype(1.),
              message='variance not defined for components of df <= 1'),
      ], result_where_defined)

  def _mode(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, shape=self._batch_shape_tensor(loc=loc))

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if is_init:
      # _batch_shape() will raise error if it can statically prove that `df`,
      # `loc`, `scale`, and `skewness` have incompatible shapes.
      try:
        self._batch_shape()
      except ValueError as e:
        raise ValueError('Arguments `df`, `loc`, `scale` and `skewness` '
                         'must have compatible shapes; '
                         f'df.shape={self.loc.shape}, '
                         f'loc.shape={self.loc.shape}, '
                         f'scale.shape={self.scale.shape}, '
                         f'skewness.shape={self.skewness.shape}.') from e
      # We don't bother checking the shapes in the dynamic case because
      # all member functions access the four arguments anyway.

    if not self.validate_args:
      assert not assertions  # Should never happen.
      return []

    if is_init != tensor_util.is_ref(self.df):
      assertions.append(
          assert_util.assert_positive(
              self.df, message='Argument `df` must be positive.'))
    if is_init != tensor_util.is_ref(self.scale):
      assertions.append(
          assert_util.assert_positive(
              self.scale, message='Argument `scale` must be positive.'))
    if is_init != tensor_util.is_ref(self.skewness):
      assertions.append(
          assert_util.assert_positive(
              self.skewness, message='Argument `skewness` must be positive.'))

    return assertions


def standardize(value, loc, scale, skewness):
  """Apply mean-variance-skewness standardization to input `value`.

  Note that `scale` and `skewness` can be negative.

  Args:
    value: Floating-point tensor; the value(s) to be standardized.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype(
      [value, loc, scale, skewness], tf.float32)

  value, loc, scale, skewness = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (value, loc, scale, skewness)]

  return (value - loc) / tf.math.abs(scale) * tf.math.abs(
      tf.where(value < loc, skewness, tf.math.reciprocal(skewness)))


def log_prob(value, df, loc, scale, skewness):
  """Compute the log probability of Two-Piece Student's t-distribution.

  Note that `scale` and `skewness` can be negative.

  Args:
    value: Floating-point tensor; where to compute the log probabilities.
    df: Floating-point tensor; the degree(s) of freedom of the
      distribution(s). Must contain only positive values.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype(
      [value, df, loc, scale, skewness], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  value, df, loc, scale, skewness = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (value, df, loc, scale, skewness)]

  half = numpy_dtype(0.5)

  z = standardize(value, loc=loc, scale=scale, skewness=skewness)

  log_unnormalized_prob = -half * (df + numpy_dtype(1.)) * log1psquare(
      z * tf.math.rsqrt(df))
  log_normalization = (
      tf.math.log(tf.abs(scale)) + log1psquare(skewness) -
      tf.math.log(numpy_dtype(2.) * tf.math.abs(skewness)) +
      half * tf.math.log(df) + special.lbeta(half, half * df))

  return log_unnormalized_prob - log_normalization


def cdf(value, df, loc, scale, skewness):
  """Compute the distribution function of Two-Piece Student's t-distribution.

  Note that `scale` and `skewness` can be negative.

  Args:
    value: Floating-point tensor; where to compute the cdf.
    df: Floating-point tensor; the degree(s) of freedom of the
      distribution(s). Must contain only positive values.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype(
      [value, df, loc, scale, skewness], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  value, df, loc, scale, skewness = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (value, df, loc, scale, skewness)]

  one = numpy_dtype(1.)
  two = numpy_dtype(2.)

  t = standardize(value, loc=loc, scale=scale, skewness=skewness)
  t_cdf = student_t.stdtr(df, t=t)

  squared_skewness = tf.math.square(skewness)
  return tf.math.reciprocal(one + squared_skewness) * tf.where(
      t < 0.,
      two * t_cdf,
      one - squared_skewness + two * squared_skewness * t_cdf)


def quantile(value, df, loc, scale, skewness):
  """Compute the quantile function of Two-Piece Student's t-distribution.

  Note that `scale` and `skewness` can be negative.

  Args:
    value: Floating-point tensor; where to compute the quantile function.
    df: Floating-point tensor; the degree(s) of freedom of the
      distribution(s). Must contain only positive values.
    loc: Floating-point tensor; the location(s) of the distribution(s).
    scale: Floating-point tensor; the scale(s) of the distribution(s).
    skewness: Floating-point tensor; the skewness(es) of the distribution(s).

  Returns:
    A tensor with shape broadcast according to the arguments.
  """
  dtype = dtype_util.common_dtype(
      [value, df, loc, scale, skewness], tf.float32)
  numpy_dtype = dtype_util.as_numpy_dtype(dtype)

  value, df, loc, scale, skewness = [
      tf.convert_to_tensor(param, dtype=dtype)
      for param in (value, df, loc, scale, skewness)]

  half = numpy_dtype(0.5)
  one = numpy_dtype(1.)

  use_symmetry = (value >= half)
  value = tf.where(use_symmetry, one - value, value)
  loc = tf.where(use_symmetry, -loc, loc)
  skewness = tf.where(use_symmetry, tf.math.reciprocal(skewness), skewness)

  squared_skewness = tf.math.square(skewness)
  one_plus_squared_skewness = one + squared_skewness
  value_is_small = (value <= tf.math.reciprocal(one_plus_squared_skewness))

  probs = half * value * one_plus_squared_skewness
  probs = tf.where(
      value_is_small, probs, (probs - half) / squared_skewness + half)
  t_quantile = student_t.stdtrit(df, p=probs)

  abs_skewness = tf.math.abs(skewness)
  adj_scale = tf.math.abs(scale) * tf.where(
      value_is_small, tf.math.reciprocal(abs_skewness), abs_skewness)

  result = loc + adj_scale * t_quantile

  return tf.where(use_symmetry, -result, result)
