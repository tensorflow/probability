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
"""Student's t distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import gamma as gamma_lib
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.numeric import log1psquare


__all__ = [
    'StudentT',
]


def sample_n(n, df, loc, scale, batch_shape, dtype, seed):
  """Draw n samples from a Student T distribution.

  Note that `scale` can be negative or zero.
  The sampling method comes from the fact that if:
    X ~ Normal(0, 1)
    Z ~ Chi2(df)
    Y = X / sqrt(Z / df)
  then:
    Y ~ StudentT(df)

  Args:
    n: int, number of samples
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s). Must
      contain only positive values.
    batch_shape: Callable to compute batch shape
    dtype: Return dtype.
    seed: Optional seed for random draw.

  Returns:
    samples: a `Tensor` with prepended dimensions `n`.
  """
  normal_seed, gamma_seed = samplers.split_seed(seed, salt='student_t')
  shape = ps.concat([[n], batch_shape], 0)

  normal_sample = samplers.normal(shape, dtype=dtype, seed=normal_seed)
  df = df * tf.ones(batch_shape, dtype=dtype)
  gamma_sample = gamma_lib.random_gamma(
      [n], concentration=0.5 * df, rate=0.5, seed=gamma_seed)
  samples = normal_sample * tf.math.rsqrt(gamma_sample / df)
  return samples * scale + loc


def log_prob(x, df, loc, scale):
  """Compute log probability of Student T distribution.

  Note that scale can be negative.

  Args:
    x: Floating-point `Tensor`. Where to compute the log probabilities.
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.
  """
  # Writing `y` this way reduces XLA mem copies.
  y = (x - loc) * (tf.math.rsqrt(df) / scale)
  log_unnormalized_prob = -0.5 * (df + 1.) * log1psquare(y)
  log_normalization = (
      tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(df) +
      0.5 * np.log(np.pi) + tfp_math.log_gamma_difference(0.5, 0.5 * df))
  return log_unnormalized_prob - log_normalization


def cdf(x, df, loc, scale):
  """Compute cumulative density function of Student T distribution.

  Note that scale can be negative.

  Args:
    x: Floating-point `Tensor`. Where to compute the log probabilities.
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    loc: Floating-point `Tensor`; the location(s) of the distribution(s).
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s).

  Returns:
    A `Tensor` with shape broadcast according to the arguments.
  """
  y = (x - loc) / tf.abs(scale)
  x_t = df / (y**2. + df)
  neg_cdf = 0.5 * tf.math.betainc(
      0.5 * tf.broadcast_to(df, ps.shape(x_t)), 0.5, x_t)
  return tf.where(y < 0., neg_cdf, 1. - neg_cdf)


def entropy(df, scale, batch_shape, dtype):
  """Compute entropy of the StudentT distribution.

  Args:
    df: Floating-point `Tensor`. The degrees of freedom of the
      distribution(s). `df` must contain only positive values.
    scale: Floating-point `Tensor`; the scale(s) of the distribution(s). Must
      contain only positive values.
    batch_shape: Floating-point `Tensor` of the batch shape
    dtype: Return dtype.

  Returns:
    A `Tensor` of the entropy for a Student's T with these parameters.
  """
  v = tf.ones(batch_shape, dtype=dtype)
  u = v * df
  return (tf.math.log(tf.abs(scale)) + 0.5 * tf.math.log(df) +
          tfp_math.lbeta(u / 2., v / 2.) + 0.5 * (df + 1.) *
          (tf.math.digamma(0.5 * (df + 1.)) - tf.math.digamma(0.5 * df)))


class StudentT(distribution.Distribution):
  """Student's t-distribution.

  This distribution has parameters: degree of freedom `df`, location `loc`,
  and `scale`.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, mu, sigma) = (1 + y**2 / df)**(-0.5 (df + 1)) / Z
  where,
  y = (x - mu) / sigma
  Z = abs(sigma) sqrt(df pi) Gamma(0.5 df) / Gamma(0.5 (df + 1))
  ```

  where:
  * `loc = mu`,
  * `scale = sigma`, and,
  * `Z` is the normalization constant, and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).

  The StudentT distribution is a member of the [location-scale family](
  https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ StudentT(df, loc=0, scale=1)
  Y = loc + scale * X
  ```

  Notice that `scale` has semantics more similar to standard deviation than
  variance. However it is not actually the std. deviation; the Student's
  t-distribution std. dev. is `scale sqrt(df / (df - 2))` when `df > 2`.

  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in the paper

  [Michael Figurnov, Shakir Mohamed, Andriy Mnih.
  Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

  #### Examples

  Examples of initialization of one or a batch of distributions.

  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  # Define a single scalar Student t distribution.
  single_dist = tfd.StudentT(df=3)

  # Evaluate the pdf at 1, returning a scalar Tensor.
  single_dist.prob(1.)

  # Define a batch of two scalar valued Student t's.
  # The first has degrees of freedom 2, mean 1, and scale 11.
  # The second 3, 2 and 22.
  multi_dist = tfd.StudentT(df=[2, 3], loc=[1, 2.], scale=[11, 22.])

  # Evaluate the pdf of the first distribution on 0, and the second on 1.5,
  # returning a length two tensor.
  multi_dist.prob([0, 1.5])

  # Get 3 samples, returning a 3 x 2 tensor.
  multi_dist.sample(3)
  ```

  Arguments are broadcast when possible.

  ```python
  # Define a batch of two Student's t distributions.
  # Both have df 2 and mean 1, but different scales.
  dist = tfd.StudentT(df=2, loc=1, scale=[11, 22.])

  # Evaluate the pdf of both distributions on the same point, 3.0,
  # returning a length 2 tensor.
  dist.prob(3.0)
  ```

  Compute the gradients of samples w.r.t. the parameters:

  ```python
  df = tf.constant(2.0)
  loc = tf.constant(2.0)
  scale = tf.constant(11.0)
  dist = tfd.StudentT(df=df, loc=loc, scale=scale)
  samples = dist.sample(5)  # Shape [5]
  loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
  # Unbiased stochastic gradients of the loss function
  grads = tf.gradients(loss, [df, loc, scale])
  ```

  """

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='StudentT'):
    """Construct Student's t distributions.

    The distributions have degree of freedom `df`, mean `loc`, and scale
    `scale`.

    The parameters `df`, `loc`, and `scale` must be shaped in a way that
    supports broadcasting (e.g. `df + loc + scale` is a valid operation).

    Args:
      df: Floating-point `Tensor`. The degrees of freedom of the
        distribution(s). `df` must contain only positive values.
      loc: Floating-point `Tensor`. The mean(s) of the distribution(s).
      scale: Floating-point `Tensor`. The scaling factor(s) for the
        distribution(s). Note that `scale` is not technically the standard
        deviation of this distribution but has semantics more similar to
        standard deviation than variance.
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
      TypeError: if loc and scale are different dtypes.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = tensor_util.convert_nonref_to_tensor(
          scale, name='scale', dtype=dtype)
      dtype_util.assert_same_float_dtype((self._df, self._loc, self._scale))
      super(StudentT, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(('df', 'loc', 'scale'),
            ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)] * 3)))

  @classmethod
  def _params_event_ndims(cls):
    return dict(df=0, loc=0, scale=0)

  @property
  def df(self):
    """Degrees of freedom in these Student's t distribution(s)."""
    return self._df

  @property
  def loc(self):
    """Locations of these Student's t distribution(s)."""
    return self._loc

  @property
  def scale(self):
    """Scaling factors of these Student's t distribution(s)."""
    return self._scale

  def _batch_shape_tensor(self, df=None, loc=None, scale=None):
    return ps.broadcast_shape(
        ps.shape(self.df if df is None else df),
        ps.broadcast_shape(
            ps.shape(self.loc if loc is None else loc),
            ps.shape(self.scale if scale is None else scale)))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tf.broadcast_static_shape(self.df.shape,
                                  self.loc.shape),
        self.scale.shape)

  def _event_shape_tensor(self):
    return tf.constant([], dtype=tf.int32)

  def _event_shape(self):
    return tf.TensorShape([])

  def _sample_n(self, n, seed=None):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, loc=loc, scale=scale)
    return sample_n(
        n,
        df=df,
        loc=loc,
        scale=scale,
        batch_shape=batch_shape,
        dtype=self.dtype,
        seed=seed)

  def _log_prob(self, x):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    scale = tf.convert_to_tensor(self.scale)
    return log_prob(x, df, loc, scale)

  def _cdf(self, x):
    df = tf.convert_to_tensor(self.df)
    return cdf(x, df, self.loc, self.scale)

  def _entropy(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    batch_shape = self._batch_shape_tensor(df=df, scale=scale)
    return entropy(df, scale, batch_shape, self.dtype)

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=True`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc)
    mean = loc * tf.ones(self._batch_shape_tensor(loc=loc),
                         dtype=self.dtype)
    if self.allow_nan_stats:
      return tf.where(
          df > 1.,
          mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='mean not defined for components of df <= 1'),
      ], mean)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```
      df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _variance(self):
    df = tf.convert_to_tensor(self.df)
    scale = tf.convert_to_tensor(self.scale)
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = tf.where(df > 2., df - 2., tf.ones_like(df))
    # Abs(scale) superfluous.
    var = (tf.ones(self._batch_shape_tensor(df=df, scale=scale),
                   dtype=self.dtype)
           * tf.square(scale) * df / denom)
    # When 1 < df <= 2, variance is infinite.
    result_where_defined = tf.where(
        df > 2.,
        var,
        dtype_util.as_numpy_dtype(self.dtype)(np.inf))

    if self.allow_nan_stats:
      return tf.where(
          df > 1.,
          result_where_defined,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      return distribution_util.with_dependencies([
          assert_util.assert_less(
              tf.ones([], dtype=self.dtype),
              df,
              message='variance not defined for components of df <= 1'),
      ], result_where_defined)

  def _mode(self):
    loc = tf.convert_to_tensor(self.loc)
    return tf.broadcast_to(loc, self._batch_shape_tensor(loc=loc))

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._df):
      assertions.append(assert_util.assert_positive(
          self._df, message='Argument `df` must be positive.'))
    return assertions

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector (consider one that
    # transforms away the heavy tails).
    return identity_bijector.Identity(validate_args=self.validate_args)
