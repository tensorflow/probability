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
"""Multivariate Student's t-distribution."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import chi2 as chi2_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'MultivariateStudentTLinearOperator',
]


class MultivariateStudentTLinearOperator(distribution.Distribution):
  """The [Multivariate Student's t-distribution](

  https://en.wikipedia.org/wiki/Multivariate_t-distribution) on `R^k`.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, loc, Sigma) = (1 + ||y||**2 / df)**(-0.5 (df + k)) / Z
  where,
  y = inv(Sigma) (x - loc)
  Z = abs(det(Sigma)) sqrt(df pi)**k Gamma(0.5 df) / Gamma(0.5 (df + k))
  ```

  where:

  * `df` is a positive scalar.
  * `loc` is a vector in `R^k`,
  * `Sigma` is a positive definite `shape' matrix in `R^{k x k}`, parameterized
     as `scale @ scale.T` in this class,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.

  The Multivariate Student's t-distribution distribution is a member of the
  [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 3-variate Student's t.
  df = 3.
  loc = [1., 2, 3]
  scale = [[ 0.6,  0. ,  0. ],
           [ 0.2,  0.5,  0. ],
           [ 0.1, -0.3,  0.4]]
  sigma = tf.matmul(scale, scale, adjoint_b=True)
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  mvt = tfd.MultivariateStudentTLinearOperator(
      df=df,
      loc=loc,
      scale=tf.linalg.LinearOperatorLowerTriangular(scale))

  # Covariance is closely related to the sigma matrix (for df=3, it is 3x of the
  # sigma matrix).

  mvt.covariance()
  # ==> [[ 1.08,  0.36,  0.18],
  #      [ 0.36,  0.87, -0.39],
  #      [ 0.18, -0.39,  0.78]]

  # Compute the pdf of an`R^3` observation; return a scalar.
  mvt.prob([-1., 0, 1])  # shape: []

  """

  def __init__(self,
               df,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name='MultivariateStudentTLinearOperator'):
    """Construct Multivariate Student's t-distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `df`, `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` must broadcast with this.

    Additional leading dimensions (if any) will index batches.

    Args:
      df: A positive floating-point `Tensor`. Has shape `[B1, ..., Bb]` where `b
        >= 0`.
      loc: Floating-point `Tensor`. Has shape `[B1, ..., Bb, k]` where `k` is
        the event size.
      scale: Instance of `LinearOperator` with a floating `dtype` and shape
        `[B1, ..., Bb, k, k]`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/variance/etc...) is undefined for
        any batch member If `True`, batch members with valid parameters leading
        to undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if not `scale.dtype.is_floating`.
      ValueError: if not `scale.is_non_singular`.
    """
    parameters = dict(locals())
    if not dtype_util.is_floating(scale.dtype):
      raise TypeError('`scale` must have floating-point dtype.')
    if validate_args and not scale.is_non_singular:
      raise ValueError('`scale` must be non-singular.')

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, loc, scale], dtype_hint=tf.float32)
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)
      self._loc = tensor_util.convert_nonref_to_tensor(
          loc, name='loc', dtype=dtype)
      self._scale = scale

      super(MultivariateStudentTLinearOperator, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)
      self._parameters = parameters

  @property
  def loc(self):
    """The location parameter of the distribution.

    `loc` applies an elementwise shift to the distribution.

    ```none
    X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
    Y = scale @ X + loc
    ```

    Returns:
      The `loc` `Tensor`.
    """
    return self._loc

  @property
  def scale(self):
    """The scale parameter of the distribution.

    `scale` applies an affine scale to the distribution.

    ```none
    X ~ MultivariateT(loc=0, scale=1)   # Identity scale, zero shift.
    Y = scale @ X + loc
    ```

    Returns:
      The `scale` `LinearOperator`.
    """
    return self._scale

  @property
  def df(self):
    """The degrees of freedom of the distribution.

    This controls the degrees of freedom of the distribution. The tails of the
    distribution get more heavier the smaller `df` is. As `df` goes to
    infinitiy, the distribution approaches the Multivariate Normal with the same
    `loc` and `scale`.

    Returns:
      The `df` `Tensor`.
    """
    return self._df

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))
            ),
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        scale=parameter_properties.BatchedComponentProperties())

  def _event_shape_tensor(self):
    return self.scale.range_dimension_tensor()[tf.newaxis]

  def _event_shape(self):
    return self.scale.range_dimension

  def _sample_shape(self):
    return tf.concat([self.batch_shape_tensor(), self.event_shape_tensor()], -1)

  def _sample_n(self, n, seed=None):
    # Like with the univariate Student's t, sampling can be implemented as a
    # ratio of samples from a multivariate gaussian with the appropriate
    # covariance matrix and a sample from the chi-squared distribution.
    normal_seed, chi2_seed = samplers.split_seed(seed, salt='multivariate t')

    loc = tf.broadcast_to(self.loc, self._sample_shape())
    mvn = mvn_linear_operator.MultivariateNormalLinearOperator(
        loc=tf.zeros_like(loc), scale=self.scale)
    normal_samp = mvn.sample(n, seed=normal_seed)

    df = tf.broadcast_to(self.df, self.batch_shape_tensor())
    chi2 = chi2_lib.Chi2(df=df)
    chi2_samp = chi2.sample(n, seed=chi2_seed)

    return (self._loc +
            normal_samp * tf.math.rsqrt(chi2_samp / self._df)[..., tf.newaxis])

  def _log_normalization(self):
    df = tf.convert_to_tensor(self.df)
    num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)
    return (tfp_math.log_gamma_difference(num_dims / 2., df / 2.) +
            num_dims / 2. * (tf.math.log(df) + np.log(np.pi)) +
            self.scale.log_abs_determinant())

  def _log_unnormalized_prob(self, value):
    df = tf.convert_to_tensor(self.df)
    value = value - self._loc
    value = self.scale.solve(value[..., tf.newaxis])

    num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)
    mahalanobis = tf.norm(value, axis=[-1, -2])
    return -(num_dims + df) / 2. * tfp_math.log1psquare(
        mahalanobis / tf.sqrt(df))

  def _log_prob(self, value):
    return self._log_unnormalized_prob(value) - self._log_normalization()

  @distribution_util.AppendDocstring(
      """The mean of Student's T equals `loc` if `df > 1`, otherwise it is
      `NaN`. If `self.allow_nan_stats=False`, then an exception will be raised
      rather than returning `NaN`.""")
  def _mean(self):
    mean = tf.broadcast_to(self.loc, self._sample_shape())

    if self.allow_nan_stats:
      return tf.where(
          self.df[..., tf.newaxis] > 1., mean,
          dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      with tf.control_dependencies([
          assert_util.assert_less(
              tf.cast(1., self.dtype),
              self.df,
              message='Mean not defined for components of df <= 1.'),
      ]):
        return tf.identity(mean)

  def _mode(self):
    return tf.broadcast_to(self.loc, self._sample_shape())

  def _std_var_helper(self, statistic, statistic_name, statistic_ndims,
                      df_factor_fn):
    """Helper to compute stddev, covariance and variance."""
    df = tf.reshape(
        self.df,
        tf.concat(
            [tf.shape(self.df),
             tf.ones([statistic_ndims], dtype=tf.int32)], -1))
    # We need to put the tf.where inside the outer tf.where to ensure we never
    # hit a NaN in the gradient.
    denom = tf.where(df > 2., df - 2., dtype_util.as_numpy_dtype(df.dtype)(1.))
    statistic = statistic * df_factor_fn(df / denom)
    # When 1 < df <= 2, stddev/variance are infinite.
    result_where_defined = tf.where(
        df > 2., statistic, dtype_util.as_numpy_dtype(self.dtype)(np.inf))

    if self.allow_nan_stats:
      return tf.where(df > 1., result_where_defined,
                      dtype_util.as_numpy_dtype(self.dtype)(np.nan))
    else:
      with tf.control_dependencies([
          assert_util.assert_less(
              tf.cast(1., self.dtype),
              df,
              message='{} not defined for components of df <= 1.'.format(
                  statistic_name.capitalize())),
      ]):
        return tf.identity(result_where_defined)

  @distribution_util.AppendDocstring("""
      The covariance for Multivariate Student's t equals

      ```
      scale @ scale.T * df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```

      If `self.allow_nan_stats=False`, then an exception will be raised
      rather than returning `NaN`.""")
  def _covariance(self):
    if distribution_util.is_diagonal_scale(self.scale):
      mvn_cov = tf.linalg.diag(tf.square(self.scale.diag_part()))
    else:
      mvn_cov = self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)

    cov_shape = tf.concat(
        [self._sample_shape(), self._event_shape_tensor()], -1)
    mvn_cov = tf.broadcast_to(mvn_cov, cov_shape)
    return self._std_var_helper(mvn_cov, 'covariance', 2, lambda x: x)

  @distribution_util.AppendDocstring("""
      The variance for Student's T equals

      ```none
      diag(scale @ scale.T) * df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```

      If `self.allow_nan_stats=False`, then an exception will be raised
      rather than returning `NaN`.""")
  def _variance(self):
    if distribution_util.is_diagonal_scale(self.scale):
      mvn_var = tf.square(self.scale.diag_part())
    elif (isinstance(self.scale, tf.linalg.LinearOperatorLowRankUpdate) and
          self.scale.is_self_adjoint):
      mvn_var = tf.linalg.diag_part(self.scale.matmul(self.scale.to_dense()))
    else:
      mvn_var = tf.linalg.diag_part(
          self.scale.matmul(self.scale.to_dense(), adjoint_arg=True))

    mvn_var = tf.broadcast_to(mvn_var, self._sample_shape())
    return self._std_var_helper(mvn_var, 'variance', 1, lambda x: x)

  @distribution_util.AppendDocstring("""
      The standard deviation for Student's T equals

      ```none
      sqrt(diag(scale @ scale.T)) * df / (df - 2), when df > 2
      infinity, when 1 < df <= 2
      NaN, when df <= 1
      ```
      """)
  def _stddev(self):
    if distribution_util.is_diagonal_scale(self.scale):
      mvn_std = tf.abs(self.scale.diag_part())
    elif (isinstance(self.scale, tf.linalg.LinearOperatorLowRankUpdate) and
          self.scale.is_self_adjoint):
      mvn_std = tf.sqrt(
          tf.linalg.diag_part(self.scale.matmul(self.scale.to_dense())))
    else:
      mvn_std = tf.sqrt(
          tf.linalg.diag_part(
              self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)))

    mvn_std = tf.broadcast_to(mvn_std, self._sample_shape())
    return self._std_var_helper(mvn_std, 'standard deviation', 1, tf.sqrt)

  def _entropy(self):
    df = tf.broadcast_to(self.df, self.batch_shape_tensor())
    num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)

    shape_factor = self._scale.log_abs_determinant()
    beta_factor = (num_dims / 2. * (tf.math.log(df) + np.log(np.pi)) +
                   tfp_math.log_gamma_difference(num_dims / 2., df / 2.))
    digamma_factor = (num_dims + df) / 2. * (
        tf.math.digamma((num_dims + df) / 2.) - tf.math.digamma(df / 2.))
    return shape_factor + beta_factor + digamma_factor

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != tensor_util.is_ref(self._df):
      assertions.append(assert_util.assert_positive(
          self._df, message='Argument `df` must be positive.'))
    return assertions
