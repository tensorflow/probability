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
"""The Wishart distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import cholesky_outer_product as cholesky_outer_product_bijector
from tensorflow_probability.python.bijectors import fill_scale_tril as fill_scale_tril_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.bijectors import transform_diagonal as transform_diagonal_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'WishartLinearOperator',
    'WishartTriL',
]


class WishartLinearOperator(distribution.Distribution):
  """The matrix Wishart distribution on positive definite matrices.

  This distribution is defined by a scalar number of degrees of freedom `df` and
  an instance of `LinearOperator`, which provides matrix-free access to a
  symmetric positive definite operator, which defines the scale matrix.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
  Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
  ```

  where:

  * `df >= k` denotes the degrees of freedom,
  * `scale` is a symmetric, positive definite, `k x k` matrix,
  * `Z` is the normalizing constant, and,
  * `Gamma_k` is the [multivariate Gamma function](
    https://en.wikipedia.org/wiki/Multivariate_gamma_function).

  #### Examples

  See the `Wishart` class for examples of initializing and using this class.
  """

  def __init__(self,
               df,
               scale,
               input_output_cholesky=False,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` tensor, the degrees of freedom of the
        distribution(s). `df` must be greater than or equal to `k`.
      scale: `float` or `double` instance of `LinearOperator`.
      input_output_cholesky: Python `bool`. If `True`, functions whose input or
        output have the semantics of samples assume inputs are in Cholesky form
        and return outputs in Cholesky form. In particular, if this flag is
        `True`, input to `log_prob` is presumed of Cholesky form and output from
        `sample`, `mean`, and `mode` are of Cholesky form.  Setting this
        argument to `True` is purely a computational optimization and does not
        change the underlying distribution; for instance, `mean` returns the
        Cholesky of the mean, not the mean of Cholesky factors. The `variance`
        and `stddev` methods are unaffected by this flag.
        Default value: `False` (i.e., input/output does not have Cholesky
        semantics).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      TypeError: if scale is not floating-type
      TypeError: if scale.dtype != df.dtype
      ValueError: if df < k, where scale operator event shape is
        `(k, k)`
    """
    parameters = dict(locals())
    self._input_output_cholesky = input_output_cholesky
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([scale, df], dtype_hint=tf.float32)
      self._scale = scale
      self._df = tensor_util.convert_nonref_to_tensor(
          df, name='df', dtype=dtype)

      super(WishartLinearOperator, self).__init__(
          dtype=dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @property
  def df(self):
    """Wishart distribution degree(s) of freedom."""
    return self._df

  def _square_scale(self):
    scale = self._scale
    return scale.matmul(scale, adjoint_arg=True).to_dense()

  def scale_matrix(self):
    """Wishart distribution scale matrix."""
    if self.input_output_cholesky:
      return self._scale.to_dense()
    else:
      return self._square_scale()

  @property
  def scale(self):
    """Wishart distribution scale matrix as an Linear Operator."""
    return self._scale

  @property
  def input_output_cholesky(self):
    """Boolean indicating if `Tensor` input/outputs are Cholesky factorized."""
    return self._input_output_cholesky

  def _event_shape_tensor(self):
    dimension = self._scale.domain_dimension_tensor()
    return tf.stack([dimension, dimension])

  def _event_shape(self):
    dimension = self._scale.domain_dimension
    return tf.TensorShape([dimension, dimension])

  def _batch_shape_tensor(self, df=None):
    df = tf.convert_to_tensor(self.df) if df is None else df
    return tf.broadcast_dynamic_shape(
        tf.shape(df), self._scale.batch_shape_tensor())

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.df.shape, self._scale.batch_shape)

  def _sample_n(self, n, seed):
    df = tf.convert_to_tensor(self.df)
    batch_shape = self._batch_shape_tensor(df)
    event_shape = self._event_shape_tensor()
    batch_ndims = tf.shape(batch_shape)[0]

    ndims = batch_ndims + 3  # sample_ndims=1, event_ndims=2
    shape = tf.concat([[n], batch_shape, event_shape], 0)
    normal_seed, gamma_seed = samplers.split_seed(seed, salt='Wishart')

    # Complexity: O(nbk**2)
    x = samplers.normal(
        shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=normal_seed)

    # Complexity: O(nbk)
    # This parameterization is equivalent to Chi2, i.e.,
    # ChiSquared(k) == Gamma(alpha=k/2, beta=1/2)
    expanded_df = df * tf.ones(
        self._scale.batch_shape_tensor(),
        dtype=dtype_util.base_dtype(df.dtype))

    g = samplers.gamma(
        shape=[n],
        alpha=self._multi_gamma_sequence(0.5 * expanded_df, self._dimension()),
        beta=0.5,
        dtype=self.dtype,
        seed=gamma_seed)

    # Complexity: O(nbk**2)
    x = tf.linalg.band_part(x, -1, 0)  # Tri-lower.

    # Complexity: O(nbk)
    x = tf.linalg.set_diag(x, tf.sqrt(g))

    # Make batch-op ready.
    # Complexity: O(nbk**2)
    perm = tf.concat([tf.range(1, ndims), [0]], 0)
    x = tf.transpose(a=x, perm=perm)
    shape = tf.concat([batch_shape, [event_shape[0]], [event_shape[1] * n]], 0)
    x = tf.reshape(x, shape)

    # Complexity: O(nbM) where M is the complexity of the operator solving a
    # vector system. For LinearOperatorLowerTriangular, each matmul is O(k^3) so
    # this step has complexity O(nbk^3).
    x = self._scale.matmul(x)

    # Undo make batch-op ready.
    # Complexity: O(nbk**2)
    shape = tf.concat([batch_shape, event_shape, [n]], 0)
    x = tf.reshape(x, shape)
    perm = tf.concat([[ndims - 1], tf.range(0, ndims - 1)], 0)
    x = tf.transpose(a=x, perm=perm)

    if not self.input_output_cholesky:
      # Complexity: O(nbk**3)
      x = tf.matmul(x, x, adjoint_b=True)

    return x

  def _log_prob(self, x):
    if self.input_output_cholesky:
      x_sqrt = x
    else:
      # Complexity: O(nbk**3)
      x_sqrt = tf.linalg.cholesky(x)

    df = tf.convert_to_tensor(self.df)
    batch_shape = self._batch_shape_tensor(df)
    event_shape = self._event_shape_tensor()
    dimension = self._dimension()
    x_ndims = tf.rank(x_sqrt)
    num_singleton_axes_to_prepend = (
        tf.maximum(tf.size(batch_shape) + 2, x_ndims) - x_ndims)
    x_with_prepended_singletons_shape = tf.concat([
        tf.ones([num_singleton_axes_to_prepend], dtype=tf.int32),
        tf.shape(x_sqrt)
    ], 0)
    x_sqrt = tf.reshape(x_sqrt, x_with_prepended_singletons_shape)
    ndims = tf.rank(x_sqrt)
    # sample_ndims = ndims - batch_ndims - event_ndims
    sample_ndims = ndims - tf.size(batch_shape) - 2
    sample_shape = tf.shape(x_sqrt)[:sample_ndims]

    # We need to be able to pre-multiply each matrix by its corresponding
    # batch scale matrix. Since a Distribution Tensor supports multiple
    # samples per batch, this means we need to reshape the input matrix `x`
    # so that the first b dimensions are batch dimensions and the last two
    # are of shape [dimension, dimensions*number_of_samples]. Doing these
    # gymnastics allows us to do a batch_solve.
    #
    # After we're done with sqrt_solve (the batch operation) we need to undo
    # this reshaping so what we're left with is a Tensor partitionable by
    # sample, batch, event dimensions.

    # Complexity: O(nbk**2) since transpose must access every element.
    scale_sqrt_inv_x_sqrt = x_sqrt
    perm = tf.concat([tf.range(sample_ndims, ndims),
                      tf.range(0, sample_ndims)], 0)
    scale_sqrt_inv_x_sqrt = tf.transpose(a=scale_sqrt_inv_x_sqrt, perm=perm)
    last_dim_size = (
        tf.cast(dimension, dtype=tf.int32) *
        tf.reduce_prod(x_with_prepended_singletons_shape[:sample_ndims]))
    shape = tf.concat(
        [x_with_prepended_singletons_shape[sample_ndims:-2],
         [tf.cast(dimension, dtype=tf.int32), last_dim_size]],
        axis=0)
    scale_sqrt_inv_x_sqrt = tf.reshape(scale_sqrt_inv_x_sqrt, shape)

    # Complexity: O(nbM*k) where M is the complexity of the operator solving a
    # vector system. For LinearOperatorLowerTriangular, each solve is O(k**2) so
    # this step has complexity O(nbk^3).
    scale_sqrt_inv_x_sqrt = self._scale.solve(scale_sqrt_inv_x_sqrt)

    # Undo make batch-op ready.
    # Complexity: O(nbk**2)
    shape = tf.concat(
        [tf.shape(scale_sqrt_inv_x_sqrt)[:-2], event_shape, sample_shape],
        axis=0)
    scale_sqrt_inv_x_sqrt = tf.reshape(scale_sqrt_inv_x_sqrt, shape)
    perm = tf.concat([
        tf.range(ndims - sample_ndims, ndims),
        tf.range(0, ndims - sample_ndims)
    ], 0)
    scale_sqrt_inv_x_sqrt = tf.transpose(a=scale_sqrt_inv_x_sqrt, perm=perm)

    # Write V = SS', X = LL'. Then:
    # tr[inv(V) X] = tr[inv(S)' inv(S) L L']
    #              = tr[inv(S) L L' inv(S)']
    #              = tr[(inv(S) L) (inv(S) L)']
    #              = sum_{ik} (inv(S) L)_{ik}**2
    # The second equality follows from the cyclic permutation property.
    # Complexity: O(nbk**2)
    trace_scale_inv_x = tf.reduce_sum(
        tf.square(scale_sqrt_inv_x_sqrt), axis=[-2, -1])

    # Complexity: O(nbk)
    half_log_det_x = tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(x_sqrt)), axis=[-1])

    # Complexity: O(nbk**2)
    log_prob = ((df - dimension - 1.) * half_log_det_x -
                0.5 * trace_scale_inv_x -
                self._log_normalization(df=df, scale=self._scale))

    # Set shape hints.
    # Try to merge what we know from the input x with what we know from the
    # parameters of this distribution.
    if tensorshape_util.rank(x.shape) is not None and tensorshape_util.rank(
        self.batch_shape) is not None:
      tensorshape_util.set_shape(
          log_prob,
          tf.broadcast_static_shape(x.shape[:-2], self.batch_shape))

    return log_prob

  def _entropy(self):
    dimension = self._dimension()
    half_dp1 = 0.5 * dimension + 0.5
    df = tf.convert_to_tensor(self.df)
    half_df = 0.5 * df
    return (dimension * (half_df + half_dp1 * math.log(2.)) +
            2 * half_dp1 * self._scale.log_abs_determinant() +
            self._multi_lgamma(half_df, dimension) +
            (half_dp1 - half_df) * self._multi_digamma(half_df, dimension))

  def _mean(self):
    # Because df is a scalar, we need to expand dimensions to match
    # scale. We use ellipses notation (...) to select all dimensions
    # and add two dimensions to the end.
    df = tf.convert_to_tensor(self.df)
    df = df[..., tf.newaxis, tf.newaxis]
    if self.input_output_cholesky:
      return tf.sqrt(df) * self._scale.to_dense()
    return df * self._square_scale()

  def _variance(self):
    # Because df is a scalar, we need to expand dimensions to match
    # scale. We use ellipses notation (...) to select all dimensions
    # and add two dimensions to the end.
    df = tf.convert_to_tensor(self.df)
    df = df[..., tf.newaxis, tf.newaxis]
    x = self._scale.matmul(self._scale, adjoint_arg=True)
    d = x.diag_part()[..., tf.newaxis]
    v = df * (tf.square(x.to_dense()) + tf.matmul(d, d, adjoint_b=True))
    return v

  def _mode(self):
    df = tf.convert_to_tensor(self.df)
    df = df[..., tf.newaxis, tf.newaxis]
    s = df - self._dimension() - 1.
    s = tf.where(
        s < 0.,
        dtype_util.as_numpy_dtype(s.dtype)(np.nan), s)
    if self.input_output_cholesky:
      return tf.sqrt(s) * self._scale.to_dense()
    return s * self._square_scale()

  def mean_log_det(self, name='mean_log_det'):
    """Computes E[log(det(X))] under this Wishart distribution."""
    with self._name_and_control_scope(name):
      dimension = self._dimension()
      return (self._multi_digamma(0.5 * self.df, dimension) +
              dimension * math.log(2.) +
              2 * self._scale.log_abs_determinant())

  def _log_normalization(self, df=None, scale=None, name='log_normalization'):
    df = tf.convert_to_tensor(self.df) if df is None else df
    scale = self._scale if scale is None else scale
    dimension = self._dimension()
    return (df * scale.log_abs_determinant() +
            0.5 * df * dimension * math.log(2.) +
            self._multi_lgamma(0.5 * df, dimension))

  def log_normalization(self, df=None, name='log_normalization'):
    """Computes the log normalizing constant, log(Z)."""
    with self._name_and_control_scope(name):
      return self._log_normalization(df=df, name=name)

  def _multi_gamma_sequence(self, a, p, name='multi_gamma_sequence'):
    """Creates sequence used in multivariate (di)gamma; shape = shape(a)+[p]."""
    with tf.name_scope(name):
      # Linspace only takes scalars, so we'll add in the offset afterwards.
      seq = tf.linspace(
          tf.constant(0., dtype=self.dtype),
          0.5 - 0.5 * p, tf.cast(p, tf.int32))
      return seq + a[..., tf.newaxis]

  def _multi_lgamma(self, a, p, name='multi_lgamma'):
    """Computes the log multivariate gamma function; log(Gamma_p(a))."""
    with tf.name_scope(name):
      seq = self._multi_gamma_sequence(a, p)
      return (0.25 * p * (p - 1.) * math.log(math.pi) +
              tf.reduce_sum(tf.math.lgamma(seq), axis=[-1]))

  def _multi_digamma(self, a, p, name='multi_digamma'):
    """Computes the multivariate digamma function; Psi_p(a)."""
    with tf.name_scope(name):
      seq = self._multi_gamma_sequence(a, p)
      return tf.reduce_sum(tf.math.digamma(seq), axis=[-1])

  def _dimension(self):
    """Scalar dimension of underlying vector space."""
    with tf.name_scope('dimension'):
      if tf.compat.dimension_value(self._scale.shape[-1]) is None:
        return tf.cast(
            self._scale.domain_dimension_tensor(),
            dtype=self._scale.dtype,
            name='dimension')
      else:
        return tf.convert_to_tensor(
            tf.compat.dimension_value(self._scale.shape[-1]),
            dtype=self._scale.dtype,
            name='dimension')

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    tril_bijector = chain_bijector.Chain([
        transform_diagonal_bijector.TransformDiagonal(
            diag_bijector=softplus_bijector.Softplus(
                validate_args=self.validate_args),
            validate_args=self.validate_args),
        fill_scale_tril_bijector.FillScaleTriL(
            validate_args=self.validate_args)
    ], validate_args=self.validate_args)
    if self.input_output_cholesky:
      return tril_bijector
    return chain_bijector.Chain([
        cholesky_outer_product_bijector.CholeskyOuterProduct(
            validate_args=self.validate_args),
        tril_bijector
    ], validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    assertions = []
    if is_init:
      if not dtype_util.is_floating(self._scale.dtype):
        raise TypeError(
            'scale.dtype={} is not a floating-point type.'.format(
                self._scale.dtype))
      if not self._scale.is_square:
        raise ValueError('scale must be square.')
      dtype_util.assert_same_float_dtype([self._df, self._scale])

    df_val = tf.get_static_value(self._df)
    dim_val = tf.compat.dimension_value(self._scale.shape[-1])
    msg = ('Degrees of freedom (`df = {}`) cannot be less than dimension of '
           'scale matrix (`scale.dimension = {}`).')
    if is_init and df_val is not None and dim_val is not None:
      df_val = np.asarray(df_val)
      dim_val = np.asarray(dim_val)
      if not dim_val.shape:
        dim_val = dim_val[np.newaxis, ...]
      if not df_val.shape:
        df_val = df_val[np.newaxis, ...]
      if np.any(df_val < dim_val):
        raise ValueError(msg.format(df_val, dim_val))

    elif self.validate_args:
      if (is_init != tensor_util.is_ref(self._df) or
          is_init != tensor_util.is_ref(self._scale)):
        df = tf.convert_to_tensor(self._df)
        dimension = self._dimension()
        assertions.append(assert_util.assert_less_equal(
            dimension, df, message=(msg.format(df, dimension))))

    return assertions


class WishartTriL(WishartLinearOperator):
  """The matrix Wishart distribution parameterized with Cholesky factors.

  This distribution is defined by a scalar degrees of freedom `df` and a scale
  matrix, expressed as a lower triangular Cholesky factor.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(X; df, scale) = det(X)**(0.5 (df-k-1)) exp(-0.5 tr[inv(scale) X]) / Z
  Z = 2**(0.5 df k) |det(scale)|**(0.5 df) Gamma_k(0.5 df)
  ```

  where:
  * `df >= k` denotes the degrees of freedom,
  * `scale` is a symmetric, positive definite, `k x k` matrix equivalent to
    `scale_tril * scale_tril.T`,
  * `Z` is the normalizing constant, and,
  * `Gamma_k` is the [multivariate Gamma function](
    https://en.wikipedia.org/wiki/Multivariate_gamma_function).


  #### Examples

  ```python
  # Initialize a single 3x3 Wishart with Cholesky factored scale matrix and 5
  # degrees-of-freedom.(*)
  df = 5
  chol_scale = tf.linalg.cholesky(...)  # Shape is [3, 3].
  dist = tfd.WishartTriL(df=df, scale_tril=chol_scale)

  # Evaluate this on an observation in R^3, returning a scalar.
  x = ...  # A 3x3 positive definite matrix.
  dist.prob(x)  # Shape is [], a scalar.

  # Evaluate this on a two observations, each in R^{3x3}, returning a length two
  # Tensor.
  x = [x0, x1]  # Shape is [2, 3, 3].
  dist.prob(x)  # Shape is [2].

  # (*) - To efficiently create a trainable covariance matrix, see the example
  #   in tfp.distributions.matrix_diag_transform.
  ```
  """

  def __init__(self,
               df,
               scale_tril=None,
               input_output_cholesky=False,
               validate_args=False,
               allow_nan_stats=True,
               name='WishartTriL'):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
        or equal to dimension of the scale matrix.
      scale_tril: `float` or `double` `Tensor`. The Cholesky factorization
        of the symmetric positive definite scale matrix of the distribution.
      input_output_cholesky: Python `bool`. If `True`, functions whose input or
        output have the semantics of samples assume inputs are in Cholesky form
        and return outputs in Cholesky form. In particular, if this flag is
        `True`, input to `log_prob` is presumed of Cholesky form and output from
        `sample`, `mean`, and `mode` are of Cholesky form.  Setting this
        argument to `True` is purely a computational optimization and does not
        change the underlying distribution; for instance, `mean` returns the
        Cholesky of the mean, not the mean of Cholesky factors. The `variance`
        and `stddev` methods are unaffected by this flag.
        Default value: `False` (i.e., input/output does not have Cholesky
        semantics).
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([df, scale_tril], tf.float32)
      df = tensor_util.convert_nonref_to_tensor(df, name='df', dtype=dtype)
      self._scale_tril = tensor_util.convert_nonref_to_tensor(
          scale_tril, name='scale_tril', dtype=dtype)

      super(WishartTriL, self).__init__(
          df=df,
          scale=tf.linalg.LinearOperatorLowerTriangular(
              tril=self._scale_tril,
              is_non_singular=True,
              is_positive_definite=True,
              is_square=True),
          input_output_cholesky=input_output_cholesky,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
      self._parameters = parameters

  @classmethod
  def _params_event_ndims(cls):
    return dict(df=0, scale_tril=2)

  @property
  def scale_tril(self):
    """Cholesky decomposition of Wishart scale matrix."""
    return self._scale_tril

  def _parameter_control_dependencies(self, is_init):
    assertions = super(
        WishartTriL, self)._parameter_control_dependencies(is_init)

    if not self.validate_args:
      assert not assertions
      return []

    if is_init != tensor_util.is_ref(self._scale_tril):
      shape = prefer_static.shape(self._scale_tril)
      assertions.extend(
          [assert_util.assert_positive(
              tf.linalg.diag_part(self._scale_tril),
              message='`scale_tril` must be positive definite.'),
           assert_util.assert_equal(
               shape[-1],
               shape[-2],
               message='`scale_tril` must be square.')]
          )
    return assertions
