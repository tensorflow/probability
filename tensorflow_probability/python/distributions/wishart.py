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
import tensorflow as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization

__all__ = [
    "Wishart",
]


class _WishartLinearOperator(distribution.Distribution):
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
               scale_operator,
               input_output_cholesky=False,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` tensor, the degrees of freedom of the
        distribution(s). `df` must be greater than or equal to `k`.
      scale_operator: `float` or `double` instance of `LinearOperator`.
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
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
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
    with tf.compat.v1.name_scope(name) as name:
      with tf.compat.v1.name_scope("init", values=[df, scale_operator]):
        if not scale_operator.dtype.is_floating:
          raise TypeError(
              "scale_operator.dtype=%s is not a floating-point type" %
              scale_operator.dtype)
        if not scale_operator.is_square:
          print(scale_operator.to_dense().eval())
          raise ValueError("scale_operator must be square.")

        self._scale_operator = scale_operator
        self._df = tf.convert_to_tensor(
            value=df, dtype=scale_operator.dtype, name="df")
        tf.debugging.assert_same_float_dtype([self._df, self._scale_operator])
        if tf.compat.dimension_value(self._scale_operator.shape[-1]) is None:
          self._dimension = tf.cast(
              self._scale_operator.domain_dimension_tensor(),
              dtype=self._scale_operator.dtype,
              name="dimension")
        else:
          self._dimension = tf.convert_to_tensor(
              value=tf.compat.dimension_value(self._scale_operator.shape[-1]),
              dtype=self._scale_operator.dtype,
              name="dimension")
        df_val = tf.get_static_value(self._df)
        dim_val = tf.get_static_value(self._dimension)
        if df_val is not None and dim_val is not None:
          df_val = np.asarray(df_val)
          if not df_val.shape:
            df_val = [df_val]
          if np.any(df_val < dim_val):
            raise ValueError(
                "Degrees of freedom (df = %s) cannot be less than "
                "dimension of scale matrix (scale.dimension = %s)"
                % (df_val, dim_val))
        elif validate_args:
          assertions = tf.compat.v1.assert_less_equal(
              self._dimension,
              self._df,
              message=("Degrees of freedom (df = %s) cannot be "
                       "less than dimension of scale matrix "
                       "(scale.dimension = %s)" % (self._dimension, self._df)))
          self._df = distribution_util.with_dependencies(
              [assertions], self._df)
    super(_WishartLinearOperator, self).__init__(
        dtype=self._scale_operator.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=(
            [self._df, self._dimension] + self._scale_operator.graph_parents),
        name=name)

  @property
  def df(self):
    """Wishart distribution degree(s) of freedom."""
    return self._df

  def _square_scale_operator(self):
    return self.scale_operator.matmul(
        self.scale_operator.to_dense(), adjoint_arg=True)

  def scale(self):
    """Wishart distribution scale matrix."""
    if self._input_output_cholesky:
      return self.scale_operator.to_dense()
    else:
      return self._square_scale_operator()

  @property
  def scale_operator(self):
    """Wishart distribution scale matrix as an Linear Operator."""
    return self._scale_operator

  @property
  def input_output_cholesky(self):
    """Boolean indicating if `Tensor` input/outputs are Cholesky factorized."""
    return self._input_output_cholesky

  @property
  def dimension(self):
    """Dimension of underlying vector space. The `p` in `R^(p*p)`."""
    return self._dimension

  def _event_shape_tensor(self):
    dimension = self.scale_operator.domain_dimension_tensor()
    return tf.stack([dimension, dimension])

  def _event_shape(self):
    dimension = self.scale_operator.domain_dimension
    return tf.TensorShape([dimension, dimension])

  def _batch_shape_tensor(self):
    return tf.broadcast_dynamic_shape(
        tf.shape(input=self.df), self.scale_operator.batch_shape_tensor())

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        self.df.shape, self.scale_operator.batch_shape)

  def _sample_n(self, n, seed):
    batch_shape = self.batch_shape_tensor()
    event_shape = self.event_shape_tensor()
    batch_ndims = tf.shape(input=batch_shape)[0]

    ndims = batch_ndims + 3  # sample_ndims=1, event_ndims=2
    shape = tf.concat([[n], batch_shape, event_shape], 0)
    stream = seed_stream.SeedStream(seed, salt="Wishart")

    # Complexity: O(nbk**2)
    x = tf.random.normal(
        shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=stream())

    # Complexity: O(nbk)
    # This parametrization is equivalent to Chi2, i.e.,
    # ChiSquared(k) == Gamma(alpha=k/2, beta=1/2)
    expanded_df = self.df * tf.ones(
        self.scale_operator.batch_shape_tensor(),
        dtype=self.df.dtype.base_dtype)

    g = tf.random.gamma(
        shape=[n],
        alpha=self._multi_gamma_sequence(0.5 * expanded_df, self.dimension),
        beta=0.5,
        dtype=self.dtype,
        seed=stream())

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
    x = self.scale_operator.matmul(x)

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

    batch_shape = self.batch_shape_tensor()
    event_shape = self.event_shape_tensor()
    x_ndims = tf.rank(input=x_sqrt)
    num_singleton_axes_to_prepend = (
        tf.maximum(tf.size(input=batch_shape) + 2, x_ndims) - x_ndims)
    x_with_prepended_singletons_shape = tf.concat([
        tf.ones([num_singleton_axes_to_prepend], dtype=tf.int32),
        tf.shape(input=x_sqrt)], 0)
    x_sqrt = tf.reshape(x_sqrt, x_with_prepended_singletons_shape)
    ndims = tf.rank(x_sqrt)
    # sample_ndims = ndims - batch_ndims - event_ndims
    sample_ndims = ndims - tf.size(input=batch_shape) - 2
    sample_shape = tf.shape(input=x_sqrt)[:sample_ndims]

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
    last_dimsize = (
        tf.cast(self.dimension, dtype=tf.int32) *
        tf.reduce_prod(
            input_tensor=x_with_prepended_singletons_shape[:sample_ndims]))
    shape = tf.concat([x_with_prepended_singletons_shape[sample_ndims:-2],
                       [tf.cast(self.dimension, dtype=tf.int32),
                        last_dimsize]], 0)
    scale_sqrt_inv_x_sqrt = tf.reshape(scale_sqrt_inv_x_sqrt, shape)

    # Complexity: O(nbM*k) where M is the complexity of the operator solving a
    # vector system. For LinearOperatorLowerTriangular, each solve is O(k**2) so
    # this step has complexity O(nbk^3).
    scale_sqrt_inv_x_sqrt = self.scale_operator.solve(
        scale_sqrt_inv_x_sqrt)

    # Undo make batch-op ready.
    # Complexity: O(nbk**2)
    shape = tf.concat([batch_shape, event_shape, sample_shape], 0)
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
        input_tensor=tf.square(scale_sqrt_inv_x_sqrt), axis=[-2, -1])

    # Complexity: O(nbk)
    half_log_det_x = tf.reduce_sum(
        input_tensor=tf.math.log(tf.linalg.diag_part(x_sqrt)), axis=[-1])

    # Complexity: O(nbk**2)
    log_prob = ((self.df - self.dimension - 1.) * half_log_det_x -
                0.5 * trace_scale_inv_x -
                self.log_normalization())

    # Set shape hints.
    # Try to merge what we know from the input x with what we know from the
    # parameters of this distribution.
    if x.shape.ndims is not None and self.batch_shape.ndims is not None:
      log_prob.set_shape(
          tf.broadcast_static_shape(x.shape[:-2], self.batch_shape))

    return log_prob

  def _prob(self, x):
    return tf.exp(self._log_prob(x))

  def _entropy(self):
    half_dp1 = 0.5 * self.dimension + 0.5
    half_df = 0.5 * self.df
    return (self.dimension * (half_df + half_dp1 * math.log(2.)) +
            2 * half_dp1 * self.scale_operator.log_abs_determinant() +
            self._multi_lgamma(half_df, self.dimension) +
            (half_dp1 - half_df) * self._multi_digamma(half_df, self.dimension))

  def _mean(self):
    # Because df is a scalar, we need to expand dimensions to match
    # scale_operator. We use ellipses notation (...) to select all dimensions
    # and add two dimensions to the end.
    df = self.df[..., tf.newaxis, tf.newaxis]
    if self.input_output_cholesky:
      return tf.sqrt(df) * self.scale_operator.to_dense()
    return df * self._square_scale_operator()

  def _variance(self):
    # Because df is a scalar, we need to expand dimensions to match
    # scale_operator. We use ellipses notation (...) to select all dimensions
    # and add two dimensions to the end.
    df = self.df[..., tf.newaxis, tf.newaxis]
    x = tf.sqrt(df) * self._square_scale_operator()
    d = tf.expand_dims(tf.linalg.diag_part(x), -1)
    v = tf.square(x) + tf.matmul(d, d, adjoint_b=True)
    return v

  def _mode(self):
    s = self.df - self.dimension - 1.
    s = tf.where(
        tf.less(s, 0.), tf.constant(float("NaN"), dtype=self.dtype, name="nan"),
        s)
    if self.input_output_cholesky:
      return tf.sqrt(s) * self.scale_operator.to_dense()
    return s * self._square_scale_operator()

  def mean_log_det(self, name="mean_log_det"):
    """Computes E[log(det(X))] under this Wishart distribution."""
    with self._name_scope(name):
      return (self._multi_digamma(0.5 * self.df, self.dimension) +
              self.dimension * math.log(2.) +
              2 * self.scale_operator.log_abs_determinant())

  def log_normalization(self, name="log_normalization"):
    """Computes the log normalizing constant, log(Z)."""
    with self._name_scope(name):
      return (self.df * self.scale_operator.log_abs_determinant() +
              0.5 * self.df * self.dimension * math.log(2.) +
              self._multi_lgamma(0.5 * self.df, self.dimension))

  def _multi_gamma_sequence(self, a, p, name="multi_gamma_sequence"):
    """Creates sequence used in multivariate (di)gamma; shape = shape(a)+[p]."""
    with self._name_scope(name, values=[a, p]):
      # Linspace only takes scalars, so we'll add in the offset afterwards.
      seq = tf.linspace(
          tf.constant(0., dtype=self.dtype), 0.5 - 0.5 * p, tf.cast(
              p, tf.int32))
      return seq + tf.expand_dims(a, [-1])

  def _multi_lgamma(self, a, p, name="multi_lgamma"):
    """Computes the log multivariate gamma function; log(Gamma_p(a))."""
    with self._name_scope(name, values=[a, p]):
      seq = self._multi_gamma_sequence(a, p)
      return (0.25 * p * (p - 1.) * math.log(math.pi) +
              tf.reduce_sum(input_tensor=tf.math.lgamma(seq), axis=[-1]))

  def _multi_digamma(self, a, p, name="multi_digamma"):
    """Computes the multivariate digamma function; Psi_p(a)."""
    with self._name_scope(name, values=[a, p]):
      seq = self._multi_gamma_sequence(a, p)
      return tf.reduce_sum(input_tensor=tf.math.digamma(seq), axis=[-1])


class Wishart(_WishartLinearOperator):
  """The matrix Wishart distribution on positive definite matrices.

  This distribution is defined by a scalar degrees of freedom `df` and a scale
  matrix, which can either be a full symmetric matrix or a lower triangular
  Cholesky factor.

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

  ```python
  # Initialize a single 3x3 Wishart with Cholesky factored scale matrix and 5
  # degrees-of-freedom.(*)
  df = 5
  chol_scale = tf.cholesky(...)  # Shape is [3, 3].
  dist = tfd.Wishart(df=df, scale_tril=chol_scale)

  # Evaluate this on an observation in R^3, returning a scalar.
  x = ...  # A 3x3 positive definite matrix.
  dist.prob(x)  # Shape is [], a scalar.

  # Evaluate this on a two observations, each in R^{3x3}, returning a length two
  # Tensor.
  x = [x0, x1]  # Shape is [2, 3, 3].
  dist.prob(x)  # Shape is [2].

  # Initialize two 3x3 Wisharts with full scale matrices.
  df = [5, 4]
  scale = ...  # Shape is [2, 3, 3].
  dist = tfd.Wishart(df=df, scale=scale)

  # Evaluate this on four observations.
  x = [[x0, x1], [x2, x3]]  # Shape is [2, 2, 3, 3].
  dist.prob(x)  # Shape is [2, 2].

  # (*) - To efficiently create a trainable covariance matrix, see the example
  #   in tfp.distributions.matrix_diag_transform.
  ```
  """

  def __init__(self,
               df,
               scale=None,
               scale_tril=None,
               input_output_cholesky=False,
               validate_args=False,
               allow_nan_stats=True,
               name="Wishart"):
    """Construct Wishart distributions.

    Args:
      df: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
        or equal to dimension of the scale matrix.
      scale: `float` or `double` `Tensor`. The symmetric positive definite
        scale matrix of the distribution. Exactly one of `scale` and
        'scale_tril` must be passed.
      scale_tril: `float` or `double` `Tensor`. The Cholesky factorization
        of the symmetric positive definite scale matrix of the distribution.
        Exactly one of `scale` and 'scale_tril` must be passed.
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
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      ValueError: if zero or both of 'scale' and 'scale_tril' are passed in.
    """
    parameters = dict(locals())

    with tf.compat.v1.name_scope(name) as name:
      with tf.compat.v1.name_scope("init", values=[df, scale, scale_tril]):
        if (scale is None) == (scale_tril is None):
          raise ValueError("Must pass scale or scale_tril, but not both.")

        dtype = dtype_util.common_dtype([df, scale, scale_tril], tf.float32)
        df = tf.convert_to_tensor(value=df, name="df", dtype=dtype)
        if scale is not None:
          scale = tf.convert_to_tensor(value=scale, name="scale", dtype=dtype)
          if validate_args:
            scale = distribution_util.assert_symmetric(scale)
          scale_tril = tf.linalg.cholesky(scale)
        else:  # scale_tril is not None
          scale_tril = tf.convert_to_tensor(
              value=scale_tril, name="scale_tril", dtype=dtype)
          if validate_args:
            scale_tril = distribution_util.with_dependencies([
                tf.compat.v1.assert_positive(
                    tf.linalg.diag_part(scale_tril),
                    message="scale_tril must be positive definite"),
                tf.compat.v1.assert_equal(
                    tf.shape(input=scale_tril)[-1],
                    tf.shape(input=scale_tril)[-2],
                    message="scale_tril must be square")
            ], scale_tril)

      super(Wishart, self).__init__(
          df=df,
          scale_operator=tf.linalg.LinearOperatorLowerTriangular(
              tril=scale_tril,
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
    return dict(df=0, scale=2, scale_tril=2)
