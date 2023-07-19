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
"""Matrix Normal distribution classes."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'MatrixNormalLinearOperator',
]


# Note the operations below are variants of the usual vec and unvec operations
# that avoid transposes.


def _vec(x):
  return tf.reshape(
      x, prefer_static.concat(
          [prefer_static.shape(x)[:-2], [-1]], axis=0))


def _unvec(x, matrix_shape):
  return tf.reshape(x, prefer_static.concat(
      [prefer_static.shape(x)[:-1], matrix_shape], axis=0))


class MatrixNormalLinearOperator(distribution.AutoCompositeTensorDistribution):
  """The Matrix Normal distribution on `n x p` matrices.

  The Matrix Normal distribution is defined over `n x p` matrices and
  parameterized by a (batch of) `n x p` `loc` matrices, a (batch of) `n x n`
  `scale_row` matrix and a (batch of) `p x p` `scale_column` matrix.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale_row, scale_column) =
     mvn_pdf(vec(x); vec(loc), scale_column (x) scale_row)
  ```

  where:

  * `loc` is a `n x p` matrix,
  * `scale_row` is a linear operator in `R^{n x n}`, such that the covariance
    between rows can be expressed as `row_cov = scale_row @ scale_row.T`,
  * `scale_column` is a linear operator in `R^{p x p}`, such that the covariance
    between columns can be expressed as
    `col_cov = scale_column @ scale_column.T`,
  * `mvn_pdf` is the Multivariate Normal probability density function.
  * `vec` is the operation that converts a matrix to a column vector (
    in numpy terms this is `X.T.flatten()`)
  * `(x)` is the Kronecker product.

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 2 x 3 Matrix Normal.
  mu = [[1., 2, 3], [3., 4, 5]]
  col_cov = [[ 0.36,  0.12,  0.06],
             [ 0.12,  0.29, -0.13],
             [ 0.06, -0.13,  0.26]]
  scale_column = tf.linalg.LinearOperatorTriL(tf.cholesky(col_cov))
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])
  scale_row = tf.linalg.LinearOperatorDiag([0.9, 0.8])

  mvn = tfd.MatrixNormalLinearOperator(
      loc=mu,
      scale_row=scale_row,
      scale_column=scale_column)

  # Initialize a 4-batch of 2 x 5-variate Matrix Normals.
  mu = tf.ones([2, 3, 5])
  scale_column_diag = [1., 2., 3., 4., 5.]
  scale_row_diag = [[0.3, 0.4, 0.6], [1., 2., 3.]]

  mvn = tfd.MatrixNormalLinearOperator(
      loc=mu,
      scale_row=tf.linalg.LinearOperatorDiag(scale_row_diag),
      scale_column=tf.linalg.LinearOperatorDiag(scale_column_diag))
  ```

  """

  def __init__(self,
               loc,
               scale_row,
               scale_column,
               validate_args=False,
               allow_nan_stats=True,
               name='MatrixNormalLinearOperator'):
    """Construct Matrix Normal distribution on `R^{n x p}`.

    The `batch_shape` is the broadcast shape between `loc`, `scale_row`
    and `scale_column` arguments.

    The `event_shape` is given by the matrix implied by `loc`.

    Args:
      loc: Floating-point `Tensor`, having shape `[B1, ..., Bb, n, p]`.
      scale_row: Instance of `LinearOperator` with the same `dtype` as `loc`
        and shape `[B1, ..., Bb, n, n]`.
      scale_column: Instance of `LinearOperator` with the same `dtype` as `loc`
        and shape `[B1, ..., Bb, p, p]`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """
    parameters = dict(locals())

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, scale_column, scale_row], dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
    self._loc = loc

    if not hasattr(scale_row, 'matmul'):
      raise ValueError('`scale_row` must be a `tf.linalg.LinearOperator`.')
    if not hasattr(scale_column, 'matmul'):
      raise ValueError('`scale_column` must be a `tf.linalg.LinearOperator`.')
    if validate_args and not scale_row.is_non_singular:
      raise ValueError('`scale_row` must be non-singular.')
    if validate_args and not scale_column.is_non_singular:
      raise ValueError('`scale_column` must be non-singular.')

    self._scale_row = scale_row
    self._scale_column = scale_column

    super(MatrixNormalLinearOperator, self).__init__(
        dtype=dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        name=name)
    self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=2),
        scale_row=parameter_properties.BatchedComponentProperties(),
        scale_column=parameter_properties.BatchedComponentProperties())

  def _as_multivariate_normal(self, loc=None):
    # Rebuild the Multivariate Normal Distribution on every call because the
    # underlying tensor shapes might have changed.
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    return mvn_linear_operator.MultivariateNormalLinearOperator(
        loc=_vec(loc),
        scale=tf.linalg.LinearOperatorKronecker(
            [self.scale_row, self.scale_column]),
        validate_args=self.validate_args)

  def _mean(self):
    shape = tf.concat([
        self.batch_shape_tensor(),
        self.event_shape_tensor(),
    ], 0)
    return tf.broadcast_to(self.loc, shape)

  def _variance(self):
    loc = tf.convert_to_tensor(self.loc)
    variance = self._as_multivariate_normal(loc=loc).variance()
    return _unvec(variance, self._event_shape_tensor(loc=loc))

  def _mode(self):
    return self._mean()

  def _log_prob(self, x):
    return self._as_multivariate_normal().log_prob(_vec(x))

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    samples = self._as_multivariate_normal(loc=loc).sample(n, seed=seed)
    return _unvec(samples, self._event_shape_tensor(loc=loc))

  def _sample_and_log_prob(self, sample_shape, seed):
    loc = tf.convert_to_tensor(self.loc)
    x, lp = self._as_multivariate_normal(
        loc=loc).experimental_sample_and_log_prob(
            sample_shape, seed=seed)
    return _unvec(x, self._event_shape_tensor(loc=loc)), lp

  def _entropy(self):
    return self._as_multivariate_normal().entropy()

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale_row(self):
    """Distribution parameter for row scale."""
    return self._scale_row

  @property
  def scale_column(self):
    """Distribution parameter for column scale."""
    return self._scale_column

  def _event_shape_tensor(self, loc=None):
    return tf.shape(self.loc if loc is None else loc)[-2:]

  def _event_shape(self):
    return self.loc.shape[-2:]

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    assertions = []
    if is_init != any(
        tensor_util.is_ref(v) for v in self.scale_column.variables):
      assertions.append(self.scale_column.assert_non_singular())
    if is_init != any(
        tensor_util.is_ref(v) for v in self.scale_row.variables):
      assertions.append(self.scale_column.assert_non_singular())
    return assertions


@kullback_leibler.RegisterKL(MatrixNormalLinearOperator,
                             MatrixNormalLinearOperator)
def _kl_matrix_normal_matrix_normal(a, b, name=None):
  """Batched KL divergence `KL(a || b)` for `MatrixNormalLinearOperator`-s.

  With `X`, `Y` both multivariate Normals in `R^k` with means `mu_a`, `mu_b` and
  covariance `C_a`, `C_b` respectively,

  ```
  KL(a || b) = 0.5 * ( L - k + T + Q ),
  L := Log[Det(C_b)] - Log[Det(C_a)]
  T := trace(C_b^{-1} C_a),
  Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
  ```

  By expanding in the case of the MatrixNormal, this can be optimized
  without explicitly constructing the two multivariate normal distributions.

  Args:
    a: Instance of `MatrixNormalLinearOperator`.
    b: Instance of `MatrixNormalLinearOperator`.
    name: (optional) name to use for created ops. Default "kl_mn".

  Returns:
    Batchwise `KL(a || b)`.
  """
  def squared_frobenius_norm(x):
    """Helper to make KL calculation slightly more readable."""
    return tf.reduce_sum(tf.square(x), axis=[-2, -1])

  def cholesky_solve(linop, rhs, adjoint_arg=False):
    """Like tf.linalg.cholesky_solve, but done via linop solve."""
    y = linop.solve(rhs, adjoint_arg=adjoint_arg)
    return linop.solve(y, adjoint=True)

  with tf.name_scope(name or 'kl_matrix_normal_matrix_normal'):
    # Calculation is based on:
    # http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    # and,
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # i.e.,
    #   If Ca = AA', Cb = BB', then
    #   tr[inv(Cb) Ca] = tr[inv(B)' inv(B) A A']
    #                  = tr[inv(B) A A' inv(B)']
    #                  = tr[(inv(B) A) (inv(B) A)']
    #                  = sum_{ij} (inv(B) A)_{ij}**2
    #                  = ||inv(B) A||_F**2
    # where ||.||_F is the Frobenius norm and the second equality follows from
    # the cyclic permutation property.
    k_inv_s_h = b.scale_row.solve(a.scale_row.to_dense())
    k_inv_s_x = b.scale_column.solve(a.scale_column.to_dense())

    mt = b.mean() - a.mean()

    n = tf.cast(b.scale_row.domain_dimension_tensor(), b.dtype)
    p = tf.cast(b.scale_column.domain_dimension_tensor(), b.dtype)

    kl_div = (
        p * (
            b.scale_row.log_abs_determinant()
            - a.scale_row.log_abs_determinant()
        )
        + n * (
            b.scale_column.log_abs_determinant()
            - a.scale_column.log_abs_determinant()
        )
        - 0.5 * n * p
        + 0.5 * (
            squared_frobenius_norm(k_inv_s_h)
            * squared_frobenius_norm(k_inv_s_x)
        )
        + 0.5 * tf.reduce_sum(
            cholesky_solve(
                b.scale_column, mt, adjoint_arg=True
            )
            * tf.linalg.matrix_transpose(
                cholesky_solve(b.scale_row, mt)
            ),
            [-1, -2],
        )
    )
    tensorshape_util.set_shape(
        kl_div, tf.broadcast_static_shape(a.batch_shape, b.batch_shape)
    )
    return kl_div
