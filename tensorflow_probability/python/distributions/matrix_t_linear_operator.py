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
"""Matrix T distribution classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import multivariate_student_t
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'MatrixTLinearOperator',
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


class MatrixTLinearOperator(distribution.AutoCompositeTensorDistribution):
  """The Matrix T distribution on `n x p` matrices.

  The Matrix T distribution is defined over `n x p` matrices and
  parameterized by a (batch of) `n x p` `loc` matrices, a (batch of) `n x n`
  `scale_row` matrix and a (batch of) `p x p` `scale_column` matrix.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; df, loc, scale_row, scale_column) =
     mvt_pdf(vec(x); df, vec(loc), scale_column (x) scale_row)
  ```

  where:

  * `df` is the degrees of freedom parameter.
  * `loc` is a `n x p` matrix,
  * `scale_row` is a linear operator in `R^{n x n}`, such that the covariance
    between rows can be expressed as `row_cov = scale_row @ scale_row.T`,
  * `scale_column` is a linear operator in `R^{p x p}`, such that the covariance
    between columns can be expressed as
    `col_cov = scale_column @ scale_column.T`,
  * `mvt_pdf` is the Multivariate T probability density function.
  * `vec` is the operation that converts a matrix to a column vector (
    in numpy terms this is `X.T.flatten()`)
  * `(x)` is the Kronecker product.

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 2 x 3 Matrix T.
  mu = [[1., 2, 3], [3., 4, 5]]
  col_cov = [[ 0.36,  0.12,  0.06],
             [ 0.12,  0.29, -0.13],
             [ 0.06, -0.13,  0.26]]
  scale_column = tf.linalg.LinearOperatorTriL(tf.cholesky(col_cov))
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])
  scale_row = tf.linalg.LinearOperatorDiag([0.9, 0.8])

  mvn = tfd.MatrixTLinearOperator(
      df=2.,
      loc=mu,
      scale_row=scale_row,
      scale_column=scale_column)

  # Initialize a 4-batch of 2 x 5-variate Matrix Ts.
  mu = tf.ones([2, 3, 5])
  scale_column_diag = [1., 2., 3., 4., 5.]
  scale_row_diag = [[0.3, 0.4, 0.6], [1., 2., 3.]]

  mvn = tfd.MatrixTLinearOperator(
      df=[1., 2., 3., 4.],
      loc=mu,
      scale_row=tf.linalg.LinearOperatorDiag(scale_row_diag),
      scale_column=tf.linalg.LinearOperatorDiag(scale_column_diag))
  ```

  NOTE: This Matrix-T distribution is parameterized slightly differently than
  the literature. Specifically, the parameterization is such that it is
  consistent with the vector case of a Multivariate Student-T.
  This also ensures that in the limit as `df` goes to infinity, we recover the
  Matrix-Normal distribution.

  The usual parameterization can be recovered by multiplying a scale factor
  by `sqrt(1. / df)` (which will in turn scale the `variance` and `covariance`
  by `1. / df`.


  #### References

  [1] G. Arjun, N. Daya, "Matrix Variate Distributions",
      Vol. 104. CRC Press.
  """

  def __init__(self,
               df,
               loc,
               scale_row,
               scale_column,
               validate_args=False,
               allow_nan_stats=True,
               name='MatrixTLinearOperator'):
    """Construct Matrix T distribution on `R^{n x p}`.

    The `batch_shape` is the broadcast shape between `loc`, `scale_row`
    and `scale_column` arguments.

    The `event_shape` is given by the matrix implied by `loc`.

    Args:
      df: Floating-point `Tensor`, having shape `[B1, ..., Bb]` representing
        the degrees of freedom parameter. Has same `dtype` as `loc`.
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
          [df, loc, scale_column, scale_row], dtype_hint=tf.float32)
      df = tensor_util.convert_nonref_to_tensor(
          df, dtype=dtype, name='df')
      loc = tensor_util.convert_nonref_to_tensor(
          loc, dtype=dtype, name='loc')
    self._df = df
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

    super(MatrixTLinearOperator, self).__init__(
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
        df=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        loc=parameter_properties.ParameterProperties(event_ndims=2),
        scale_row=parameter_properties.BatchedComponentProperties(),
        scale_column=parameter_properties.BatchedComponentProperties())

  def _as_multivariate_t(self, loc=None):
    # Rebuild the Multivariate T Distribution on every call because the
    # underlying tensor shapes might have changed.
    df = tf.convert_to_tensor(self.df)
    loc = tf.convert_to_tensor(self.loc if loc is None else loc)
    return multivariate_student_t.MultivariateStudentTLinearOperator(
        df=df,
        loc=_vec(loc),
        scale=tf.linalg.LinearOperatorKronecker(
            [self.scale_row, self.scale_column]),
        validate_args=self.validate_args)

  def _mean(self):
    shape = tf.concat([
        self.batch_shape_tensor(),
        self.event_shape_tensor()], axis=0)
    return tf.broadcast_to(self.loc, shape)

  def _variance(self):
    loc = tf.convert_to_tensor(self.loc)
    variance = self._as_multivariate_t(loc=loc).variance()
    return _unvec(variance, self._event_shape_tensor(loc=loc))

  def _mode(self):
    return self._mean()

  def _log_prob(self, x):
    return self._as_multivariate_t().log_prob(_vec(x))

  def _sample_n(self, n, seed=None):
    loc = tf.convert_to_tensor(self.loc)
    samples = self._as_multivariate_t(loc=loc).sample(n, seed=seed)
    return _unvec(samples, self._event_shape_tensor(loc=loc))

  def _sample_and_log_prob(self, sample_shape, seed):
    loc = tf.convert_to_tensor(self.loc)
    x, lp = self._as_multivariate_t(
        loc=loc).experimental_sample_and_log_prob(
            sample_shape, seed=seed)
    return _unvec(x, self._event_shape_tensor(loc=loc)), lp

  def _entropy(self):
    return self._as_multivariate_t().entropy()

  @property
  def df(self):
    """Distribution parameter for degrees of freedom."""
    return self._df

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
    if is_init != tensor_util.is_ref(self._df):
      assertions.append(assert_util.assert_positive(
          self._df, message='Argument `df` must be positive.'))
    if is_init != any(
        tensor_util.is_ref(v) for v in self.scale_column.variables):
      assertions.append(self.scale_column.assert_non_singular())
    if is_init != any(
        tensor_util.is_ref(v) for v in self.scale_row.variables):
      assertions.append(self.scale_column.assert_non_singular())
    return assertions
