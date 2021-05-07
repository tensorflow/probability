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
"""A MultivariateNormalLinearOperator parametrized by a precision."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import scale_matvec_linear_operator
from tensorflow_probability.python.bijectors import shift as shift_bijector
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = ['MultivariateNormalPrecisionFactorLinearOperator']


class MultivariateNormalPrecisionFactorLinearOperator(
    transformed_distribution.TransformedDistribution):
  """A multivariate normal on `R^k`, parametrized by a precision factor.

  The multivariate normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `precision_factor` `LinearOperator`, and optionally a `precision`.

  The precision of this distribution is the inverse of its covariance matrix.
  The `precision_factor` is a matrix such that,

  ```
  precision = precision_factor @ precision_factor.T,
  ```

  where `@` denotes matrix-multiplication and `.T` transposition.

  Providing `precision` may improve efficiency in computation of the log
  probability density. This will be the case if matrix-vector products with
  the `precision` linear operator are more efficient than with
  `precision_factor`. For example, if `precision` has a sparse structure
  `D + X @ X.T`, where `D` is diagonal and `X` is low rank, then one may use a
  `LinearOperatorLowRankUpdate` for the `precision` arg.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, precision_factor) = exp(-0.5 ||y||**2) / Z,
  y = precision_factor @ (x - loc),
  Z = (2 pi)**(0.5 k) / |det(precision_factor)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.


  #### Examples

  ```python
  tfd_e = tfp.experimental.distributions

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  precision = tf.linalg.inv(cov)
  precision_factor = tf.linalg.cholesky(precision)

  mvn = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
      loc=mu,
      precision_factor=tf.linalg.LinearOperatorFullmatrix(precision_factor),
  )

  # Covariance is equal to `cov`.
  mvn.covariance()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an`R^3` observation; return a scalar.
  mvn.prob([-1., 0, 1])  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  variance = [[1., 2, 3],
              [0.5, 1, 1.5]]     # shape: [2, 3]
  inverse_variance = 1. / tf.constant(variance)
  diagonal_precision_factors = tf.sqrt(inverse_variance)

  mvn = tfd_e.MultivariateNormalPrecisionFactorLinearOperator(
      loc=mu,
      precision_factor=tf.linalg.LinearOperatorDiag(diagonal_precision_factors),
  )

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x)           # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               precision_factor=None,
               precision=None,
               validate_args=False,
               allow_nan_stats=True,
               name='MultivariateNormalPrecisionFactorLinearOperator'):
    """Initialize distribution.

    Precision is the inverse of the covariance matrix, and
    `precision_factor @ precision_factor.T = precision`.

    The `batch_shape` of this distribution is the broadcast of
    `loc.shape[:-1]` and `precision_factor.batch_shape`.

    The `event_shape` of this distribution is determined by `loc.shape[-1:]`,
    OR `precision_factor.shape[-1:]`, which must match.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      precision_factor: Required nonsingular `tf.linalg.LinearOperator` instance
        with same `dtype` and shape compatible with `loc`.
      precision: Optional square `tf.linalg.LinearOperator` instance with same
        `dtype` and shape compatible with `loc` and `precision_factor`.
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
      if precision_factor is None:
        raise ValueError(
            'Argument `precision_factor` must be provided. Found `None`')

      dtype = dtype_util.common_dtype([loc, precision_factor, precision],
                                      dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(loc, dtype=dtype, name='loc')

      self._loc = loc
      self._precision_factor = precision_factor
      self._precision = precision

      batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          loc, precision_factor)

      # Proof of factors (used throughout code):
      # Let,
      #   C = covariance,
      #   P = inv(covariance) = precision
      #   P = F @ F.T  (so F is the `precision_factor`).
      #
      # Then, the log prob term is
      #  x.T @ inv(C) @ x
      #  = x.T @ P @ x
      #  = x.T @ F @ F.T @ x
      #  = || F.T @ x ||**2
      # notice it involves F.T, which is why we set adjoint=True in various
      # places.
      #
      # Also, if w ~ Normal(0, I), then we can sample by setting
      #  x = inv(F.T) @ w + loc,
      # since then
      #  E[(x - loc) @ (x - loc).T]
      #  = E[inv(F.T) @ w @ w.T @ inv(F)]
      #  = inv(F.T) @ inv(F)
      #  = inv(F @ F.T)
      #  = inv(P)
      #  = C.

      if precision is not None:
        precision.shape.assert_is_compatible_with(precision_factor.shape)

      bijector = invert.Invert(
          scale_matvec_linear_operator.ScaleMatvecLinearOperator(
              scale=precision_factor,
              validate_args=validate_args,
              adjoint=True)
      )
      if loc is not None:
        shift = shift_bijector.Shift(shift=loc, validate_args=validate_args)
        bijector = shift(bijector)

      super(MultivariateNormalPrecisionFactorLinearOperator, self).__init__(
          distribution=mvn_diag.MultivariateNormalDiag(
              loc=tf.zeros(
                  ps.concat([batch_shape, event_shape], axis=0), dtype=dtype)),
          bijector=bijector,
          validate_args=validate_args,
          name=name)
      self._parameters = parameters

  @property
  def loc(self):
    # Note: if the `loc` kwarg is None, this is `None`.
    return self._loc

  @property
  def precision_factor(self):
    return self._precision_factor

  @property
  def precision(self):
    return self._precision

  experimental_is_sharded = False

  def _mean(self):
    shape = tensorshape_util.concatenate(self.batch_shape, self.event_shape)
    has_static_shape = tensorshape_util.is_fully_defined(shape)
    if not has_static_shape:
      shape = tf.concat([
          self.batch_shape_tensor(),
          self.event_shape_tensor(),
      ], 0)

    if self.loc is None:
      return tf.zeros(shape, self.dtype)

    return tf.broadcast_to(self.loc, shape)

  def _covariance(self):
    if self._precision is None:
      inv_precision_factor = self._precision_factor.inverse()
      cov = inv_precision_factor.matmul(inv_precision_factor, adjoint=True)
    else:
      cov = self._precision.inverse()
    return cov.to_dense()

  def _variance(self):
    if self._precision is None:
      precision = self._precision_factor.matmul(
          self._precision_factor, adjoint_arg=True)
    else:
      precision = self._precision
    variance = precision.inverse().diag_part()
    return tf.broadcast_to(
        variance,
        ps.broadcast_shape(ps.shape(variance),
                           ps.shape(self.loc)))

  def _stddev(self):
    return tf.sqrt(self._variance())

  def _mode(self):
    return self._mean()

  def _log_prob_unnormalized(self, value):
    """Unnormalized log probability.

    Costs a matvec and reduce_sum over a squared (batch of) vector(s).

    Args:
      value: Floating point `Tensor`.

    Returns:
      Floating point `Tensor` with batch shape.
    """
    # We override log prob functions in order to make use of self._precision.
    if self._loc is None:
      dx = value
    else:
      dx = value - self._loc

    if self._precision is None:
      # See "Proof of factors" above for use of adjoint=True.
      dy = self._precision_factor.matvec(dx, adjoint=True)
      return -0.5 * tf.reduce_sum(dy**2, axis=-1)
    return -0.5 * tf.einsum('...i,...i->...', dx, self._precision.matvec(dx))

  def _log_prob(self, value):
    """Log probability of multivariate normal.

    Costs a log_abs_determinant, matvec, and a reduce_sum over a squared
    (batch of) vector(s)

    Args:
      value: Floating point `Tensor`.

    Returns:
      Floating point `Tensor` with batch shape.
    """
    dim = self.precision_factor.domain_dimension_tensor()
    return (ps.cast(-0.5 * np.log(2 * np.pi), self.dtype) *
            ps.cast(dim, self.dtype) +
            # Notice the sign on the LinearOperator.log_abs_determinant is
            # positive, since it is precision_factor not scale.
            self._precision_factor.log_abs_determinant() +
            self._log_prob_unnormalized(value))

  _composite_tensor_nonshape_params = ('loc', 'precision_factor', 'precision')
