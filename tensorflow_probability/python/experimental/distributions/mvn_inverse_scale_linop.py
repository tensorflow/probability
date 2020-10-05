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
"""A MultivariateNormalLinearOperator parametrized by inverse scale."""

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers


__all__ = ['MultivariateNormalInverseScaleLinearOperator']


class MultivariateNormalInverseScaleLinearOperator(
    mvn_linear_operator.MultivariateNormalLinearOperator):
  """A multivariate normal distribution on `R^k`, parametrized by inverse scale.

  The multivariate normal distribution is defined over `R^k` and may be
  parameterized by a (batch of) length-`k` `loc` vector (aka "mu") and a
  (batch of) `k x k` `inverse_scale` matrix.

  If the covariance of this distribution is `C`, and the "scale" is a matrix
  `S` such that `C = S @ S.T`, then `inverse_scale = inverse(S)`. In other
  words, `inverse_scale` is the inverse of the scale matrix.

  This `Distribution` optionally allows specifying the `precision` as a
  `tf.linalg.LinearOperator`. If the covariance of this distribution is `C`,
  then `precision = inv(C) = inverse_scale.T @ inverse_scale`.

  Supplying `precision` may lead to savings in computing the log probability
  when the `precision` operator `P` has a particular structure. For example, if
  `P = D + X @ X.T`, where `X` is low rank, then matrix multiplication with `P`
  may be more efficient than with `inverse_scale`. These optimizations will
  rely on `P` implementing that efficient matrix multiplication.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, inverse_scale) = exp(-0.5 ||y||**2) / Z,
  y = inverse_scale @ (x - loc),
  Z = (2 pi)**(0.5 k) / |det(inverse_scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `inverse_scale` is a linear operator in `R^{k x k}`, where covariance
    `cov = S @ S^T` and `inverse_scale = inv(S)`,
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
  scale = tf.linalg.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])

  mvn = tfd_e.MultivariateNormalInverseScaleLinearOperator(
      loc=mu,
      inverse_scale=tf.linalg.LinearOperatorLowerTriangular(scale).inverse())

  # Covariance agrees with cholesky(cov) parameterization.
  mvn.covariance()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an`R^3` observation; return a scalar.
  mvn.prob([-1., 0, 1])  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  scale_diag = [[1., 2, 3],
                [0.5, 1, 1.5]]     # shape: [2, 3]

  mvn = tfd_e.MultivariateNormalInverseScaleLinearOperator(
      loc=mu,
      inverse_scale=tf.linalg.LinearOperatorDiag(scale_diag))

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x)           # shape: [2]
  ```

  """

  def __init__(self,
               loc,
               inverse_scale,
               precision=None,
               validate_args=False,
               allow_nan_stats=True,
               name='MultivariateNormalInverseScaleLinearOperator'):
    """Initialize distribution.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.


    Args:
      loc: Floating-point `Tensor`. May have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      inverse_scale: `tf.linalg.LinearOperator` instance, indicating the inverse
        of the scale of the distribution. Specifically, if the covariance is
        `S S.T`, then `inverse_scale = inv(S)`.
      precision: `tf.linalg.LinearOperator` instance, which should represent a
        matrix `P` such that `inv(P) = covariance`. Equivalently,
        `P = inverse_scale.T @ inverse_scale`.  If this is provided, it will be
        used to calculate the log probability.
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
      self._inverse_scale = inverse_scale
      self._precision = precision
      super(MultivariateNormalInverseScaleLinearOperator, self).__init__(
          loc=loc,
          scale=inverse_scale.inverse(),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters

  @property
  def inverse_scale(self):
    return self._inverse_scale

  @property
  def precision(self):
    return self._precision

  def _log_prob_unnormalized(self, value):
    """Unnormalized log probability.

    Costs a matvec and reduce_sum over a squared (batch of) vector(s).

    Args:
      value: Floating point `Tensor`.

    Returns:
      Floating point `Tensor` with batch shape.
    """
    dx = value - self._loc
    if self._precision is None:
      inv_scale_dx = self._inverse_scale.matvec(dx)
      return -0.5 * tf.reduce_sum(inv_scale_dx**2, axis=-1)
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
    return (-0.5 * tf.cast(self.event_shape[-1], tf.float32) *
            np.log(2 * np.pi) +
            self._inverse_scale.log_abs_determinant() +
            self._log_prob_unnormalized(value))

  def _sample_n(self, n, seed=None):
    """Draw n samples with the given inverse scale.

    Sampling costs one solve using the inverse scale. For example, if the
    `inverse_scale` is a `tf.linalg.LinearOperator(A).inverse()`,
    then sampling costs one matrix multiply.

    Args:
      n: Int, number of samples to return
      seed: Optional.

    Returns:
      Floating point `Tensor`.
    """
    return self._inverse_scale.solvevec(
        samplers.normal(
            shape=ps.concat([[n], ps.shape(self._loc)], axis=0), seed=seed),
        adjoint=True)
