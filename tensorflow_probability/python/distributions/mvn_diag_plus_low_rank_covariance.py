# Copyright 2021 The TensorFlow Probability Authors.
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
"""MVN with covariance parameterized by a diagonal and a low rank update."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import mvn_low_rank_update_linear_operator_covariance
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util

__all__ = [
    'MultivariateNormalDiagPlusLowRankCovariance',
]


class MultivariateNormalDiagPlusLowRankCovariance(
    mvn_low_rank_update_linear_operator_covariance
    .MultivariateNormalLowRankUpdateLinearOperatorCovariance):
  """The multivariate normal distribution on `R^k`.

  This Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (the mean) and a (batch of) `k x k`
  `covariance` matrix.

  The covariance matrix for this particular Normal is a (typically low rank)
  perturbation of a diagonal matrix.
  Compare to `MultivariateNormalDiagPlusLowRank` which perturbs the *scale*
  rather than covariance.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, covariance) = exp(-0.5 y^T @ inv(covariance) @ y) / Z,
  y := x - loc
  Z := (2 pi)**(0.5 k) |det(covariance)|**0.5,
  ```

  where `^T` denotes matrix transpose and `@` matrix multiplication

  The MultivariateNormal distribution can also be parameterized as a
  [location-scale family](https://en.wikipedia.org/wiki/Location-scale_family),
  i.e., it can be constructed using a matrix `scale` such that
  `covariance = scale @ scale^T`, and then

  ```none
  X ~ MultivariateNormal(loc=0, scale=I)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 2-variate Gaussian.
  # The covariance is a rank 1 update of a diagonal matrix.
  loc = [1., 2.]
  cov_diag_factor = [1., 1.]
  cov_perturb_factor = tf.ones((2, 1)) * np.sqrt(2)  # Unit vector
  mvn = MultivariateNormalDiagPlusLowRankCovariance(
      loc,
      cov_diag_factor,
      cov_perturb_factor)

  # Covariance agrees with
  #   tf.linalg.matrix_diag(cov_diag_factor)
  #     + cov_perturb_factor @ cov_perturb_factor.T
  mvn.covariance()
  # ==> [[ 2., 1.],
  #      [ 1., 2.]]

  # Compute the pdf of an`R^2` observation; return a scalar.
  mvn.prob([-1., 0])  # shape: []

  # Initialize a 2-batch of 2-variate Gaussians.
  mu = [[1., 2],
        [11, 22]]              # shape: [2, 2]
  cov_diag_factor = [[1., 2],
                            [0.5, 1]]     # shape: [2, 2]
  cov_perturb_factor = tf.ones((2, 1)) * np.sqrt(2)  # Broadcasts!
  mvn = MultivariateNormalDiagPlusLowRankCovariance(
      loc,
      cov_diag_factor,
      cov_perturb_factor)

  # Compute the pdf of two `R^2` observations; return a length-2 vector.
  x = [[-0.9, 0],
       [-10, 0]]     # shape: [2, 2]
  mvn.prob(x)    # shape: [2]
  ```

  """

  def __init__(
      self,
      loc=None,
      cov_diag_factor=None,
      cov_perturb_factor=None,
      validate_args=False,
      allow_nan_stats=True,
      name='MultivariateNormalDiagPlusLowRankCovariance'):
    """Construct Multivariate Normal distribution on `R^k`.

    The covariance matrix is constructed as an efficient implementation of:

    ```
    update = cov_perturb_factor @ cov_perturb_factor^T
    covariance = tf.linalg.matrix_diag(cov_diag_factor) + update
    ```

    The `batch_shape` is the broadcast shape between `loc` and covariance args.

    The `event_shape` is given by last dimension of the matrix implied by the
    covariance. The last dimension of `loc` (if provided) must broadcast with
    this.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      cov_diag_factor: `Tensor` of same dtype as `loc` and broadcastable
        shape. Should have positive entries.
      cov_perturb_factor: `Tensor` of same dtype as `loc` and shape that
        broadcasts with `loc.shape + [M]`, where if `M < k` this is a low rank
        update.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      ValueError: if either of `cov_diag_factor` or
        `cov_perturb_factor` is unspecified.
    """
    parameters = dict(locals())
    if cov_diag_factor is None:
      raise ValueError('Missing required `cov_diag_factor` parameter.')
    if cov_perturb_factor is None:
      raise ValueError(
          'Missing required `cov_perturb_factor` parameter.')

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, cov_diag_factor, cov_perturb_factor],
          dtype_hint=tf.float32)
      cov_diag_factor = tensor_util.convert_nonref_to_tensor(
          cov_diag_factor, dtype=dtype, name='cov_diag_factor')
      cov_perturb_factor = tensor_util.convert_nonref_to_tensor(
          cov_perturb_factor,
          dtype=dtype,
          name='cov_perturb_factor')
      loc = tensor_util.convert_nonref_to_tensor(loc, dtype=dtype, name='loc')

      cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
          base_operator=tf.linalg.LinearOperatorDiag(
              cov_diag_factor,
              # The user is required to provide a positive
              # cov_diag_factor. If they don't, then unexpected behavior
              # will happen, and may not be caught unless validate_args=True.
              is_positive_definite=True,
          ),
          u=cov_perturb_factor,
          # If cov_diag_factor > 0, then cov_operator is SPD since
          # it is of the form D + UU^T.
          is_positive_definite=True)

    super(MultivariateNormalDiagPlusLowRankCovariance, self).__init__(
        loc=loc,
        cov_operator=cov_operator,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        name=name)
    self._parameters = parameters
    self._cov_diag_factor = cov_diag_factor
    self._cov_perturb_factor = cov_perturb_factor

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        cov_diag_factor=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        cov_perturb_factor=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=parameter_properties.SHAPE_FN_NOT_IMPLEMENTED),
    )

  @property
  def cov_diag_factor(self):
    """The diagonal term in the covariance."""
    return self._cov_diag_factor

  @property
  def cov_perturb_factor(self):
    """The (probably low rank) update term in the covariance."""
    return self._cov_perturb_factor

  _composite_tensor_nonshape_params = (
      'loc', 'cov_diag_factor', 'cov_perturb_factor')
