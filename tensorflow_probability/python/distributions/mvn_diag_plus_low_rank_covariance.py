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
"""Multivariate Normal distribution class initialized with a diagonal plus low-rank covariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.internal import dtype_util
from tensorflow.python.ops import control_flow_ops
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import distribution_util

__all__ = [
    "MultivariateNormalDiagPlusLowRankCovariance",
]

class MultivariateNormalDiagPlusLowRankCovariance(mvn_tril.MultivariateNormalTriL):
  """The multivariate normal distribution on `R^k`.
  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `covariance_matrix` matrices that are the covariance.
  This is different than the other multivariate normals, which are parameterized
  by a matrix more akin to the standard deviation.

  #### Mathematical Details
  The probability density function (pdf) is, with `@` as matrix multiplication,
  ```none
  pdf(x; loc, covariance_matrix) = exp(-0.5 y) / Z,
  y = (x - loc)^T @ inv(covariance_matrix) @ (x - loc)
  Z = (2 pi)**(0.5 k) |det(covariance_matrix)|**(0.5).
  covariance_matrix = cov_perturb_factor * cov_perturb_tril * cov_perturb_tril^T * cov_perturb_factor^T + cov_identity_multiplier * I + diag(cov_diag)
  ```
  where:
  * `loc` is a vector in `R^k`,
  * `cov_diag` is a vector in `R^k`,
  * `cov_identity_multiplier` is a scalar,
  * `cov_perturb_factor` is a matrix in R^{k x s}`,
  * `cov_perturb_tril` is a lower-triangular matrix in R^{s x s},
  * `Z` denotes the normalization constant.
  Additional leading dimensions (if any) in `loc` and `covariance_matrix` allow
  for batch dimensions.
  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed e.g. as,
  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  scale = Cholesky(covariance_matrix)
  Y = scale @ X + loc
  ```

  """

  def __init__(self,
               loc=None,
               cov_diag=None,
               cov_identity_multiplier=None,
               cov_perturb_factor=None,
               cov_perturb_tril=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalFullCovariance"):
    """Construct Multivariate Normal distribution on `R^k`.
    The `batch_shape` is the broadcast shape between `loc` and
    `covariance_matrix` arguments.
    The `event_shape` is given by last dimension of the matrix implied by
    `covariance_matrix`. The last dimension of `loc` (if provided) must
    broadcast with this.
    A non-batch `covariance_matrix` matrix is a `k x k` symmetric positive
    definite matrix.  In other words it is (real) symmetric with all eigenvalues
    strictly positive.
    Additional leading dimensions (if any) will index batches.
    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      covariance_matrix: Floating-point, symmetric positive definite `Tensor` of
        same `dtype` as `loc`.  The strict upper triangle of `covariance_matrix`
        is ignored, so if `covariance_matrix` is not symmetric no error will be
        raised (unless `validate_args is True`).  `covariance_matrix` has shape
        `[B1, ..., Bb, k, k]` where `b >= 0` and `k` is the event size.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      ValueError: if neither `loc` nor `covariance_matrix` are specified.
    """
    parameters = dict(locals())

    def _convert_to_tensor(x, name, dtype=None):
      return None if x is None else tf.convert_to_tensor(
          x, name=name, dtype=dtype)

    self

    with tf.name_scope(name) as name:
      with tf.name_scope(
          "init",
          values=[
              loc, cov_diag, cov_identity_multiplier, cov_perturb_factor, cov_perturb_tril
          ]):
        dtype = dtype_util.common_dtype([
            loc, cov_diag, cov_identity_multiplier, cov_perturb_factor,
            cov_perturb_tril
        ], tf.float32)
        has_low_rank = (cov_perturb_factor is not None or
                        cov_perturb_tril is not None)
        diagonal = distribution_util.make_diag_scale(
            loc=loc,
            scale_diag=cov_diag,
            scale_identity_multiplier=cov_identity_multiplier,
            validate_args=validate_args,
            assert_positive=has_low_rank,
            dtype=dtype)
        cov_perturb_factor = _convert_to_tensor(
            cov_perturb_factor, name="cov_perturb_factor", dtype=dtype)
        cov_perturb_tril = _convert_to_tensor(
            cov_perturb_tril, name="cov_perturb_tril", dtype=dtype)
        cov_perturb_factor_sheared = tf.matmul(cov_perturb_factor, cov_perturb_tril, name='cov_perturb_factor_sheared')
        if has_low_rank:
          covariance_matrix = tf.linalg.LinearOperatorLowRankUpdate(
              diagonal,
              u=cov_perturb_factor_sheared,
              diag_update=None,
              is_diag_update_positive=True,
              is_non_singular=True,  # Implied by is_positive_definite=True.
              is_self_adjoint=True,
              is_positive_definite=True,
              is_square=True)
        else:
          covariance_matrix = diagonal

    # Convert the covariance_matrix up to a scale_tril and call MVNTriL.
      loc = loc if loc is None else tf.convert_to_tensor(
          loc, name="loc", dtype=dtype)
      scale_tril = tf.cholesky(covariance_matrix.to_dense())
      super(MultivariateNormalDiagPlusLowRankCovariance, self).__init__(
          loc=loc,
          scale_tril=scale_tril,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters
    self._diagonal = diagonal
    self._covariance_matrix = covariance_matrix
    self._auxiliary_factor = cov_perturb_factor_sheared / self.diagonal[..., None]
    self._auxiliary_tril = tf.cholesky(tf.eye(cov_perturb_tril.shape[0].value) + tf.matmul(self._auxiliary_factor, cov_perturb_factor_sheared, transpose_a=True))

  @property
  def diagonal(self):
    return self._diagonal.diag_part()

  def covariance(self):
    return self._covariance_matrix.to_dense()

  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()

  def _log_normalization(self):
    return (0.5 * self.loc.shape[0].value * math.log(2. * math.pi) + 
            0.5 * tf.reduce_sum(tf.log(self.diagonal)) + tf.reduce_sum(tf.log(tf.diag_part(self._auxiliary_tril))))

  def _log_unnormalized_prob(self, x):
    z = (x - self.loc) / tf.sqrt(self.diagonal)
    auxz = tf.linalg.triangular_solve(self._auxiliary_tril, tf.matmul(self._auxiliary_factor, x[...,None] - self.loc[...,None], transpose_a=True))[...,0]
    return -0.5 * tf.reduce_sum(tf.square(z), axis=-1) + 0.5 * tf.reduce_sum(tf.square(auxz), axis=-1)