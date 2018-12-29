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
"""Multivariate Normal distribution class initialized with a full covariance."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.internal import dtype_util
from tensorflow.python.ops import control_flow_ops

import functools
import numpy as np

from tensorflow_probability.python import math
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization



__all__ = [
    "MultivariateNormalFullCovariance",
]

def _broadcast_to_shape(x, shape):
  return x + tf.zeros(shape=shape, dtype=x.dtype)

class MultivariateNormalFullCovariance(distribution.Distribution):
  """The multivariate normal distribution on `R^k`.

  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `covariance_matrix` matrices that are the covariance.
  This is different than the other multivariate normals, which are parameterized
  by a matrix more akin to the standard deviation.

  #### Mathematical Details

  The probability density function (pdf) is, with `@` as matrix multiplication,

  ```none
  pdf(x; loc, covariance) = exp(-0.5 y) / Z,
  y = (x - loc)^T @ inv(covariance) @ (x - loc)
  Z = (2 pi)**(0.5 k) |det(covariance)|**(0.5).
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `covariance_matrix` is an `R^{k x k}` symmetric positive definite matrix,
  * `Z` denotes the normalization constant.

  Additional leading dimensions (if any) in `loc` and `covariance_matrix` allow
  for batch dimensions.

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  mvn = tfd.MultivariateNormalFullCovariance(
      loc=mu,
      covariance=cov)

  mvn.mean().eval()
  # ==> [1., 2, 3]

  # Covariance agrees with covariance.
  mvn.covariance().eval()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an observation in `R^3` ; return a scalar.
  mvn.prob([-1., 0, 1]).eval()  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  covariance = ...  # shape: [2, 3, 3], symmetric, positive definite.
  mvn = tfd.MultivariateNormalFullCovariance(
      loc=mu,
      covariance=covariance)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x).eval()    # shape: [2]

  ```

  """

  def __init__(self,
               loc=None,
               covariance_matrix=None,
               validate_args=False,
               allow_nan_stats=True,
               name="MultivariateNormalFullCovariance"):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `covariance_matrix`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `covariance_matrix`. The last dimension of `loc` must broadcast with this.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. Has shape `[B1, ..., Bb, k]` where `k` is
        the event size.
      covariance_matrix: Instance of `LinearOperator` or `Tensor` with a floating `dtype` and shape
        `[B1, ..., Bb, k, k]`. If type `Tensor`, is automatically wrapped in `LinearOperatorFullCovariance`.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/variance/etc...) is undefined for
        any batch member If `True`, batch members with valid parameters leading
        to undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if not `covariance_matrix.dtype.is_floating`.
      ValueError: if not `covariance_matrix.is_positive_definite`.
      ValueError: if not `covariance_matrix.is_self_adjoint`.
    """

    parameters = dict(locals())
    if loc is None and covariance_matrix is None:
      raise ValueError("Must specify one or both of `loc`, `covariance_matrix`.")
    elif loc is None:
      dtype = covariance_matrix.dtype
      loc = tf.zeros(covariance_matrix.shape[-1].value, dtype)
    elif covariance_matrix is None:
      dtype = loc.dtype
      covariance_matrix = tf.linalg.LinearOperatorIdentity(loc.shape[-1], dtype=dtype)

    if isinstance(covariance_matrix, tf.linalg.LinearOperator):
      if not covariance_matrix.dtype.is_floating:
        raise TypeError("`covariance_matrix` must have floating-point dtype.")
      if validate_args and not covariance_matrix.is_positive_definite:
        raise ValueError("`covariance_matrix` must be positive definite.")
      if validate_args and not covariance_matrix.is_self_adjoint:
        raise ValueError("`covariance_matrix` must be self-adjoint.")

    # catch and cast if input is not a LinearOperator
    else:
      # check if not symmetric
      covariance_matrix = tf.convert_to_tensor(
        covariance_matrix, name="covariance_matrix")
      if validate_args:
        covariance_matrix = control_flow_ops.with_dependencies([
          tf.assert_near(
              covariance_matrix,
              tf.matrix_transpose(covariance_matrix),
              message="Matrix was not symmetric")
          ], covariance_matrix)
      # LinearOperator applies cholesky which will fail if not PSD
      covariance_matrix = tf.linalg.LinearOperatorFullMatrix(covariance_matrix,
                                                      is_positive_definite=True,
                                                      is_non_singular=True,
                                                      is_self_adjoint=True,
                                                      is_square=True)
                                    
    with tf.name_scope(name, values=[loc] + covariance_matrix.graph_parents) as name:
      dtype = dtype_util.common_dtype([loc, covariance_matrix],
                                      preferred_dtype=tf.float32)

      self._loc = tf.convert_to_tensor(loc, name="loc", dtype=dtype)
      self._covariance_matrix = covariance_matrix

    super(MultivariateNormalFullCovariance, self).__init__(
        dtype=dtype,
        reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._loc] + self._covariance_matrix.graph_parents,
        name=name,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats)
    self._parameters = parameters

  @property
  def loc(self):
    """The location parameter of the distribution.

    `loc` applies an elementwise shift to the distribution.

    ```none
    X ~ MultivariateNormalFullCovariance(loc=0, covariance=1)   # Identity covariance, zero shift.
    Y = covariance.cholesky() @ X + loc
    ```

    Returns:
      The `loc` `Tensor`.
    """
    return self._loc

  @property
  def covariance_matrix(self):
    """The covariance parameter of the distribution.

    `covariance_matrix` applies an affine covariance to the distribution.

    ```none
    X ~ MultivariateNormalFullCovariance(loc=0, covariance=1)   # Identity covariance, zero shift.
    Y = covariance.cholesky() @ X + loc
    ```

    Returns:
      The `covariance_matrix` `LinearOperator`.
    """
    return self._covariance_matrix

  def _batch_shape_tensor(self):
    shape_list = [
        self._covariance_matrix.batch_shape_tensor(),
        tf.shape(self.loc)[:-1]
    ]
    return functools.reduce(tf.broadcast_dynamic_shape, shape_list)

  def _batch_shape(self):
    shape_list = [self._covariance_matrix.batch_shape, self.loc.shape[:-1]]
    return functools.reduce(tf.broadcast_static_shape, shape_list)

  def _event_shape_tensor(self):
    return self._covariance_matrix.range_dimension_tensor()[tf.newaxis]

  def _event_shape(self):
    return self._covariance_matrix.range_dimension

  def _sample_shape(self):
    return tf.concat([self.batch_shape_tensor(), self.event_shape_tensor()], -1)

  def _mean(self):
    return _broadcast_to_shape(self.loc, self._sample_shape())

  def _mode(self):
    return _broadcast_to_shape(self.loc, self._sample_shape())

  def _variance(self):
    return self._covariance_matrix.diag_part()

  def _covariance(self):
    return self._covariance_matrix.to_dense()

  def _stddev(self):
    return tf.sqrt(self._variance())

  def _sample_n(self, n, seed=None):
    seed = seed_stream.SeedStream(seed, salt="multivariate normal")

    loc = _broadcast_to_shape(self.loc, self._sample_shape())
    mvn = mvn_tril.MultivariateNormalTriL(
      loc=self._loc, scale_tril=self._covariance_matrix.cholesky().to_dense())
    return mvn.sample(n, seed=seed())
            
  def _log_normalization(self):
    num_dims = tf.cast(self.event_shape_tensor()[0], self.dtype)
    return (num_dims / 2.) * np.log(2. * np.pi) + 0.5 * self._covariance_matrix.log_abs_determinant()

  def _log_unnormalized_prob(self, value):
    value -= self._loc
    value = tf.matmul(value[..., tf.newaxis], self._covariance_matrix.solve(value[..., tf.newaxis]), transpose_a=True)
    return -1. / 2. * value[..., 0, 0]

  def _log_prob(self, value):
    return self._log_unnormalized_prob(value) - self._log_normalization()


@kullback_leibler.RegisterKL(MultivariateNormalFullCovariance,
                             MultivariateNormalFullCovariance)
def _kl_brute_force_covariance(a, b, name=None):
  """Batched KL divergence `KL(a || b)` for multivariate Normals.

  With `X`, `Y` both multivariate Normals in `R^k` with means `mu_a`, `mu_b` and
  covariance `C_a`, `C_b` respectively,

  ```
  KL(a || b) = 0.5 * ( L - k + T + Q ),
  L := Log[Det(C_b)] - Log[Det(C_a)]
  T := trace(C_b^{-1} C_a),
  Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
  ```

  This `Op` computes the trace by solving `C_b^{-1} C_a`. Although efficient
  methods for solving systems with `C_b` may be available, a dense version of
  (the square root of) `C_a` is used, so performance is `O(B s k**2)` where `B`
  is the batch size, and `s` is the cost of solving `C_b x = y` for vectors `x`
  and `y`.

  Args:
    a: Instance of `MultivariateNormalFullCovariance`.
    b: Instance of `MultivariateNormalFullCovariance`.
    name: (optional) name to use for created ops. Default "kl_mvn".

  Returns:
    Batchwise `KL(a || b)`.
  """

  with tf.name_scope(
      name,
      "kl_mvn",
      values=[a.loc, b.loc] + a.covariance_matrix.graph_parents + b.covariance_matrix.graph_parents):
   
    L = b.covariance_matrix.log_abs_determinant() - a.covariance_matrix.log_abs_determinant()
    T = tf.trace(b.covariance_matrix.solve(a.covariance_matrix.to_dense()))
    Q = tf.reduce_sum((b.loc - a.loc) * b.covariance_matrix.solvevec(b.loc - a.loc), axis=[-1])
    k = tf.cast(a.covariance_matrix.domain_dimension_tensor(), a.dtype)
     
    kl_div = 0.5 * (L - k + T + Q)

    kl_div.set_shape(tf.broadcast_static_shape(a.batch_shape, b.batch_shape))
    return kl_div


