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
"""Multivariate Normal distribution classes."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import fill_scale_tril as fill_scale_tril_bijector
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math import generic
from tensorflow_probability.python.stats import sample_stats

from tensorflow.python.ops.linalg import linear_operator  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'MultivariateNormalTriL',
]


@linear_operator.make_composite_tensor
class KahanLogDetLinOpTriL(tf.linalg.LinearOperatorLowerTriangular):
  """Override `LinearOperatorLowerTriangular` logdet to use Kahan summation."""

  def _log_abs_determinant(self):
    return generic.reduce_kahan_sum(
        tf.math.log(tf.math.abs(self._get_diag())), axis=[-1]).total


class MultivariateNormalTriL(
    mvn_linear_operator.MultivariateNormalLinearOperator):
  """The multivariate normal distribution on `R^k`.

  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
  `scale` matrix; `covariance = scale @ scale.T` where `@` denotes
  matrix-multiplication.

  #### Mathematical Details

  The probability density function (pdf) is,

  ```none
  pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z,
  y = inv(scale) @ (x - loc),
  Z = (2 pi)**(0.5 k) |det(scale)|,
  ```

  where:

  * `loc` is a vector in `R^k`,
  * `scale` is a matrix in `R^{k x k}`, `covariance = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.

  A (non-batch) `scale` matrix is:

  ```none
  scale = scale_tril
  ```

  where `scale_tril` is lower-triangular `k x k` matrix with non-zero diagonal,
  i.e., `tf.diag_part(scale_tril) != 0`.

  Additional leading dimensions (if any) will index batches.

  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  Trainable (batch) lower-triangular matrices can be created with
  `tfp.distributions.matrix_diag_transform()` and/or
  `tfp.math.fill_triangular()`

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 3-variate Gaussian.
  mu = [1., 2, 3]
  cov = [[ 0.36,  0.12,  0.06],
         [ 0.12,  0.29, -0.13],
         [ 0.06, -0.13,  0.26]]
  scale = tf.linalg.cholesky(cov)
  # ==> [[ 0.6,  0. ,  0. ],
  #      [ 0.2,  0.5,  0. ],
  #      [ 0.1, -0.3,  0.4]])
  mvn = tfd.MultivariateNormalTriL(
      loc=mu,
      scale_tril=scale)

  mvn.mean()
  # ==> [1., 2, 3]

  # Covariance agrees with cholesky(cov) parameterization.
  mvn.covariance()
  # ==> [[ 0.36,  0.12,  0.06],
  #      [ 0.12,  0.29, -0.13],
  #      [ 0.06, -0.13,  0.26]]

  # Compute the pdf of an observation in `R^3` ; return a scalar.
  mvn.prob([-1., 0, 1])  # shape: []

  # Initialize a 2-batch of 3-variate Gaussians.
  mu = [[1., 2, 3],
        [11, 22, 33]]              # shape: [2, 3]
  tril = ...  # shape: [2, 3, 3], lower triangular, non-zero diagonal.
  mvn = tfd.MultivariateNormalTriL(
      loc=mu,
      scale_tril=tril)

  # Compute the pdf of two `R^3` observations; return a length-2 vector.
  x = [[-0.9, 0, 0.1],
       [-10, 0, 9]]     # shape: [2, 3]
  mvn.prob(x)           # shape: [2]

  # Instantiate a "learnable" MVN.
  dims = 4
  mvn = tfd.MultivariateNormalTriL(
      loc=tf.Variable(tf.zeros([dims], dtype=tf.float32), name="mu"),
      scale_tril=tfp.util.TransformedVariable(
          tf.eye(dims, dtype=tf.float32),
          tfp.bijectors.FillScaleTriL(),
          name="raw_scale_tril")
  ```

  """

  def __init__(self,
               loc=None,
               scale_tril=None,
               validate_args=False,
               allow_nan_stats=True,
               experimental_use_kahan_sum=False,
               name='MultivariateNormalTriL'):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

    ```none
    scale = scale_tril
    ```

    where `scale_tril` is lower-triangular `k x k` matrix with non-zero
    diagonal, i.e., `tf.diag_part(scale_tril) != 0`.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_tril: Floating-point, lower-triangular `Tensor` with non-zero
        diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]` where
        `b >= 0` and `k` is the event size.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value "`NaN`" to
        indicate the result is undefined. When `False`, an exception is raised
        if one or more of the statistic's batch members are undefined.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values as well as
        when computing the log-determinant of the scale matrix. Doing so
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: if neither `loc` nor `scale_tril` are specified.
    """
    parameters = dict(locals())
    if loc is None and scale_tril is None:
      raise ValueError('Must specify one or both of `loc`, `scale_tril`.')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, scale_tril], tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
      scale_tril = tensor_util.convert_nonref_to_tensor(
          scale_tril, name='scale_tril', dtype=dtype)
      self._scale_tril = scale_tril
      if scale_tril is None:
        scale = tf.linalg.LinearOperatorIdentity(
            num_rows=ps.dimension_size(loc, -1),
            dtype=loc.dtype,
            is_self_adjoint=True,
            is_positive_definite=True,
            assert_proper_shapes=validate_args)
      else:
        # No need to validate that scale_tril is non-singular.
        # LinearOperatorLowerTriangular has an assert_non_singular
        # method that is called by the Bijector.
        linop_cls = (KahanLogDetLinOpTriL if experimental_use_kahan_sum else
                     tf.linalg.LinearOperatorLowerTriangular)
        scale = linop_cls(
            scale_tril,
            is_non_singular=True,
            is_self_adjoint=False,
            is_positive_definite=False)
      super(MultivariateNormalTriL, self).__init__(
          loc=loc,
          scale=scale,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          experimental_use_kahan_sum=experimental_use_kahan_sum,
          name=name)
      self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        scale_tril=parameter_properties.ParameterProperties(
            event_ndims=2,
            shape_fn=lambda sample_shape: ps.concat(
                [sample_shape, sample_shape[-1:]], axis=0),
            default_constraining_bijector_fn=lambda: fill_scale_tril_bijector.
            FillScaleTriL(diag_shift=dtype_util.eps(dtype))))
    # pylint: enable=g-long-lambda

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'loc': tf.reduce_mean(value, axis=0),
            'scale_tril': tf.linalg.cholesky(
                sample_stats.covariance(value, sample_axis=0, event_axis=-1))}

  @property
  def scale_tril(self):
    return self._scale_tril
