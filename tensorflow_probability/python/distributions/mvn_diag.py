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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.distributions import mvn_linear_operator
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import tensor_util
from tensorflow.python.ops.linalg import linear_operator  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'MultivariateNormalDiag',
]


@linear_operator.make_composite_tensor
class KahanLogDetLinOpDiag(tf.linalg.LinearOperatorDiag):
  """Override `LinearOperatorDiag` logdet to use Kahan summation."""

  def _log_abs_determinant(self):
    log_det = tfp_math.reduce_kahan_sum(
        tf.math.log(tf.math.abs(self._diag)), axis=[-1]).total
    if dtype_util.is_complex(self.dtype):
      log_det = tf.cast(log_det, dtype=self.dtype)
    return log_det


class MultivariateNormalDiag(
    mvn_linear_operator.MultivariateNormalLinearOperator):
  """The multivariate normal distribution on `R^k`.

  The Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (aka 'mu') and a (batch of) `k x k`
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
  * `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
  * `Z` denotes the normalization constant, and,
  * `||y||**2` denotes the squared Euclidean norm of `y`.

  A (non-batch) `scale` matrix is:

  ```none
  scale = diag(scale_diag + scale_identity_multiplier * ones(k))
  ```

  where:

  * `scale_diag.shape = [k]`, and,
  * `scale_identity_multiplier.shape = []`.

  Additional leading dimensions (if any) will index batches.

  If both `scale_diag` and `scale_identity_multiplier` are `None`, then
  `scale` is the Identity matrix.

  The MultivariateNormal distribution is a member of the [location-scale
  family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
  constructed as,

  ```none
  X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
  Y = scale @ X + loc
  ```

  #### Examples

  ```python
  tfd = tfp.distributions

  # Initialize a single 2-variate Gaussian.
  mvn = tfd.MultivariateNormalDiag(
      loc=[1., -1],
      scale_diag=[1, 2.])

  mvn.mean()
  # ==> [1., -1]

  mvn.stddev()
  # ==> [1., 2]

  # Evaluate this on an observation in `R^2`, returning a scalar.
  mvn.prob([-1., 0])  # shape: []

  # Initialize a 3-batch, 2-variate scaled-identity Gaussian.
  mvn = tfd.MultivariateNormalDiag(
      loc=[1., -1],
      scale_identity_multiplier=[1, 2., 3])

  mvn.mean()  # shape: [3, 2]
  # ==> [[1., -1]
  #      [1, -1],
  #      [1, -1]]

  mvn.stddev()  # shape: [3, 2]
  # ==> [[1., 1],
  #      [2, 2],
  #      [3, 3]]

  # Evaluate this on an observation in `R^2`, returning a length-3 vector.
  mvn.prob([-1., 0])  # shape: [3]

  # Initialize a 2-batch of 3-variate Gaussians.
  mvn = tfd.MultivariateNormalDiag(
      loc=[[1., 2, 3],
           [11, 22, 33]],           # shape: [2, 3]
      scale_diag=[[1., 2, 3],
                  [0.5, 1, 1.5]])  # shape: [2, 3]

  # Evaluate this on a two observations, each in `R^3`, returning a length-2
  # vector.
  x = [[-1., 0, 1],
       [-11, 0, 11.]]   # shape: [2, 3].
  mvn.prob(x)    # shape: [2]
  ```

  """

  @deprecation.deprecated_args(
      '2020-01-01',
      '`scale_identity_multiplier` is deprecated; please combine it into '
      '`scale_diag` directly instead.',
      'scale_identity_multiplier')
  def __init__(self,
               loc=None,
               scale_diag=None,
               scale_identity_multiplier=None,
               validate_args=False,
               allow_nan_stats=True,
               experimental_use_kahan_sum=False,
               name='MultivariateNormalDiag'):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and `scale`
    arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `scale`. The last dimension of `loc` (if provided) must broadcast with this.

    Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

    ```none
    scale = diag(scale_diag + scale_identity_multiplier * ones(k))
    ```

    where:

    * `scale_diag.shape = [k]`, and,
    * `scale_identity_multiplier.shape = []`.

    Additional leading dimensions (if any) will index batches.

    If both `scale_diag` and `scale_identity_multiplier` are `None`, then
    `scale` is the Identity matrix.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      scale_diag: Non-zero, floating-point `Tensor` representing a diagonal
        matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
        and characterizes `b`-batches of `k x k` diagonal matrices added to
        `scale`. When both `scale_identity_multiplier` and `scale_diag` are
        `None` then `scale` is the `Identity`.
      scale_identity_multiplier: Non-zero, floating-point `Tensor` representing
        a scaled-identity-matrix added to `scale`. May have shape
        `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
        `k x k` identity matrices added to `scale`. When both
        `scale_identity_multiplier` and `scale_diag` are `None` then `scale` is
        the `Identity`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`,
        statistics (e.g., mean, mode, variance) use the value '`NaN`' to
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
      ValueError: if at most `scale_identity_multiplier` is specified.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [loc, scale_diag, scale_identity_multiplier], dtype_hint=tf.float32)
      loc = tensor_util.convert_nonref_to_tensor(loc, name='loc', dtype=dtype)
      scale_diag = tensor_util.convert_nonref_to_tensor(
          scale_diag, name='scale_diag', dtype=dtype)
      if scale_diag is not None and scale_identity_multiplier is not None:
        raise ValueError(
            'Only one of `scale_diag` and `scale_identity_multiplier` is '
            'allowed. Furthermore, `scale_identity_multiplier` is deprecated; '
            'please combine it directly into `scale_diag` instead.')

      if scale_diag is not None:
        diag_cls = (KahanLogDetLinOpDiag if experimental_use_kahan_sum else
                    tf.linalg.LinearOperatorDiag)
        scale = diag_cls(
            diag=scale_diag,
            is_non_singular=True,
            is_self_adjoint=True,
            is_positive_definite=False)
      else:
        # Deprecated behavior; breaks variable-safety rules by calling
        # `tf.shape(loc)`.
        num_rows = tf.compat.dimension_value(loc.shape[-1])
        if num_rows is None:
          num_rows = tf.shape(loc)[-1]
        if scale_identity_multiplier is not None:
          scale_identity_multiplier = tensor_util.convert_nonref_to_tensor(
              scale_identity_multiplier,
              name='scale_identity_multiplier',
              dtype=dtype)
          scale = tf.linalg.LinearOperatorScaledIdentity(
              num_rows=num_rows,
              multiplier=scale_identity_multiplier,
              is_non_singular=True,
              is_self_adjoint=True,
              is_positive_definite=False,
              assert_proper_shapes=False)
        else:
          scale = tf.linalg.LinearOperatorIdentity(
              num_rows=num_rows,
              dtype=dtype,
              is_self_adjoint=True,
              is_positive_definite=True,
              assert_proper_shapes=validate_args)

      super(MultivariateNormalDiag, self).__init__(
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
        scale_diag=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        scale_identity_multiplier=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))),
            is_preferred=False))
    # pylint: enable=g-long-lambda

  @classmethod
  def _maximum_likelihood_parameters(cls, value):
    return {'loc': tf.reduce_mean(value, axis=0),
            'scale_diag': tf.math.reduce_std(value, axis=0)}
