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
"""MVN with covariance parameterized by a LinearOperatorLowRankUpdate."""

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util

# Not part of public API since we're unsure if the base distribution should be
# this, or a more general MVNCovariance.
__all__ = []

_mvn_sample_note = """
`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

"""


class MultivariateNormalLowRankUpdateLinearOperatorCovariance(
    distribution.AutoCompositeTensorDistribution):
  """The multivariate normal distribution on `R^k`.

  This Multivariate Normal distribution is defined over `R^k` and parameterized
  by a (batch of) length-`k` `loc` vector (the mean) and a (batch of) `k x k`
  `covariance` matrix. The covariance matrix for this particular Normal is given
  as a linear operator, `LinearOperatorLowRankUpdate`.

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
  diag = [1., 1.]
  u = tf.ones((2, 1)) * np.sqrt(2)  # Unit vector
  cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
      base_operator=tf.linalg.LinearOperatorDiag(diag,
                                                 is_positive_definite=True),
      u=u,
  )
  mvn = MultivariateNormalLowRankUpdateLinearOperatorCovariance(
      loc=loc, cov_operator=cov_operator)

  # Covariance agrees with the cov_operator.
  mvn.covariance()
  # ==> [[ 2., 1.],
  #      [ 1., 2.]]

  # Compute the pdf of an`R^2` observation; return a scalar.
  mvn.prob([-1., 0])  # shape: []

  # Initialize a 2-batch of 2-variate Gaussians.
  mu = [[1., 2],
        [11, 22]]              # shape: [2, 2]
  diag = [[1., 2],
          [0.5, 1]]     # shape: [2, 2]
  u = tf.ones((2, 1)) * np.sqrt(2)  # Unit vector, will broadcast over batches!
  cov_operator = tf.linalg.LinearOperatorLowRankUpdate(
      base_operator=tf.linalg.LinearOperatorDiag(diag,
                                                 is_positive_definite=True),
      u=u,
  )
  mvn = MultivariateNormalLowRankUpdateLinearOperatorCovariance(
      loc=loc, cov_operator=cov_operator)

  # Compute the pdf of two `R^2` observations; return a length-2 vector.
  x = [[-0.9, 0],
       [-10, 0]]     # shape: [2, 2]
  mvn.prob(x)    # shape: [2]
  ```

  """

  def __init__(self,
               loc=None,
               cov_operator=None,
               validate_args=False,
               allow_nan_stats=True,
               name='MultivariateNormalLowRankUpdateLinearOperatorCovariance'):
    """Construct Multivariate Normal distribution on `R^k`.

    The `batch_shape` is the broadcast shape between `loc` and
    `cov_operator` arguments.

    The `event_shape` is given by last dimension of the matrix implied by
    `cov_operator`. The last dimension of `loc` (if provided) must
    broadcast with this.

    Additional leading dimensions (if any) will index batches.

    Args:
      loc: Floating-point `Tensor`. If this is set to `None`, `loc` is
        implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
        `b >= 0` and `k` is the event size.
      cov_operator: Instance of `LinearOperatorLowRankUpdate` with same
        `dtype` as `loc` and shape `[B1, ..., Bb, k, k]`.  Must have structure
        `A + UU^T` or `A + UDU^T`, where `A` and `D` (if provided) are
        self-adjoint and positive definite.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      ValueError: if `cov_operator` is unspecified.
      ValueError: if `cov_operator` does not specify the self-adjoint
        positive definite conditions explained above.
      TypeError: if not `cov_operator.dtype.is_floating`
    """
    parameters = dict(locals())
    if cov_operator is None:
      raise ValueError('Missing required `cov_operator` parameter.')
    if not dtype_util.is_floating(cov_operator.dtype):
      raise TypeError(
          '`cov_operator` parameter must have floating-point dtype.')
    if not isinstance(cov_operator,
                      tf.linalg.LinearOperatorLowRankUpdate):
      raise TypeError(
          '`cov_operator` must be a LinearOperatorLowRankUpdate. '
          'Found {}'.format(type(cov_operator)))

    if cov_operator.u is not cov_operator.v:
      raise ValueError('The `U` and `V` (typically low rank) matrices of '
                       '`cov_operator` must be the same, but were not.')

    # For cov_operator, raise if the user explicitly set these to False,
    # or if False was inferred by the LinearOperator. The default value is None,
    # which will not trigger these raises.
    # pylint: disable=g-bool-id-comparison
    if cov_operator.is_self_adjoint is False:
      raise ValueError('`cov_operator` must be self-adjoint.')
    if cov_operator.is_positive_definite is False:
      raise ValueError('`cov_operator` must be positive definite.')
    # pylint: enable=g-bool-id-comparison

    # For the base_operator, we require the user to explicity set
    # is_self_adjoint and is_positive_definite.
    if not cov_operator.base_operator.is_self_adjoint:
      raise ValueError(
          'The `base_operator` of `cov_operator` must be self-adjoint. '
          'You may have to set the `is_self_adjoint` initialization hint.')
    if not cov_operator.base_operator.is_positive_definite:
      raise ValueError(
          'The `base_operator` of `cov_operator` must be positive '
          'definite. You may have to set the `is_positive_definite` '
          'initialization hint.')

    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([loc, cov_operator],
                                      dtype_hint=tf.float32)
      if loc is not None:
        loc = tensor_util.convert_nonref_to_tensor(loc, dtype=dtype, name='loc')

      # Get dynamic shapes (for self.*shape_tensor methods).
      # shapes_from_loc_and_scale tries to return TensorShapes, but may return
      # tensors. So we can only use it for the *shape_tensor methods.
      # It is useful though, since it does lots of shape checks, and is a
      # well-tested function.
      batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
          loc, cov_operator)
      self._batch_shape_tensor_value = ps.convert_to_shape_tensor(
          batch_shape, name='batch_shape')
      self._event_shape_tensor_value = ps.convert_to_shape_tensor(
          event_shape, name='event_shape')

      # Get static shapes (for self.*shape methods).
      self._batch_shape_value = cov_operator.batch_shape
      if loc is not None:
        self._batch_shape_value = tf.broadcast_static_shape(
            self._batch_shape_value, loc.shape[:-1])
      self._event_shape_value = cov_operator.shape[-1:]
      if loc is not None:
        self._event_shape_value = tf.broadcast_static_shape(
            self._event_shape_value, loc.shape[-1:])

    self._loc = loc
    self._cov_operator = cov_operator

    super(MultivariateNormalLowRankUpdateLinearOperatorCovariance,
          self).__init__(
              dtype=dtype,
              reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
              validate_args=validate_args,
              allow_nan_stats=allow_nan_stats,
              parameters=parameters,
              name=name)
    self._parameters = parameters

  @property
  def loc(self):
    """The location arg for this distribution."""
    return self._loc

  @property
  def cov_operator(self):
    """The linear operator providing the covariance of this distribution."""
    return self._cov_operator

  def _batch_shape_tensor(self, loc=None, scale=None):
    return self._batch_shape_tensor_value

  def _batch_shape(self):
    return self._batch_shape_value

  def _event_shape_tensor(self):
    return self._event_shape_tensor_value

  def _event_shape(self):
    return self._event_shape_value

  def _batch_plus_event_shape(self):
    """Prefer static version of self.batch_shape + self.event_shape."""
    return ps.concat([
        self._batch_shape_tensor(),
        self._event_shape_tensor(),
    ],
                     axis=0)

  def _entropy(self):
    d = tf.cast(self._event_shape_tensor()[-1], self.dtype)
    const = (d / 2.) * tf.cast(np.log(2. * np.pi * np.exp(1.)),
                               dtype=self.dtype)
    entropy_value = const + 0.5 * self.cov_operator.log_abs_determinant()
    return tf.broadcast_to(entropy_value, self._batch_shape_tensor())

  def _sample_n(self, n, seed=None):
    seed_1, seed_2 = samplers.split_seed(seed, n=2)

    cov = self.cov_operator

    # Convert, in case cov.u is a ref
    u = tf.convert_to_tensor(cov.u, name='u')

    full_shape = ps.concat(
        [[n], self._batch_shape_tensor(),
         self._event_shape_tensor()], axis=0)
    low_rank_shape = ps.concat(
        [[n], self._batch_shape_tensor(),
         ps.shape(u)[-1:]], axis=0)
    w1 = samplers.normal(
        shape=full_shape,
        mean=0.,
        stddev=1.,
        dtype=self.dtype,
        seed=seed_1)
    w2 = samplers.normal(
        shape=low_rank_shape,
        mean=0.,
        stddev=1.,
        dtype=self.dtype,
        seed=seed_2)

    # Important: cov.diag_operator is the diagonal part of the perturbation.
    # More details: cov takes one of two forms,
    #   = B + U D U^T,
    # with B = cov.base_operator, U = cov.u, D = cov.diag_operator
    # or,
    #   = B + U U^T

    # B^{1/2} @ w1
    base_matvec_w1 = cov.base_operator.cholesky().matvec(w1)

    # U @ D^{1/2} @ w2
    d_one_half = cov.diag_operator.cholesky()
    low_rank_matvec_w2 = tf.linalg.matvec(u, d_one_half.matvec(w2))

    samples = base_matvec_w1 + low_rank_matvec_w2
    if self.loc is not None:
      samples += self.loc

    return samples

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        loc=parameter_properties.ParameterProperties(event_ndims=1),
        cov_operator=parameter_properties.BatchedComponentProperties())

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _log_prob(self, x):
    # This is how any MVN log prob could be written, assuming we had the
    # covariance as a linear operator.
    if self.loc is not None:
      x = x - self.loc
    quad_form = tf.reduce_sum(x * self.cov_operator.solvevec(x), axis=-1)
    d = tf.cast(self._event_shape_tensor()[-1], self.dtype)
    log_normalizer = (
        (d / 2.) * np.log(2. * np.pi) +
        (1. / 2.) * self.cov_operator.log_abs_determinant())
    return -(1. / 2.) * quad_form - log_normalizer

  @distribution_util.AppendDocstring(_mvn_sample_note)
  def _prob(self, x):
    return tf.math.exp(self.log_prob(x))

  def _mean(self):
    shape = self._batch_plus_event_shape()
    if self.loc is None:
      return tf.zeros(shape, self.dtype)

    return tf.broadcast_to(self.loc, shape)

  def _covariance(self):
    cov = self.cov_operator.to_dense()
    if self.loc is not None:
      batch_plus_event_shape = self._batch_plus_event_shape()
      shape = ps.concat([
          batch_plus_event_shape,
          batch_plus_event_shape[-1:],
      ],
                        axis=0)
      cov = tf.broadcast_to(cov, shape)
    return cov

  def _variance(self):
    variance = self.cov_operator.diag_part()
    if self.loc is not None:
      variance = tf.broadcast_to(variance, self._batch_plus_event_shape())
    return variance

  def _mode(self):
    return self._mean()

  def _default_event_space_bijector(self):
    return identity_bijector.Identity(validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []

    if is_init != any(
        tensor_util.is_ref(v) for v in self.cov_operator.variables):
      return [
          self.cov_operator.assert_self_adjoint(),
          self.cov_operator.assert_positive_definite(),
      ]

    return []

  _composite_tensor_nonshape_params = ('loc', 'cov_operator')
