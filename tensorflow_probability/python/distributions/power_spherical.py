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
"""The Power Spherical distribution over vectors on the unit hypersphere."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.bijectors import square as square_bijector
from tensorflow_probability.python.distributions import beta as beta_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import spherical_uniform
from tensorflow_probability.python.distributions import von_mises_fisher
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.random import random_ops


__all__ = ['PowerSpherical']


class PowerSpherical(distribution.Distribution):
  r"""The Power Spherical distribution over unit vectors on `S^{n-1}`.

  The Power Spherical distribution [1] is a distribution over vectors
  on the unit hypersphere `S^{n-1}` embedded in `n` dimensions (`R^n`).

  It serves as an alternative to the von Mises-Fisher distribution with a
  simpler (faster) `log_prob` calculation, as well as a reparameterizable
  sampler. In contrast, the Power Spherical distribution does have
  `-mean_direction` as a point with zero density (and hence a neighborhood
  around that having arbitrarily small density), in contrast with the
  von Mises-Fisher distribution which has non-zero density everywhere.

  NOTE: `mean_direction` is not in general the mean of the distribution. For
  spherical distributions, the mean is generally not in the support of the
  distribution.

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, kappa) = C(kappa) (1 + mu^T x) ** k
  where,
  C(kappa) = 2**(a + b) pi**b Gamma(a) / Gamma(a + b)
  a = (n - 1) / 2. + k
  b = (n - 1) / 2.
  ```

  where:
  * `mean_direction = mu`; a unit vector in `R^k`,
  * `concentration = kappa`; scalar real >= 0, concentration of samples around
    `mean_direction`, where 0 pertains to the uniform distribution on the
    hypersphere, and \inf indicates a delta function at `mean_direction`.

  #### Examples

  A single instance of a PowerSpherical distribution is defined by a mean
  direction unit vector.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  tfd = tfp.distributions

  # Initialize a single 3-dimension PowerSpherical distribution.
  mu = [0., 1, 0]
  conc = 1.
  ps = tfd.PowerSpherical(mean_direction=mu, concentration=conc)

  # Evaluate this on an observation in S^2 (in R^3), returning a scalar.
  ps.prob([1., 0, 0])

  # Initialize a batch of two 3-variate vMF distributions.
  mu = [[0., 1, 0],
        [1., 0, 0]]
  conc = [1., 2]
  ps = tfd.PowerSpherical(mean_direction=mu, concentration=conc)

  # Evaluate this on two observations, each in S^2, returning a length two
  # tensor.
  x = [[0., 0, 1],
       [0., 1, 0]]
  ps.prob(x)
  ```

  #### References

  [1] Nicola de Cao, Wilker Aziz. The Power Spherical distribution.
      https://arxiv.org/abs/2006.04437.
  """

  def __init__(self,
               mean_direction,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='PowerSpherical'):
    """Creates a new `PowerSpherical` instance.

    Args:
      mean_direction: Floating-point `Tensor` with shape [B1, ... Bn, N].
        A unit vector indicating the mode of the distribution, or the
        unit-normalized direction of the mean.
      concentration: Floating-point `Tensor` having batch shape [B1, ... Bn]
        broadcastable with `mean_direction`. The level of concentration of
        samples around the `mean_direction`. `concentration=0` indicates a
        uniform distribution over the unit hypersphere, and `concentration=+inf`
        indicates a `Deterministic` distribution (delta function) at
        `mean_direction`.
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
      ValueError: For known-bad arguments, i.e. unsupported event dimension.
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype([mean_direction, concentration],
                                      tf.float32)
      self._mean_direction = tensor_util.convert_nonref_to_tensor(
          mean_direction, name='mean_direction', dtype=dtype)
      self._concentration = tensor_util.convert_nonref_to_tensor(
          concentration, name='concentration', dtype=dtype)

      super(PowerSpherical, self).__init__(
          dtype=self._concentration.dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          parameters=parameters,
          name=name)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
    return dict(
        mean_direction=parameter_properties.ParameterProperties(
            event_ndims=1,
            default_constraining_bijector_fn=parameter_properties
            .BIJECTOR_NOT_IMPLEMENTED),
        concentration=parameter_properties.ParameterProperties(
            shape_fn=lambda sample_shape: sample_shape[:-1],
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))))
    # pylint: enable=g-long-lambda

  @property
  def mean_direction(self):
    """Mean direction parameter."""
    return self._mean_direction

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  def _batch_shape_tensor(self, mean_direction=None, concentration=None):
    return ps.broadcast_shape(
        ps.shape(self.mean_direction if mean_direction is None
                 else mean_direction)[:-1],
        ps.shape(self.concentration if concentration is None
                 else concentration))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tensorshape_util.with_rank_at_least(self.mean_direction.shape, 1)[:-1],
        self.concentration.shape)

  def _event_shape_tensor(self, mean_direction=None):
    return ps.shape(self.mean_direction if mean_direction is None
                    else mean_direction)[-1:]

  def _event_shape(self):
    return tensorshape_util.with_rank(self.mean_direction.shape[-1:], rank=1)

  def _log_prob(self, x):
    concentration = tf.convert_to_tensor(self.concentration)
    return (self._log_unnormalized_prob(x, concentration=concentration) -
            self._log_normalization(concentration=concentration))

  def _log_unnormalized_prob(self, samples, concentration=None):
    if concentration is None:
      concentration = tf.convert_to_tensor(self.concentration)

    inner_product = tf.reduce_sum(samples * self.mean_direction, axis=-1)
    inner_product = tf.clip_by_value(inner_product, -1., 1.)
    return tf.math.xlog1py(concentration, inner_product)

  def _log_normalization(self, concentration=None, mean_direction=None):
    """Computes the log-normalizer of the distribution."""
    if concentration is None:
      concentration = tf.convert_to_tensor(self.concentration)
    event_size = tf.cast(self._event_shape_tensor(
        mean_direction=mean_direction)[-1], self.dtype)

    concentration1 = concentration + (event_size - 1.) / 2.
    concentration0 = (event_size - 1.) / 2.

    return ((concentration1 + concentration0) * np.log(2.) +
            concentration0 * np.log(np.pi) +
            tfp_math.log_gamma_difference(concentration0, concentration1))

  def _sample_control_dependencies(self, samples):
    """Check samples for proper shape and whether samples are unit vectors."""
    inner_sample_dim = samples.shape[-1]
    event_size = self.event_shape[-1]
    shape_msg = ('Samples must have innermost dimension matching that of '
                 '`self.mean_direction`.')
    if event_size is not None and inner_sample_dim is not None:
      if event_size != inner_sample_dim:
        raise ValueError(shape_msg)

    assertions = []
    if not self.validate_args:
      return assertions
    assertions.append(assert_util.assert_near(
        tf.cast(1., dtype=self.dtype),
        tf.linalg.norm(samples, axis=-1),
        message='Samples must be unit length.'))
    assertions.append(assert_util.assert_equal(
        tf.shape(samples)[-1:],
        self.event_shape_tensor(),
        message=shape_msg))
    return assertions

  def _mean(self):
    mean_direction = tf.convert_to_tensor(self.mean_direction)
    concentration = tf.convert_to_tensor(self.concentration)
    event_size = tf.cast(self._event_shape_tensor(
        mean_direction=mean_direction)[0], dtype=self.dtype)
    return (concentration / (
        event_size - 1. + concentration))[..., tf.newaxis] * mean_direction

  def _covariance(self):
    mean_direction = tf.convert_to_tensor(self.mean_direction)
    concentration = tf.convert_to_tensor(self.concentration)

    event_size = tf.cast(self._event_shape_tensor(
        mean_direction=mean_direction)[0], dtype=self.dtype)

    covariance = -concentration[..., tf.newaxis, tf.newaxis] * tf.linalg.matmul(
        mean_direction[..., tf.newaxis],
        mean_direction[..., tf.newaxis, :])
    covariance = tf.linalg.set_diag(
        covariance, tf.linalg.diag_part(covariance) + (
            concentration + event_size - 1.)[..., tf.newaxis])

    covariance = ((2 * concentration +  event_size - 1.)/ (
        tf.math.square(concentration + event_size - 1.) * (
            concentration + event_size)))[
                ..., tf.newaxis, tf.newaxis] * covariance
    return covariance

  def _sample_n(self, n, seed=None):
    mean_direction = tf.convert_to_tensor(self.mean_direction)
    concentration = tf.convert_to_tensor(self.concentration)
    event_size_int = self._event_shape_tensor(
        mean_direction=mean_direction)[0]
    event_size = tf.cast(event_size_int, dtype=self.dtype)

    beta_seed, uniform_seed = samplers.split_seed(seed, salt='power_spherical')

    broadcasted_concentration = tf.broadcast_to(
        concentration, self._batch_shape_tensor(
            mean_direction=mean_direction, concentration=concentration))
    beta = beta_lib.Beta(
        (event_size - 1.) / 2. + broadcasted_concentration,
        (event_size - 1.) / 2.)
    beta_samples = beta.sample(n, seed=beta_seed)

    u_shape = ps.concat([[n], self._batch_shape_tensor(
        mean_direction=mean_direction, concentration=concentration)], axis=0)

    spherical_samples = random_ops.spherical_uniform(
        shape=u_shape,
        dimension=event_size_int - 1,
        dtype=self.dtype,
        seed=uniform_seed)

    t = 2. * beta_samples - 1.
    y = tf.concat([
        t[..., tf.newaxis],
        tf.math.sqrt(1. - tf.math.square(t))[
            ..., tf.newaxis] * spherical_samples], axis=-1)

    u = tf.concat(
        [(1. - mean_direction[..., 0])[..., tf.newaxis],
         -mean_direction[..., 1:]], axis=-1)
    # Much like `VonMisesFisher`, we use `l2_normalize` which does
    # nothing if the zero vector is passed in, and thus the householder
    # reflection will do nothing.
    # This is consistent with sampling
    # with `mu = [1, 0, 0, ..., 0]` since samples will be of the
    # form: [w, sqrt(1 - w**2) * u] = w * mu + sqrt(1 - w**2) * v,
    # where:
    #   * `u` is a unit vector sampled from the unit hypersphere.
    #   * `v` is `[0, u]`.
    # This form is the same as sampling from the tangent-normal decomposition.
    u = tf.math.l2_normalize(u, axis=-1)
    return tf.math.l2_normalize(
        y - 2. * tf.math.reduce_sum(y * u, axis=-1, keepdims=True) * u, axis=-1)

  def _entropy(self, concentration=None, mean_direction=None):
    concentration = (
        tf.convert_to_tensor(self.concentration) if
        concentration is None else concentration)
    mean_direction = (
        tf.convert_to_tensor(self.mean_direction) if
        mean_direction is None else mean_direction)
    event_size = tf.cast(self._event_shape_tensor(
        mean_direction=mean_direction)[-1], self.dtype)
    concentration1 = concentration + (event_size - 1.) / 2.
    concentration0 = (event_size - 1.) / 2.
    entropy = (self._log_normalization(
        concentration=concentration, mean_direction=mean_direction) -
               concentration * (
                   np.log(2.) + tf.math.digamma(concentration1) -
                   tf.math.digamma(concentration1 + concentration0)))
    return tf.broadcast_to(
        entropy, self._batch_shape_tensor(
            mean_direction=mean_direction, concentration=concentration))

  def _default_event_space_bijector(self):
    # TODO(b/145620027) Finalize choice of bijector.
    return chain_bijector.Chain([
        invert_bijector.Invert(
            square_bijector.Square(validate_args=self.validate_args),
            validate_args=self.validate_args),
        softmax_centered_bijector.SoftmaxCentered(
            validate_args=self.validate_args)
    ], validate_args=self.validate_args)

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args:
      return []
    mean_direction = tf.convert_to_tensor(self.mean_direction)
    concentration = tf.convert_to_tensor(self.concentration)

    assertions = []
    if is_init != tensor_util.is_ref(self._mean_direction):
      assertions.append(
          assert_util.assert_greater(
              tf.shape(mean_direction)[-1],
              1,
              message='`mean_direction` must be a vector of at least size 2.'))
      assertions.append(
          assert_util.assert_near(
              tf.cast(1., self.dtype),
              tf.linalg.norm(mean_direction, axis=-1),
              message='`mean_direction` must be unit-length'))
    if is_init != tensor_util.is_ref(self._concentration):
      assertions.append(
          assert_util.assert_non_negative(
              concentration, message='`concentration` must be non-negative'))
    return assertions


@kullback_leibler.RegisterKL(PowerSpherical, spherical_uniform.SphericalUniform)
def _kl_power_uniform_spherical(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b).

  Args:
    a: instance of a PowerSpherical distribution object.
    b: instance of a SphericalUniform distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_power_uniform_spherical".

  Returns:
    Batchwise KL(a || b)

  Raises:
    ValueError: If the two distributions are over spheres of different
      dimensions.

  #### References

  [1] Nicola de Cao, Wilker Aziz. The Power Spherical distribution.
      https://arxiv.org/abs/2006.04437.
  """
  with tf.name_scope(name or 'kl_power_uniform_spherical'):
    msg = (
        'Can not compute the KL divergence between a `PowerSpherical` and '
        '`SphericalUniform` of different dimensions.')
    deps = []
    if a.event_shape[-1] is not None:
      if a.event_shape[-1] != b.dimension:
        raise ValueError(
            (msg + 'Got {} vs. {}').format(a.event_shape[-1], b.dimension))
    elif a.validate_args or b.validate_args:
      deps += [assert_util.assert_equal(
          a.event_shape_tensor()[-1], b.dimension, message=msg)]

    with tf.control_dependencies(deps):
      return b.entropy() - a.entropy()


@kullback_leibler.RegisterKL(PowerSpherical, von_mises_fisher.VonMisesFisher)
def _kl_power_spherical_vmf(a, b, name=None):
  """Calculate the batched KL divergence KL(a || b).

  Args:
    a: instance of a PowerSpherical distribution object.
    b: instance of a VonMisesFisher distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_power_spherical_vmf".

  Returns:
    Batchwise KL(a || b)

  Raises:
    ValueError: If the two distributions are over spheres of different
      dimensions.

  #### References

  [1] Nicola de Cao, Wilker Aziz. The Power Spherical distribution.
      https://arxiv.org/abs/2006.04437.
  """
  with tf.name_scope(name or 'kl_power_spherical_vmf'):
    msg = (
        'Can not compute the KL divergence between a `PowerSpherical` and '
        '`VonMisesFisher` of different dimensions.')
    deps = []
    if a.event_shape[-1] is not None and b.event_shape[-1] is not None:
      if a.event_shape[-1] != b.event_shape[-1]:
        raise ValueError(
            (msg + 'Got {} vs. {}').format(
                a.event_shape[-1], b.event_shape[-1]))
    elif a.validate_args or b.validate_args:
      deps += [assert_util.assert_equal(
          a.event_shape_tensor()[-1], b.event_shape_tensor()[-1], message=msg)]

    with tf.control_dependencies(deps):
      a_mean_direction = tf.convert_to_tensor(a.mean_direction)
      a_concentration = tf.convert_to_tensor(a.concentration)
      b_mean_direction = tf.convert_to_tensor(b.mean_direction)
      b_concentration = tf.convert_to_tensor(b.concentration)

      event_size = tf.cast(a._event_shape_tensor(  # pylint:disable=protected-access
          mean_direction=a_mean_direction)[-1], a.dtype)
      kl = (-a._entropy(concentration=a_concentration,  # pylint:disable=protected-access
                        mean_direction=a_mean_direction) +
            b._log_normalization(  # pylint:disable=protected-access
                concentration=b_concentration) -
            a_concentration * b_concentration * tf.reduce_sum(
                a_mean_direction * b_mean_direction, axis=-1) / (
                    a_concentration + event_size - 1.))
      return kl
