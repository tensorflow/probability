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
"""The von Mises-Fisher distribution over vectors on the unit hypersphere."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain as chain_bijector
from tensorflow_probability.python.bijectors import invert as invert_bijector
from tensorflow_probability.python.bijectors import softmax_centered as softmax_centered_bijector
from tensorflow_probability.python.bijectors import square as square_bijector
from tensorflow_probability.python.distributions import beta as beta_lib
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util


__all__ = ['VonMisesFisher']


def _bessel_ive(v, z, cache=None):
  """Computes I_v(z)*exp(-abs(z)) using a recurrence relation, where z > 0."""
  # TODO(b/67497980): Switch to a more numerically faithful implementation.
  z = tf.convert_to_tensor(z)

  wrap = lambda result: tf.debugging.check_numerics(result, 'besseli{}'.format(v
                                                                              ))

  if float(v) >= 2:
    raise ValueError(
        'Evaluating bessel_i by recurrence becomes imprecise for large v')

  cache = cache or {}
  safe_z = tf.where(z > 0, z, tf.ones_like(z))
  if v in cache:
    return wrap(cache[v])
  if v == 0:
    cache[v] = tf.math.bessel_i0e(z)
  elif v == 1:
    cache[v] = tf.math.bessel_i1e(z)
  elif v == 0.5:
    # sinh(x)*exp(-abs(x)), sinh(x) = (e^x - e^{-x}) / 2
    sinhe = lambda x: (tf.exp(x - tf.abs(x)) - tf.exp(-x - tf.abs(x))) / 2
    cache[v] = (
        np.sqrt(2 / np.pi) * sinhe(z) *
        tf.where(z > 0, tf.math.rsqrt(safe_z), tf.ones_like(safe_z)))
  elif v == -0.5:
    # cosh(x)*exp(-abs(x)), cosh(x) = (e^x + e^{-x}) / 2
    coshe = lambda x: (tf.exp(x - tf.abs(x)) + tf.exp(-x - tf.abs(x))) / 2
    cache[v] = (
        np.sqrt(2 / np.pi) * coshe(z) *
        tf.where(z > 0, tf.math.rsqrt(safe_z), tf.ones_like(safe_z)))
  if v <= 1:
    return wrap(cache[v])
  # Recurrence relation:
  cache[v] = (_bessel_ive(v - 2, z, cache) -
              (2 * (v - 1)) * _bessel_ive(v - 1, z, cache) / z)
  return wrap(cache[v])


class VonMisesFisher(distribution.Distribution):
  r"""The von Mises-Fisher distribution over unit vectors on `S^{n-1}`.

  The von Mises-Fisher distribution is a directional distribution over vectors
  on the unit hypersphere `S^{n-1}` embedded in `n` dimensions (`R^n`).

  #### Mathematical details

  The probability density function (pdf) is,

  ```none
  pdf(x; mu, kappa) = C(kappa) exp(kappa * mu^T x)
  where,
  C(kappa) = (2 pi)^{-n/2} kappa^{n/2-1} / I_{n/2-1}(kappa),
  I_v(z) being the modified Bessel function of the first kind of order v
  ```

  where:
  * `mean_direction = mu`; a unit vector in `R^k`,
  * `concentration = kappa`; scalar real >= 0, concentration of samples around
    `mean_direction`, where 0 pertains to the uniform distribution on the
    hypersphere, and \inf indicates a delta function at `mean_direction`.


  NOTE: Currently only n in {2, 3, 4, 5} are supported. For n=5 some numerical
  instability can occur for low concentrations (<.01).

  NOTE: `mean_direction` is not in general the mean of the distribution. For
  spherical distributions, the mean is generally not in the support of the
  distribution.

  #### Examples

  A single instance of a vMF distribution is defined by a mean direction (or
  mode) unit vector and a scalar concentration parameter.

  Extra leading dimensions, if provided, allow for batches.

  ```python
  tfd = tfp.distributions

  # Initialize a single 3-dimension vMF distribution.
  mu = [0., 1, 0]
  conc = 1.
  vmf = tfd.VonMisesFisher(mean_direction=mu, concentration=conc)

  # Evaluate this on an observation in S^2 (in R^3), returning a scalar.
  vmf.prob([1., 0, 0])

  # Initialize a batch of two 3-variate vMF distributions.
  mu = [[0., 1, 0],
        [1., 0, 0]]
  conc = [1., 2]
  vmf = tfd.VonMisesFisher(mean_direction=mu, concentration=conc)

  # Evaluate this on two observations, each in S^2, returning a length two
  # tensor.
  x = [[0., 0, 1],
       [0., 1, 0]]
  vmf.prob(x)
  ```
  """

  def __init__(self,
               mean_direction,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name='VonMisesFisher'):
    """Creates a new `VonMisesFisher` instance.

    Args:
      mean_direction: Floating-point `Tensor` with shape [B1, ... Bn, D].
        A unit vector indicating the mode of the distribution, or the
        unit-normalized direction of the mean. NOTE: `D` is currently
        restricted to <= 5.
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

      static_event_dim = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(
              self._mean_direction.shape, 1)[-1])
      if static_event_dim is not None and static_event_dim > 5:
        raise ValueError('von Mises-Fisher ndims > 5 is not currently '
                         'supported')

      # mean_direction is always reparameterized.
      # concentration is only for event_dim==3, via an inversion sampler.
      reparameterization_type = (
          reparameterization.FULLY_REPARAMETERIZED
          if static_event_dim == 3 else
          reparameterization.NOT_REPARAMETERIZED)
      super(VonMisesFisher, self).__init__(
          dtype=self._concentration.dtype,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          reparameterization_type=reparameterization_type,
          parameters=parameters,
          name=name)

  @classmethod
  def _params_event_ndims(cls):
    return dict(mean_direction=1, concentration=0)

  @property
  def mean_direction(self):
    """Mean direction parameter."""
    return self._mean_direction

  @property
  def concentration(self):
    """Concentration parameter."""
    return self._concentration

  def _batch_shape_tensor(self, mean_direction=None, concentration=None):
    return tf.broadcast_dynamic_shape(
        tf.shape(self.mean_direction if mean_direction is None
                 else mean_direction)[:-1],
        tf.shape(self.concentration if concentration is None
                 else concentration))

  def _batch_shape(self):
    return tf.broadcast_static_shape(
        tensorshape_util.with_rank_at_least(self.mean_direction.shape, 1)[:-1],
        self.concentration.shape)

  def _event_shape_tensor(self, mean_direction=None):
    return tf.shape(self.mean_direction if mean_direction is None
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

    bcast_mean_dir = (self.mean_direction +
                      tf.zeros_like(concentration)[..., tf.newaxis])
    inner_product = tf.reduce_sum(samples * bcast_mean_dir, axis=-1)
    return concentration * inner_product

  def _log_normalization(self, concentration=None):
    """Computes the log-normalizer of the distribution."""
    if concentration is None:
      concentration = tf.convert_to_tensor(self.concentration)

    event_dim = tf.compat.dimension_value(self.event_shape[0])
    if event_dim is None:
      raise ValueError('von Mises-Fisher _log_normalizer currently only '
                       'supports statically known event shape')
    safe_conc = tf.where(concentration > 0, concentration,
                         tf.ones_like(concentration))
    safe_lognorm = ((event_dim / 2 - 1) * tf.math.log(safe_conc) -
                    (event_dim / 2) * np.log(2 * np.pi) -
                    tf.math.log(_bessel_ive(event_dim / 2 - 1, safe_conc)) -
                    tf.abs(safe_conc))
    log_nsphere_surface_area = (
        np.log(2.) + (event_dim / 2) * np.log(np.pi) -
        tf.math.lgamma(tf.cast(event_dim / 2, self.dtype)))
    return tf.where(concentration > 0, -safe_lognorm,
                    log_nsphere_surface_area)

  # TODO(bjp): Odd dimension analytic CDFs are provided in [1]
  # [1]: https://ieeexplore.ieee.org/document/7347705/

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
        1.,
        tf.linalg.norm(samples, axis=-1),
        message='Samples must be unit length.'))
    assertions.append(assert_util.assert_equal(
        tf.shape(samples)[-1:],
        self.event_shape_tensor(),
        message=shape_msg))
    return assertions

  def _mode(self):
    """The mode of the von Mises-Fisher distribution is the mean direction."""
    return (self.mean_direction +
            tf.zeros_like(self.concentration)[..., tf.newaxis])

  def _mean(self):
    # Derivation: https://sachinruk.github.io/blog/von-Mises-Fisher/
    concentration = tf.convert_to_tensor(self.concentration)
    mean_direction = tf.convert_to_tensor(self.mean_direction)

    event_dim = tf.compat.dimension_value(self.event_shape[0])
    if event_dim is None:
      raise ValueError('event shape must be statically known for _bessel_ive')
    safe_conc = tf.where(concentration > 0, concentration,
                         tf.ones_like(concentration))
    safe_mean = mean_direction * (
        _bessel_ive(event_dim / 2, safe_conc) /
        _bessel_ive(event_dim / 2 - 1, safe_conc))[..., tf.newaxis]
    return tf.where(
        concentration[..., tf.newaxis] > 0.,
        safe_mean, tf.zeros_like(safe_mean))

  def _covariance(self):
    # Derivation: https://sachinruk.github.io/blog/von-Mises-Fisher/
    event_dim = tf.compat.dimension_value(self.event_shape[0])
    if event_dim is None:
      raise ValueError('event shape must be statically known for _bessel_ive')
    # TODO(b/141142878): Enable this; numerically unstable.
    if event_dim > 2:
      raise NotImplementedError(
          'vMF covariance is numerically unstable for dim>2')
    mean_direction = tf.convert_to_tensor(self.mean_direction)
    concentration = tf.convert_to_tensor(self.concentration)
    safe_conc = tf.where(concentration > 0, concentration,
                         tf.ones_like(concentration))[..., tf.newaxis]
    h = (_bessel_ive(event_dim / 2, safe_conc) /
         _bessel_ive(event_dim / 2 - 1, safe_conc))
    intermediate = (
        tf.matmul(mean_direction[..., :, tf.newaxis],
                  mean_direction[..., tf.newaxis, :]) *
        (1 - event_dim * h / safe_conc - h**2)[..., tf.newaxis])
    cov = tf.linalg.set_diag(
        intermediate,
        tf.linalg.diag_part(intermediate) + (h / safe_conc))
    return tf.where(
        concentration[..., tf.newaxis, tf.newaxis] > 0., cov,
        tf.linalg.eye(event_dim,
                      batch_shape=self._batch_shape_tensor(
                          mean_direction=mean_direction,
                          concentration=concentration)) / event_dim)

  def _rotate(self, samples, mean_direction):
    """Applies a Householder rotation to `samples`."""
    event_dim = (
        tf.compat.dimension_value(self.event_shape[0]) or
        self._event_shape_tensor(mean_direction=mean_direction)[0])
    basis = tf.concat([[1.], tf.zeros([event_dim - 1], dtype=self.dtype)],
                      axis=0)
    u = tf.math.l2_normalize(basis - mean_direction, axis=-1)
    return samples - 2 * tf.reduce_sum(samples * u, axis=-1, keepdims=True) * u

  def _sample_3d(self, n, mean_direction, concentration, seed=None):
    """Specialized inversion sampler for 3D."""
    u_shape = tf.concat([[n], self._batch_shape_tensor(
        mean_direction=mean_direction, concentration=concentration)], axis=0)
    z = samplers.uniform(u_shape, seed=seed, dtype=self.dtype)
    # TODO(bjp): Higher-order odd dim analytic CDFs are available in [1], could
    # be bisected for bounded sampling runtime (i.e. not rejection sampling).
    # [1]: Inversion sampler via: https://ieeexplore.ieee.org/document/7347705/
    # The inversion is: u = 1 + log(z + (1-z)*exp(-2*kappa)) / kappa
    # We must protect against both kappa and z being zero.
    safe_conc = tf.where(concentration > 0, concentration,
                         tf.ones_like(concentration))
    safe_z = tf.where(z > 0, z, tf.ones_like(z))
    safe_u = 1 + tf.reduce_logsumexp(
        [tf.math.log(safe_z),
         tf.math.log1p(-safe_z) - 2 * safe_conc], axis=0) / safe_conc
    # Limit of the above expression as kappa->0 is 2*z-1
    u = tf.where(concentration > 0., safe_u, 2 * z - 1)
    # Limit of the expression as z->0 is -1.
    u = tf.where(tf.equal(z, 0), -tf.ones_like(u), u)
    if not self._allow_nan_stats:
      u = tf.debugging.check_numerics(u, 'u in _sample_3d')
    return u[..., tf.newaxis]

  def _sample_n(self, n, seed=None):
    dim0_seed, otherdims_seed = samplers.split_seed(seed,
                                                    salt='von_mises_fisher')
    # The sampling strategy relies on the fact that vMF variates are symmetric
    # about the mean direction. Accordingly, if we have a sampling strategy for
    # the away-from-mean angle, then we can uniformly sample the remaining
    # dimensions on the S^{dim-2} sphere for , and rotate these samples from a
    # (1, 0, 0, ..., 0)-mode distribution into the target orientation.
    #
    # This is easy to imagine on the 1-sphere (S^1; in 2-D space): sample a
    # von-Mises distributed `x` value in [-1, 1], then uniformly select what
    # amounts to a "up" or "down" additional degree of freedom after unit
    # normalizing, followed by a final rotation to the desired mean direction
    # from a basis of (1, 0).
    #
    # On S^2 (in 3-D), selecting a vMF `x` identifies a circle in `yz` on the
    # unit sphere over which the distribution is uniform, in particular the
    # circle where x = \hat{x} intersects the unit sphere. We pick a point on
    # that circle, then rotate to the desired mean direction from a basis of
    # (1, 0, 0).
    mean_direction = tf.convert_to_tensor(self.mean_direction)
    concentration = tf.convert_to_tensor(self.concentration)
    event_dim = (
        tf.compat.dimension_value(self.event_shape[0]) or
        self._event_shape_tensor(mean_direction=mean_direction)[0])

    sample_batch_shape = tf.concat([[n], self._batch_shape_tensor(
        mean_direction=mean_direction, concentration=concentration)], axis=0)
    dim = tf.cast(event_dim - 1, self.dtype)
    if event_dim == 3:
      samples_dim0 = self._sample_3d(n,
                                     mean_direction=mean_direction,
                                     concentration=concentration,
                                     seed=dim0_seed)
    else:
      # Wood'94 provides a rejection algorithm to sample the x coordinate.
      # Wood'94 definition of b:
      # b = (-2 * kappa + tf.sqrt(4 * kappa**2 + dim**2)) / dim
      # https://stats.stackexchange.com/questions/156729 suggests:
      b = dim / (2 * concentration +
                 tf.sqrt(4 * concentration**2 + dim**2))
      # TODO(bjp): Integrate any useful numerical tricks from hyperspherical VAE
      #     https://github.com/nicola-decao/s-vae-tf/
      x = (1 - b) / (1 + b)
      c = concentration * x + dim * tf.math.log1p(-x**2)
      beta = beta_lib.Beta(dim / 2, dim / 2)

      def cond_fn(w, should_continue, seed):
        del w, seed
        return tf.reduce_any(should_continue)

      def body_fn(w, should_continue, seed):
        """While loop body for sampling the angle `w`."""
        beta_seed, unif_seed, next_seed = samplers.split_seed(seed, n=3)
        z = beta.sample(sample_shape=sample_batch_shape, seed=beta_seed)
        # set_shape needed here because of b/139013403
        tensorshape_util.set_shape(z, w.shape)
        w = tf.where(should_continue,
                     (1. - (1. + b) * z) / (1. - (1. - b) * z),
                     w)
        if not self.allow_nan_stats:
          w = tf.debugging.check_numerics(w, 'w')
        unif = samplers.uniform(
            sample_batch_shape, seed=unif_seed, dtype=self.dtype)
        # set_shape needed here because of b/139013403
        tensorshape_util.set_shape(unif, w.shape)
        should_continue = should_continue & (
            concentration * w + dim * tf.math.log1p(-x * w) - c <
            # Use log1p(-unif) to prevent log(0) and ensure that log(1) is
            # possible.
            tf.math.log1p(-unif))
        return w, should_continue, next_seed

      w = tf.zeros(sample_batch_shape, dtype=self.dtype)
      should_continue = tf.ones(sample_batch_shape, dtype=tf.bool)
      samples_dim0, _, _ = tf.while_loop(
          cond=cond_fn, body=body_fn,
          loop_vars=(w, should_continue, dim0_seed))
      samples_dim0 = samples_dim0[..., tf.newaxis]
    if not self._allow_nan_stats:
      # Verify samples are w/in -1, 1, with useful error output tensors (top
      # value rather than all values).
      with tf.control_dependencies([
          assert_util.assert_less_equal(
              samples_dim0,
              dtype_util.as_numpy_dtype(self.dtype)(1.01)),
          assert_util.assert_greater_equal(
              samples_dim0,
              dtype_util.as_numpy_dtype(self.dtype)(-1.01)),
      ]):
        samples_dim0 = tf.identity(samples_dim0)
    samples_otherdims_shape = tf.concat([sample_batch_shape, [event_dim - 1]],
                                        axis=0)
    unit_otherdims = tf.math.l2_normalize(
        samplers.normal(
            samples_otherdims_shape, seed=otherdims_seed, dtype=self.dtype),
        axis=-1)
    samples = tf.concat([
        samples_dim0,  # we must avoid sqrt(1 - (>1)**2)
        tf.sqrt(tf.maximum(1 - samples_dim0**2, 0.)) * unit_otherdims
    ], axis=-1)
    samples = tf.math.l2_normalize(samples, axis=-1)
    if not self.allow_nan_stats:
      samples = tf.debugging.check_numerics(samples, 'samples')

    # Runtime assert that samples are unit length.
    if not self.allow_nan_stats:
      worst, _ = tf.math.top_k(
          tf.reshape(tf.abs(1 - tf.linalg.norm(samples, axis=-1)), [-1]))
      with tf.control_dependencies([
          assert_util.assert_near(
              dtype_util.as_numpy_dtype(self.dtype)(0),
              worst,
              atol=1e-4,
              summarize=100)
      ]):
        samples = tf.identity(samples)
    # The samples generated are symmetric around a mode at (1, 0, 0, ...., 0).
    # Now, we move the mode to `self.mean_direction` using a rotation matrix.
    if not self.allow_nan_stats:
      # Assert that the basis vector rotates to the mean direction, as expected.
      basis = tf.cast(tf.concat([[1.], tf.zeros([event_dim - 1])], axis=0),
                      self.dtype)
      with tf.control_dependencies([
          assert_util.assert_less(
              tf.linalg.norm(
                  self._rotate(basis, mean_direction=mean_direction) -
                  mean_direction, axis=-1),
              dtype_util.as_numpy_dtype(self.dtype)(1e-5))
      ]):
        return self._rotate(samples, mean_direction=mean_direction)
    return self._rotate(samples, mean_direction=mean_direction)

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
              message='`mean_direction` may not have scalar event shape'))
      assertions.append(
          assert_util.assert_less_equal(
              tf.shape(mean_direction)[-1],
              5,
              message='von Mises-Fisher ndims > 5 is not currently supported'))
      assertions.append(
          assert_util.assert_near(
              1.,
              tf.linalg.norm(mean_direction, axis=-1),
              message='`mean_direction` must be unit-length'))
    if is_init != tensor_util.is_ref(self._concentration):
      assertions.append(
          assert_util.assert_non_negative(
              concentration, message='`concentration` must be non-negative'))
    return assertions
