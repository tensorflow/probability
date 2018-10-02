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
"""Linear Gaussian State Space Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf


from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.distributions import seed_stream

from tensorflow_probability.python.internal import distribution_util as util
from tensorflow_probability.python.internal import reparameterization
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops.linalg import linear_operator_util

tfl = tf.linalg

# The built-in tf.matmul doesn't broadcast batch dimensions, so we
# need to use `matmul_with_broadcast` throughout to ensure we support
# batching.
_matmul = linear_operator_util.matmul_with_broadcast


def _broadcast_to_shape(x, shape):
  return x + tf.zeros(shape=shape, dtype=x.dtype)


def _check_equal_shape(name,
                       static_shape,
                       dynamic_shape,
                       static_target_shape,
                       dynamic_target_shape=None):
  """Check that source and target shape match, statically if possible."""

  static_target_shape = tf.TensorShape(static_target_shape)
  if static_shape.is_fully_defined() and static_target_shape.is_fully_defined():
    if static_shape != static_target_shape:
      raise ValueError("{}: required shape {} but found {}".
                       format(name, static_target_shape, static_shape))
    return None
  else:
    if dynamic_target_shape is None:
      if static_target_shape.is_fully_defined():
        dynamic_target_shape = static_target_shape.as_list()
      else:
        raise ValueError("{}: cannot infer target shape: no dynamic shape "
                         "specified and static shape {} is not fully defined".
                         format(name, static_target_shape))
    return tf.assert_equal(dynamic_shape, dynamic_target_shape,
                           message=("{}: required shape {}".
                                    format(name, static_target_shape)))


def _augment_sample_shape(partial_batch_dist,
                          full_sample_and_batch_shape,
                          validate_args=False):
  """Augment a sample shape to broadcast batch dimensions.

  Computes an augmented sample shape, so that any batch dimensions not
  part of the distribution `partial_batch_dist` are treated as identical
  distributions.

  # partial_batch_dist.batch_shape  = [      7]
  # full_sample_and_batch_shape     = [3, 4, 7]
  # => return an augmented sample shape of [3, 4] so that
  #    partial_batch_dist.sample(augmented_sample_shape) has combined
  #    sample and batch shape of [3, 4, 7].

  Args:
    partial_batch_dist: `tfd.Distribution` instance with batch shape a
      prefix of `full_sample_and_batch_shape`.
    full_sample_and_batch_shape: a Tensor or Tensor-like shape.
    validate_args: if True, check for shape errors at runtime.
  Returns:
    augmented_sample_shape: sample shape such that
      `partial_batch_dist.sample(augmented_sample_shape)` has combined
      sample and batch shape of `full_sample_and_batch_shape`.

  Raises:
    ValueError: if `partial_batch_dist.batch_shape` has more dimensions than
      `full_sample_and_batch_shape`.
    NotImplementedError: if broadcasting would be required to make
      `partial_batch_dist.batch_shape` into a prefix of
      `full_sample_and_batch_shape` .
  """
  full_ndims = util.prefer_static_shape(full_sample_and_batch_shape)[0]
  partial_batch_ndims = (partial_batch_dist.batch_shape.ndims
                         if partial_batch_dist.batch_shape.ndims is not None
                         else util.prefer_static_shape(
                             partial_batch_dist.batch_shape_tensor())[0])

  num_broadcast_dims = full_ndims - partial_batch_ndims

  expected_partial_batch_shape = (
      full_sample_and_batch_shape[num_broadcast_dims:])
  expected_partial_batch_shape_static = util.static_value(
      full_sample_and_batch_shape[num_broadcast_dims:])

  # Raise errors statically if possible.
  num_broadcast_dims_static = util.static_value(num_broadcast_dims)
  if num_broadcast_dims_static is not None:
    if num_broadcast_dims_static < 0:
      raise ValueError("Cannot broadcast distribution {} batch shape to "
                       "target batch shape with fewer dimensions"
                       .format(partial_batch_dist))
  if (expected_partial_batch_shape_static is not None and
      partial_batch_dist.batch_shape.is_fully_defined()):
    if (partial_batch_dist.batch_shape and
        any(expected_partial_batch_shape_static !=
            partial_batch_dist.batch_shape.as_list())):
      raise NotImplementedError("Broadcasting is not supported; "
                                "unexpected batch shape "
                                "(expected {}, saw {}).".format(
                                    expected_partial_batch_shape_static,
                                    partial_batch_dist.batch_shape
                                ))
  runtime_assertions = []
  if validate_args:
    runtime_assertions.append(tf.assert_greater_equal(
        tf.convert_to_tensor(
            num_broadcast_dims,
            dtype=tf.int32),
        tf.zeros((), dtype=tf.int32), message=(
            "Cannot broadcast distribution {} batch shape to "
            "target batch shape with fewer dimensions.".
            format(partial_batch_dist))))
    runtime_assertions.append(tf.assert_equal(
        expected_partial_batch_shape,
        partial_batch_dist.batch_shape_tensor(),
        message=("Broadcasting is not supported; "
                 "unexpected batch shape."),
        name="assert_batch_shape_same"))

  with tf.control_dependencies(runtime_assertions):
    return full_sample_and_batch_shape[:num_broadcast_dims]


class LinearGaussianStateSpaceModel(distribution.Distribution):
  """Observation distribution from a linear Gaussian state space model.

  The state space model, sometimes called a Kalman filter, posits a
  latent state vector `z_t` of dimension `latent_size` that evolves
  over time following linear Gaussian transitions,

  ```z_{t+1} = F * z_t + N(b; Q)```

  for transition matrix `F`, bias `b` and covariance matrix
  `Q`. At each timestep, we observe a noisy projection of the
  latent state `x_t = H * z_t + N(c; R)`. The transition and
  observation models may be fixed or may vary between timesteps.

  This Distribution represents the marginal distribution on
  observations, `p(x)`. The marginal `log_prob` is computed by
  Kalman filtering [1], and `sample` by an efficient forward
  recursion. Both operations require time linear in `T`, the total
  number of timesteps.

  #### Shapes

  The event shape is `[num_timesteps, observation_size]`, where
  `observation_size` is the dimension of each observation `x_t`.
  The observation and transition models must return consistent
  shapes.

  This implementation supports vectorized computation over a batch of
  models. All of the parameters (prior distribution, transition and
  observation operators and noise models) must have a consistent
  batch shape.

  #### Time-varying processes

  Any of the model-defining parameters (prior distribution, transition
  and observation operators and noise models) may be specified as a
  callable taking an integer timestep `t` and returning a
  time-dependent value. The dimensionality (`latent_size` and
  `observation_size`) must be the same at all timesteps.

  Importantly, the timestep is passed as a `Tensor`, not a Python
  integer, so any conditional behavior must occur *inside* the
  TensorFlow graph. For example, suppose we want to use a different
  transition model on even days than odd days. It does *not* work to
  write

  ```python
  def transition_matrix(t):
    if t % 2 == 0:
      return even_day_matrix
    else:
      return odd_day_matrix
  ```

  since the value of `t` is not fixed at graph-construction
  time. Instead we need to write

  ```python
  def transition_matrix(t):
    return tf.cond(tf.equal(tf.mod(t, 2), 0),
                   lambda : even_day_matrix,
                   lambda : odd_day_matrix)
  ```

  so that TensorFlow can switch between operators appropriately at
  runtime.

  #### Examples

  Consider a simple tracking model. The two-dimensional latent state
  represents the true position of a vehicle, and at each timestep we
  see a noisy observation of this position (e.g., a GPS reading). The
  vehicle is assumed to move by a random walk with standard deviation
  `step_std` at each step, and observation noise level `std`. We build
  the distribution over noisy observations by

  ```python
  ndims = 2
  step_std = 1.0
  noise_std = 5.0
  model = LinearGaussianStateSpaceModel(
    num_timesteps=100,
    transition_matrix=tfl.LinearOperatorIdentity(ndims),
    transition_noise=tfd.MultivariateNormalDiag(
     scale_diag=step_std**2 * tf.ones([ndims])),
    observation_matrix=tfl.LinearOperatorIdentity(ndims),
    observation_noise=tfd.MultivariateNormalDiag(
     scale_diag=noise_std**2 * tf.ones([ndims])),
    initial_state_prior=tfd.MultivariateNormalDiag(
     scale_diag=tf.ones([ndims])))
  )
  ```

  using the identity matrix for the transition and observation
  operators. We can then use this model to generate samples,
  compute marginal likelihood of observed sequences, and
  perform posterior inference.

  ```python
  x = model.sample(5) # Sample from the prior on sequences of observations.
  lp = model.log_prob(x) # Marginal likelihood of a (batch of) observations.

  # Compute the filtered posterior on latent states given observations,
  # and extract the mean and covariance for the current (final) timestep.
  _, filtered_means, filtered_covs, _, _ = model.forward_filter(x)
  final_step = tfd.MultivariateNormalFullCovariance(
                loc=filtered_means[..., -1, :],
                scale=filtered_covs[..., -1, :])
  ```

  * TODO(davmre): implement and describe full posterior inference / smoothing.

  * TODO(davmre): show example of fitting parameters.
  """

  def __init__(self,
               num_timesteps,
               transition_matrix,
               transition_noise,
               observation_matrix,
               observation_noise,
               initial_state_prior,
               initial_step=0,
               validate_args=False,
               allow_nan_stats=True,
               name="LinearGaussianStateSpaceModel"):
    """Initialize a `LinearGaussianStateSpaceModel.

    Args:
      num_timesteps: Python `int` total number of timesteps.
      transition_matrix: A transition operator, represented by a Tensor or
        LinearOperator of shape `[latent_size, latent_size]`, or by a
        callable taking as argument a scalar integer Tensor `t` and
        returning a Tensor or LinearOperator representing the transition
        operator from latent state at time `t` to time `t + 1`.
      transition_noise: An instance of
        `tfd.MultivariateNormalLinearOperator` with event shape
        `[latent_size]`, representing the mean and covariance of the
        transition noise model, or a callable taking as argument a
        scalar integer Tensor `t` and returning such a distribution
        representing the noise in the transition from time `t` to time `t + 1`.
      observation_matrix: An observation operator, represented by a Tensor
        or LinearOperator of shape `[observation_size, latent_size]`,
        or by a callable taking as argument a scalar integer Tensor
        `t` and returning a timestep-specific Tensor or
        LinearOperator.
      observation_noise: An instance of
        `tfd.MultivariateNormalLinearOperator` with event shape
        `[observation_size]`, representing the mean and covariance of
        the observation noise model, or a callable taking as argument
        a scalar integer Tensor `t` and returning a timestep-specific
        noise model.
      initial_state_prior: An instance of `MultivariateNormalLinearOperator`
        representing the prior distribution on latent states; must
        have event shape `[latent_size]`.
      initial_step: optional `int` specifying the time of the first
        modeled timestep.  This is added as an offset when passing
        timesteps `t` to (optional) callables specifying
        timestep-specific transition and observation models.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
      allow_nan_stats: Python `bool`, default `True`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.
    """

    parameters = locals()

    with tf.name_scope(name, values=[num_timesteps,
                                     transition_matrix,
                                     transition_noise,
                                     observation_matrix,
                                     observation_noise,
                                     initial_state_prior,
                                     initial_step]) as name:

      self.num_timesteps = num_timesteps
      self.initial_state_prior = initial_state_prior
      self.initial_step = initial_step
      self.final_step = self.initial_step + self.num_timesteps

      # TODO(b/78475680): Friendly dtype inference.
      dtype = initial_state_prior.dtype

      # Internally, the transition and observation matrices are
      # canonicalized as callables returning a LinearOperator. This
      # creates no overhead when the model is actually fixed, since in
      # that case we simply build the trivial callable that returns
      # the same matrix at each timestep.
      def _maybe_make_linop(x, is_square=None, name=None):
        """Converts Tensors into LinearOperators."""
        if not isinstance(x, tfl.LinearOperator):
          x = tfl.LinearOperatorFullMatrix(
              tf.convert_to_tensor(x, dtype=dtype),
              is_square=is_square,
              name=name)
        return x
      def _maybe_make_callable_from_linop(f, name, make_square_linop=None):
        """Converts fixed objects into trivial callables."""
        if not callable(f):
          linop = _maybe_make_linop(f, is_square=make_square_linop, name=name)
          f = lambda t: linop
        return f
      self.get_transition_matrix_for_timestep = (
          _maybe_make_callable_from_linop(
              transition_matrix,
              name="transition_matrix",
              make_square_linop=True))
      self.get_observation_matrix_for_timestep = (
          _maybe_make_callable_from_linop(
              observation_matrix, name="observation_matrix"))

      # Similarly, we canonicalize the transition and observation
      # noise models as callables returning a
      # tfd.MultivariateNormalLinearOperator distribution object.
      def _maybe_make_callable(f):
        if not callable(f):
          return lambda t: f
        return f
      self.get_transition_noise_for_timestep = _maybe_make_callable(
          transition_noise)
      self.get_observation_noise_for_timestep = _maybe_make_callable(
          observation_noise)

      # We call the get_observation_matrix_for_timestep once so that
      # we can infer the observation size. This potentially adds ops
      # to the graph, though will not in typical cases (e.g., where
      # the callable was generated by wrapping a fixed value using
      # _maybe_make_callable above).
      self.latent_size = util.prefer_static_value(
          initial_state_prior.event_shape_tensor())[-1]
      self.observation_size = util.prefer_static_value(
          self.get_observation_matrix_for_timestep(
              self.initial_step).shape_tensor())[-2]

      self.runtime_assertions = []
      if validate_args:
        transition_matrix = (
            self.get_transition_matrix_for_timestep(self.initial_step))
        transition_noise = (
            self.get_transition_noise_for_timestep(self.initial_step))
        observation_matrix = (
            self.get_observation_matrix_for_timestep(self.initial_step))
        observation_noise = (
            self.get_observation_noise_for_timestep(self.initial_step))

        tf.assert_same_float_dtype([initial_state_prior,
                                    transition_matrix,
                                    transition_noise,
                                    observation_matrix,
                                    observation_noise])

        latent_size_ = util.static_value(self.latent_size)
        observation_size_ = util.static_value(self.observation_size)
        runtime_assertions = [
            _check_equal_shape(
                name="transition_matrix",
                static_shape=transition_matrix.shape[-2:],
                dynamic_shape=transition_matrix.shape_tensor()[-2:],
                static_target_shape=[latent_size_, latent_size_],
                dynamic_target_shape=[self.latent_size, self.latent_size]),
            _check_equal_shape(
                name="observation_matrix",
                static_shape=observation_matrix.shape[-2:],
                dynamic_shape=observation_matrix.shape_tensor()[-2:],
                static_target_shape=[observation_size_, latent_size_],
                dynamic_target_shape=[self.observation_size, self.latent_size]),
            _check_equal_shape(
                name="initial_state_prior",
                static_shape=initial_state_prior.event_shape,
                dynamic_shape=initial_state_prior.event_shape_tensor(),
                static_target_shape=[latent_size_],
                dynamic_target_shape=[self.latent_size]),
            _check_equal_shape(
                name="transition_noise",
                static_shape=transition_noise.event_shape,
                dynamic_shape=transition_noise.event_shape_tensor(),
                static_target_shape=[latent_size_],
                dynamic_target_shape=[self.latent_size]),
            _check_equal_shape(
                name="observation_noise",
                static_shape=observation_noise.event_shape,
                dynamic_shape=observation_noise.event_shape_tensor(),
                static_target_shape=[observation_size_],
                dynamic_target_shape=[self.observation_size])]
        self.runtime_assertions = [op for op in runtime_assertions
                                   if op is not None]
        _, _ = self._batch_shape(), self._batch_shape_tensor()

      super(LinearGaussianStateSpaceModel, self).__init__(
          dtype=dtype,
          reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=[],
          name=name)

  def _batch_shape_tensor(self):
    # We assume the batch shapes of parameters don't change over time,
    # so use the initial step as a prototype.
    return tf.broadcast_dynamic_shape(
        self.initial_state_prior.batch_shape_tensor(),
        tf.broadcast_dynamic_shape(
            self.get_transition_matrix_for_timestep(
                self.initial_step).batch_shape_tensor(),
            tf.broadcast_dynamic_shape(
                self.get_transition_noise_for_timestep(
                    self.initial_step).batch_shape_tensor(),
                tf.broadcast_dynamic_shape(
                    self.get_observation_matrix_for_timestep(
                        self.initial_step).batch_shape_tensor(),
                    self.get_observation_noise_for_timestep(
                        self.initial_step).batch_shape_tensor()))))

  def _batch_shape(self):
    # We assume the batch shapes of parameters don't change over time,
    # so use the initial step as a prototype.
    return tf.broadcast_static_shape(
        self.initial_state_prior.batch_shape,
        tf.broadcast_static_shape(
            self.get_transition_matrix_for_timestep(
                self.initial_step).batch_shape,
            tf.broadcast_static_shape(
                self.get_transition_noise_for_timestep(
                    self.initial_step).batch_shape,
                tf.broadcast_static_shape(
                    self.get_observation_matrix_for_timestep(
                        self.initial_step).batch_shape,
                    self.get_observation_noise_for_timestep(
                        self.initial_step).batch_shape))))

  def _event_shape(self):
    return tf.TensorShape([
        tensor_util.constant_value(
            tf.convert_to_tensor(self.num_timesteps)),
        tensor_util.constant_value(
            tf.convert_to_tensor(self.observation_size))])

  def _event_shape_tensor(self):
    return tf.stack([self.num_timesteps, self.observation_size])

  def _sample_n(self, n, seed=None):
    _, observation_samples = self._joint_sample_n(n, seed=seed)
    return observation_samples

  def _joint_sample_n(self, n, seed=None):
    """Draw a joint sample from the prior over latents and observations."""

    with tf.name_scope("sample_n_joint"):
      stream = seed_stream.SeedStream(
          seed, salt="LinearGaussianStateSpaceModel_sample_n_joint")

      sample_and_batch_shape = util.prefer_static_value(
          tf.concat([[n], self.batch_shape_tensor()],
                    axis=0))

      # Sample the initial timestep from the prior.  Since we want
      # this sample to have full batch shape (not just the batch shape
      # of the self.initial_state_prior object which might in general be
      # smaller), we augment the sample shape to include whatever
      # extra batch dimensions are required.
      with tf.control_dependencies(self.runtime_assertions):
        initial_latent = self.initial_state_prior.sample(
            sample_shape=_augment_sample_shape(
                self.initial_state_prior,
                sample_and_batch_shape,
                self.validate_args),
            seed=stream())

        # Add a dummy dimension so that matmul() does matrix-vector
        # multiplication.
        initial_latent = initial_latent[..., tf.newaxis]

      initial_observation_matrix = (
          self.get_observation_matrix_for_timestep(self.initial_step))
      initial_observation_noise = (
          self.get_observation_noise_for_timestep(self.initial_step))

      initial_observation_pred = initial_observation_matrix.matmul(
          initial_latent)
      initial_observation = (initial_observation_pred +
                             initial_observation_noise.sample(
                                 sample_shape=_augment_sample_shape(
                                     initial_observation_noise,
                                     sample_and_batch_shape,
                                     self.validate_args),
                                 seed=stream())[..., tf.newaxis])

      sample_step = build_kalman_sample_step(
          self.get_transition_matrix_for_timestep,
          self.get_transition_noise_for_timestep,
          self.get_observation_matrix_for_timestep,
          self.get_observation_noise_for_timestep,
          full_sample_and_batch_shape=sample_and_batch_shape,
          stream=stream,
          validate_args=self.validate_args)

      # Scan over all timesteps to sample latents and observations.
      (latents, observations) = tf.scan(
          sample_step,
          elems=tf.range(self.initial_step+1, self.final_step),
          initializer=(initial_latent, initial_observation))

      # Combine the initial sampled timestep with the remaining timesteps.
      latents = tf.concat([initial_latent[tf.newaxis, ...],
                           latents], axis=0)
      observations = tf.concat([initial_observation[tf.newaxis, ...],
                                observations], axis=0)

      # Put dimensions back in order. The samples we've computed are
      # ordered by timestep, with shape `[num_timesteps, num_samples,
      # batch_shape, size, 1]` where `size` represents `latent_size`
      # or `observation_size` respectively. But timesteps are really
      # part of each probabilistic event, so we need to return a Tensor
      # of shape `[num_samples, batch_shape, num_timesteps, size]`.
      latents = tf.squeeze(latents, -1)
      latents = util.move_dimension(latents, 0, -2)
      observations = tf.squeeze(observations, -1)
      observations = util.move_dimension(observations, 0, -2)

    return latents, observations

  def _log_prob(self, x):
    log_likelihoods, _, _, _, _, _, _ = self.forward_filter(x)

    # Sum over timesteps to compute the log marginal likelihood.
    return tf.reduce_sum(log_likelihoods, axis=-1)

  def forward_filter(self, x):
    """Run a Kalman filter over a provided sequence of outputs.

    Note that the returned values `filtered_means`, `predicted_means`, and
    `observation_means` depend on the observed time series `x`, while the
    corresponding covariances are independent of the observed series; i.e., they
    depend only on the model itself. This means that the mean values have shape
    `concat([sample_shape(x), batch_shape, [num_timesteps,
    {latent/observation}_size]])`, while the covariances have shape
    `concat[(batch_shape, [num_timesteps, {latent/observation}_size,
    {latent/observation}_size]])`, which does not depend on the sample shape.

    Args:
      x: a float-type `Tensor` with rightmost dimensions
        `[num_timesteps, observation_size]` matching
        `self.event_shape`. Additional dimensions must match or be
        broadcastable to `self.batch_shape`; any further dimensions
        are interpreted as a sample shape.

    Returns:
      log_likelihoods: Per-timestep log marginal likelihoods `log
        p(x_t | x_{:t-1})` evaluated at the input `x`, as a `Tensor`
        of shape `sample_shape(x) + batch_shape + [num_timesteps].`
      filtered_means: Means of the per-timestep filtered marginal
         distributions p(z_t | x_{:t}), as a Tensor of shape
        `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`.
      filtered_covs: Covariances of the per-timestep filtered marginal
         distributions p(z_t | x_{:t}), as a Tensor of shape
        `batch_shape + [num_timesteps, latent_size, latent_size]`.
      predicted_means: Means of the per-timestep predictive
         distributions over latent states, p(z_{t+1} | x_{:t}), as a
         Tensor of shape `sample_shape(x) + batch_shape +
         [num_timesteps, latent_size]`.
      predicted_covs: Covariances of the per-timestep predictive
         distributions over latent states, p(z_{t+1} | x_{:t}), as a
         Tensor of shape `batch_shape + [num_timesteps, latent_size,
         latent_size]`.
      observation_means: Means of the per-timestep predictive
         distributions over observations, p(x_{t} | x_{:t-1}), as a
         Tensor of shape `sample_shape(x) + batch_shape +
         [num_timesteps, observation_size]`.
      observation_covs: Covariances of the per-timestep predictive
         distributions over observations, p(x_{t} | x_{:t-1}), as a
         Tensor of shape `batch_shape + [num_timesteps,
         observation_size, observation_size]`.
    """

    with tf.name_scope("forward_filter", values=[x]):
      x = tf.convert_to_tensor(x, name="x")

      # Check event shape statically if possible
      check_shape_op = _check_equal_shape(
          "x", x.shape[-2:], tf.shape(x)[-2:],
          self.event_shape, self.event_shape_tensor())
      if self.validate_args:
        runtime_assertions = (self.runtime_assertions
                              if check_shape_op is None
                              else self.runtime_assertions + [check_shape_op])
        with tf.control_dependencies(runtime_assertions):
          x = tf.identity(x)

      # Get the full output sample_shape + batch shape. Usually
      # this will just be x[:-2], i.e. the input shape excluding
      # event shape. But users can specify inputs that broadcast
      # batch dimensions, so we need to broadcast this against
      # self.batch_shape.
      if self.batch_shape.is_fully_defined() and x.shape.is_fully_defined():
        sample_and_batch_shape = tf.broadcast_static_shape(
            x.shape[:-2], self.batch_shape)
      else:
        sample_and_batch_shape = tf.broadcast_dynamic_shape(
            tf.shape(x)[:-2], self.batch_shape_tensor())

      # To scan over timesteps we need to move `num_timsteps` from the
      # event shape to the initial dimension of the tensor.
      x = util.move_dimension(x, -2, 0)

      # Observations are assumed to be vectors, but we add a dummy
      # extra dimension to allow us to use `matmul` throughout.
      x = x[..., tf.newaxis]

      # Initialize filtering distribution from the prior. The mean in
      # a Kalman filter depends on data, so should match the full
      # sample and batch shape. The covariance is data-independent, so
      # only has batch shape.
      prior_mean = _broadcast_to_shape(
          self.initial_state_prior.mean()[..., tf.newaxis],
          tf.concat([sample_and_batch_shape,
                     [self.latent_size, 1]], axis=0))
      prior_cov = _broadcast_to_shape(
          self.initial_state_prior.covariance(),
          tf.concat([self.batch_shape_tensor(),
                     [self.latent_size, self.latent_size]], axis=0))

      initial_observation_matrix = (
          self.get_observation_matrix_for_timestep(self.initial_step))
      initial_observation_noise = (
          self.get_observation_noise_for_timestep(self.initial_step))

      initial_observation_mean = _propagate_mean(prior_mean,
                                                 initial_observation_matrix,
                                                 initial_observation_noise)
      initial_observation_cov = _propagate_cov(prior_cov,
                                               initial_observation_matrix,
                                               initial_observation_noise)

      initial_state = KalmanFilterState(
          predicted_mean=prior_mean,
          predicted_cov=prior_cov,
          filtered_mean=prior_mean,  # establishes shape, value ignored
          filtered_cov=prior_cov,  # establishes shape, value ignored
          observation_mean=initial_observation_mean,
          observation_cov=initial_observation_cov,
          log_marginal_likelihood=tf.zeros(
              shape=sample_and_batch_shape, dtype=self.dtype),
          timestep=tf.convert_to_tensor(
              self.initial_step, dtype=tf.int32, name="initial_step"))

      update_step_fn = build_kalman_filter_step(
          self.get_transition_matrix_for_timestep,
          self.get_transition_noise_for_timestep,
          self.get_observation_matrix_for_timestep,
          self.get_observation_noise_for_timestep)

      filter_states = tf.scan(update_step_fn,
                              elems=x,
                              initializer=initial_state)

      log_likelihoods = util.move_dimension(
          filter_states.log_marginal_likelihood, 0, -1)

      # Move the time dimension back into the event shape.
      filtered_means = util.move_dimension(
          filter_states.filtered_mean[..., 0], 0, -2)
      filtered_covs = util.move_dimension(
          filter_states.filtered_cov, 0, -3)
      predicted_means = util.move_dimension(
          filter_states.predicted_mean[..., 0], 0, -2)
      predicted_covs = util.move_dimension(
          filter_states.predicted_cov, 0, -3)
      observation_means = util.move_dimension(
          filter_states.observation_mean[..., 0], 0, -2)
      observation_covs = util.move_dimension(
          filter_states.observation_cov, 0, -3)
      # We could directly construct the batch Distributions
      # filtered_marginals = tfd.MultivariateNormalFullCovariance(
      #      filtered_means, filtered_covs)
      # predicted_marginals = tfd.MultivariateNormalFullCovariance(
      #      predicted_means, predicted_covs)
      # but we choose not to: returning the raw means and covariances
      # saves computation in Eager mode (avoiding an immediate
      # Cholesky factorization that the user may not want) and aids
      # debugging of numerical issues.

      return (log_likelihoods,
              filtered_means, filtered_covs,
              predicted_means, predicted_covs,
              observation_means, observation_covs)

  def _mean(self):
    _, observation_mean = self._joint_mean()
    return observation_mean

  def _joint_mean(self):
    """Compute prior means for all variables via dynamic programming.

    Returns:
      latent_means: Prior means of latent states `z_t`, as a `Tensor`
        of shape `batch_shape + [num_timesteps, latent_size]`
      observation_means: Prior covariance matrices of observations
        `x_t`, as a `Tensor` of shape `batch_shape + [num_timesteps,
        observation_size]`
    """

    with tf.name_scope("mean_joint"):

      # The initial timestep is a special case, since we sample the
      # latent state from the prior rather than the transition model.

      with tf.control_dependencies(self.runtime_assertions):
        # Broadcast to ensure we represent the full batch shape.
        initial_latent_mean = _broadcast_to_shape(
            self.initial_state_prior.mean()[..., tf.newaxis],
            tf.concat([self.batch_shape_tensor(),
                       [self.latent_size, 1]], axis=0))

      initial_observation_mean = _propagate_mean(
          initial_latent_mean,
          self.get_observation_matrix_for_timestep(self.initial_step),
          self.get_observation_noise_for_timestep(self.initial_step))

      mean_step = build_kalman_mean_step(
          self.get_transition_matrix_for_timestep,
          self.get_transition_noise_for_timestep,
          self.get_observation_matrix_for_timestep,
          self.get_observation_noise_for_timestep)

      # Scan over all timesteps following the initial step.
      (latent_means, observation_means) = tf.scan(
          mean_step,
          elems=tf.range(self.initial_step+1, self.final_step),
          initializer=(initial_latent_mean, initial_observation_mean))

      # Squish the initial step back on top of the other (scanned) timesteps
      latent_means = tf.concat([initial_latent_mean[tf.newaxis, ...],
                                latent_means], axis=0)
      observation_means = tf.concat([initial_observation_mean[tf.newaxis, ...],
                                     observation_means], axis=0)

      # Put dimensions back in order. The samples we've computed have
      # shape `[num_timesteps, batch_shape, size, 1]`, where `size`
      # is the dimension of the latent or observation spaces
      # respectively, but we want to return values with shape
      # `[batch_shape, num_timesteps, size]`.
      latent_means = tf.squeeze(latent_means, -1)
      latent_means = util.move_dimension(latent_means, 0, -2)
      observation_means = tf.squeeze(observation_means, -1)
      observation_means = util.move_dimension(observation_means, 0, -2)

      return latent_means, observation_means

  def _joint_covariances(self):
    """Compute prior covariances for all variables via dynamic programming.

    Returns:
      latent_covs: Prior covariance matrices of latent states `z_t`, as
        a `Tensor` of shape `batch_shape + [num_timesteps,
        latent_size, latent_size]`
      observation_covs: Prior covariance matrices of observations
        `x_t`, as a `Tensor` of shape `batch_shape + [num_timesteps,
        observation_size, observation_size]`
    """

    with tf.name_scope("covariance_joint"):

      with tf.control_dependencies(self.runtime_assertions):
        initial_latent_cov = _broadcast_to_shape(
            self.initial_state_prior.covariance(),
            tf.concat([self.batch_shape_tensor(),
                       [self.latent_size, self.latent_size]], axis=0))

      initial_observation_cov = _propagate_cov(
          initial_latent_cov,
          self.get_observation_matrix_for_timestep(self.initial_step),
          self.get_observation_noise_for_timestep(self.initial_step))

      cov_step = build_kalman_cov_step(
          self.get_transition_matrix_for_timestep,
          self.get_transition_noise_for_timestep,
          self.get_observation_matrix_for_timestep,
          self.get_observation_noise_for_timestep)

      # Scan over all timesteps following the initial step.
      (latent_covs, observation_covs) = tf.scan(
          cov_step,
          elems=tf.range(self.initial_step+1, self.final_step),
          initializer=(initial_latent_cov, initial_observation_cov))

      # Squish the initial step back on top of the other (scanned) timesteps
      latent_covs = tf.concat([initial_latent_cov[tf.newaxis, ...],
                               latent_covs], axis=0)
      observation_covs = tf.concat([initial_observation_cov[tf.newaxis, ...],
                                    observation_covs], axis=0)

      # Put dimensions back in order. The samples we've computed have
      # shape `[num_timesteps, batch_shape, size, size]`, where `size`
      # is the dimension of the state or observation spaces
      # respectively, but we want to return values with shape
      # `[batch_shape, num_timesteps, size, size]`.
      latent_covs = util.move_dimension(latent_covs, 0, -3)
      observation_covs = util.move_dimension(observation_covs, 0, -3)
      return latent_covs, observation_covs

  def _variance(self):
    _, observation_covs = self._joint_covariances()
    return tf.matrix_diag_part(observation_covs)


KalmanFilterState = collections.namedtuple("KalmanFilterState", [
    "filtered_mean", "filtered_cov",
    "predicted_mean", "predicted_cov",
    "observation_mean", "observation_cov",
    "log_marginal_likelihood", "timestep"])


def build_kalman_filter_step(get_transition_matrix_for_timestep,
                             get_transition_noise_for_timestep,
                             get_observation_matrix_for_timestep,
                             get_observation_noise_for_timestep):
  """Build a callable that performs one step of Kalman filtering.

  Args:
    get_transition_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[latent_size, latent_size]`.
    get_transition_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[latent_size]`.
    get_observation_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[observation_size, observation_size]`.
    get_observation_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[observation_size]`.

  Returns:
    kalman_filter_step: a callable that updates a KalmanFilterState
      from timestep `t-1` to `t`.
  """

  def kalman_filter_step(state, x_t):
    """Run a single step of Kalman filtering.

    Args:
      state: A `KalmanFilterState` object representing the previous
        filter state at time `t-1`.
      x_t: A `Tensor` with event shape `[observation_size, 1]`,
        representing the vector observed at time `t`.

    Returns:
      new_state: A `KalmanFilterState` object representing the new
        filter state at time `t`.
    """

    # Given predicted mean u_{t|t-1} and covariance P_{t|t-1} from the
    # previous step, incorporate the observation x_t, producing the
    # filtered mean u_t and covariance P_t.
    (filtered_mean,
     filtered_cov,
     observation_dist) = linear_gaussian_update(
         state.predicted_mean, state.predicted_cov,
         get_observation_matrix_for_timestep(state.timestep),
         get_observation_noise_for_timestep(state.timestep),
         x_t)

    # Compute the marginal likelihood p(x_{t} | x_{:t-1}) for this
    # observation.
    log_marginal_likelihood = observation_dist.log_prob(x_t[..., 0])

    # Run the filtered posterior through the transition
    # model to predict the next time step:
    #  u_{t|t-1} = F_t u_{t-1} + b_t
    #  P_{t|t-1} = F_t P_{t-1} F_t' + Q_t
    predicted_mean, predicted_cov = kalman_transition(
        filtered_mean,
        filtered_cov,
        get_transition_matrix_for_timestep(state.timestep),
        get_transition_noise_for_timestep(state.timestep))

    return KalmanFilterState(
        filtered_mean, filtered_cov,
        predicted_mean, predicted_cov,
        observation_dist.mean()[..., tf.newaxis],
        observation_dist.covariance(),
        log_marginal_likelihood,
        state.timestep+1)

  return kalman_filter_step


def linear_gaussian_update(prior_mean, prior_cov,
                           observation_matrix, observation_noise,
                           x_observed):
  """Conjugate update for a linear Gaussian model.

  Given a normal prior on a latent variable `z`,
    `p(z) = N(prior_mean, prior_cov) = N(u, P)`,
  for which we observe a linear Gaussian transformation `x`,
    `p(x|z) = N(H * z + c, R)`,
  the posterior is also normal:
    `p(z|x) = N(u*, P*)`.

  We can write this update as
     x_expected = H * u + c # pushforward prior mean
     S = R + H * P * H'  # pushforward prior cov
     K = P * H' * S^{-1} # optimal Kalman gain
     u* = u + K * (x_observed - x_expected) # posterior mean
     P* = (I - K * H) * P (I - K * H)' + K * R * K' # posterior cov
  (see, e.g., https://en.wikipedia.org/wiki/Kalman_filter#Update)

  Args:
    prior_mean: `Tensor` with event shape `[latent_size, 1]` and
      potential batch shape `B = [b1, ..., b_n]`.
    prior_cov: `Tensor` with event shape `[latent_size, latent_size]`
      and batch shape `B` (matching `prior_mean`).
    observation_matrix: `LinearOperator` with shape
      `[observation_size, latent_size]` and batch shape broadcastable
      to `B`.
    observation_noise: potentially-batched
      `MultivariateNormalLinearOperator` instance with event shape
      `[observation_size]` and batch shape broadcastable to `B`.
    x_observed: potentially batched `Tensor` with event shape
      `[observation_size, 1]` and batch shape `B`.

  Returns:
    posterior_mean: `Tensor` with event shape `[latent_size, 1]` and
      batch shape `B`.
    posterior_cov: `Tensor` with event shape `[latent_size,
      latent_size]` and batch shape `B`.
    predictive_dist: the prior predictive distribution `p(x|z)`,
      as a `tfd.MultivariateNormalTriL` instance with event
      shape `[observation_size]` and batch shape `B`.
  """

  # Push the predicted mean for the latent state through the
  # observation model
  x_expected = _propagate_mean(prior_mean,
                               observation_matrix,
                               observation_noise)

  # Push the predictive covariance of the latent state through the
  # observation model:
  #  S = R + H * P * H'.
  # We use a temporary variable for H * P,
  # reused below to compute Kalman gain.
  tmp_obs_cov = observation_matrix.matmul(prior_cov)
  predicted_obs_cov = (
      observation_matrix.matmul(tmp_obs_cov, adjoint_arg=True)
      + observation_noise.covariance())

  # Compute optimal Kalman gain:
  #  K = P * H' * S^{-1}
  # Since both S and P are cov matrices, thus symmetric,
  # we can take the transpose and reuse our previous
  # computation:
  #      = (S^{-1} * H * P)'
  #      = (S^{-1} * tmp_obs_cov) '
  #      = (S \ tmp_obs_cov)'
  predicted_obs_cov_chol = tf.cholesky(predicted_obs_cov)
  gain_transpose = tf.cholesky_solve(predicted_obs_cov_chol, tmp_obs_cov)

  # Compute the posterior mean, incorporating the observation.
  #  u* = u + K (x_observed - x_expected)
  posterior_mean = (prior_mean +
                    _matmul(gain_transpose, x_observed - x_expected,
                            adjoint_a=True))

  # For the posterior covariance, we could use the simple update
  #  P* = P - K * H * P
  # but this is prone to numerical issues because it subtracts a
  # value from a PSD matrix.  We choose instead to use the more
  # expensive Jordan form update
  #  P* = (I - K H) * P * (I - K H)' + K R K'
  # which always produces a PSD result. This uses
  #  tmp_term = (I - K * H)'
  # as an intermediate quantity.
  latent_size = util.prefer_static_value(observation_matrix.shape_tensor())[-1]
  tmp_term = (
      tf.eye(latent_size, dtype=observation_matrix.dtype) -
      observation_matrix.matmul(gain_transpose, adjoint=True))
  posterior_cov = (
      _matmul(tmp_term, _matmul(prior_cov, tmp_term), adjoint_a=True)
      + _matmul(gain_transpose,
                _matmul(observation_noise.covariance(), gain_transpose),
                adjoint_a=True))

  predictive_dist = mvn_tril.MultivariateNormalTriL(
      loc=x_expected[..., 0],
      scale_tril=predicted_obs_cov_chol)

  return posterior_mean, posterior_cov, predictive_dist


def kalman_transition(filtered_mean, filtered_cov,
                      transition_matrix, transition_noise):
  """Propagate a filtered distribution through a transition model."""

  predicted_mean = _propagate_mean(filtered_mean,
                                   transition_matrix,
                                   transition_noise)
  predicted_cov = _propagate_cov(filtered_cov,
                                 transition_matrix,
                                 transition_noise)
  return predicted_mean, predicted_cov


def build_kalman_mean_step(get_transition_matrix_for_timestep,
                           get_transition_noise_for_timestep,
                           get_observation_matrix_for_timestep,
                           get_observation_noise_for_timestep):
  """Build a callable that performs one step of Kalman mean recursion.

  Args:
    get_transition_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[latent_size, latent_size]`.
    get_transition_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[latent_size]`.
    get_observation_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[observation_size, observation_size]`.
    get_observation_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[observation_size]`.

  Returns:
    kalman_mean_step: a callable that computes latent state and
      observation means at time `t`, given latent mean at time `t-1`.
  """

  def mean_step(previous_means, t):
    """Single step of prior mean recursion."""
    previous_latent_mean, _ = previous_means

    latent_mean = _propagate_mean(previous_latent_mean,
                                  get_transition_matrix_for_timestep(t - 1),
                                  get_transition_noise_for_timestep(t - 1))
    observation_mean = _propagate_mean(latent_mean,
                                       get_observation_matrix_for_timestep(t),
                                       get_observation_noise_for_timestep(t))
    return (latent_mean, observation_mean)

  return mean_step


def build_kalman_cov_step(get_transition_matrix_for_timestep,
                          get_transition_noise_for_timestep,
                          get_observation_matrix_for_timestep,
                          get_observation_noise_for_timestep):
  """Build a callable for one step of Kalman covariance recursion.

  Args:
    get_transition_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[latent_size, latent_size]`.
    get_transition_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[latent_size]`.
    get_observation_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[observation_size, observation_size]`.
    get_observation_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[observation_size]`.

  Returns:
    cov_step: a callable that computes latent state and observation
      covariance at time `t`, given latent covariance at time `t-1`.
  """

  def cov_step(previous_covs, t):
    """Single step of prior covariance recursion."""
    previous_latent_cov, _ = previous_covs

    latent_cov = _propagate_cov(
        previous_latent_cov,
        get_transition_matrix_for_timestep(t - 1),
        get_transition_noise_for_timestep(t - 1))
    observation_cov = _propagate_cov(
        latent_cov,
        get_observation_matrix_for_timestep(t),
        get_observation_noise_for_timestep(t))

    return (latent_cov, observation_cov)

  return cov_step


def build_kalman_sample_step(get_transition_matrix_for_timestep,
                             get_transition_noise_for_timestep,
                             get_observation_matrix_for_timestep,
                             get_observation_noise_for_timestep,
                             full_sample_and_batch_shape,
                             stream,
                             validate_args=False):
  """Build a callable for one step of Kalman sampling recursion.

  Args:
    get_transition_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[latent_size, latent_size]`.
    get_transition_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[latent_size]`.
    get_observation_matrix_for_timestep: callable taking a timestep
      as an integer `Tensor` argument, and returning a `LinearOperator`
      of shape `[observation_size, observation_size]`.
    get_observation_noise_for_timestep: callable taking a timestep as
      an integer `Tensor` argument, and returning a
      `MultivariateNormalLinearOperator` of event shape
      `[observation_size]`.
    full_sample_and_batch_shape: Desired sample and batch shape of the
      returned samples, concatenated in a single `Tensor`.
    stream: `tfd.SeedStream` instance used to generate a
      sequence of random seeds.
    validate_args: if True, perform error checking at runtime.

  Returns:
    sample_step: a callable that samples the latent state and
      observation at time `t`, given latent state at time `t-1`.
  """

  def sample_step(sampled_prev, t):
    """Sample values for a single timestep."""
    latent_prev, _ = sampled_prev

    transition_matrix = get_transition_matrix_for_timestep(t - 1)
    transition_noise = get_transition_noise_for_timestep(t - 1)

    latent_pred = transition_matrix.matmul(latent_prev)
    latent_sampled = latent_pred + transition_noise.sample(
        sample_shape=_augment_sample_shape(
            transition_noise,
            full_sample_and_batch_shape,
            validate_args),
        seed=stream())[..., tf.newaxis]

    observation_matrix = get_observation_matrix_for_timestep(t)
    observation_noise = get_observation_noise_for_timestep(t)

    observation_pred = observation_matrix.matmul(latent_sampled)
    observation_sampled = observation_pred + observation_noise.sample(
        sample_shape=_augment_sample_shape(
            observation_noise,
            full_sample_and_batch_shape,
            validate_args),
        seed=stream())[..., tf.newaxis]

    return (latent_sampled, observation_sampled)

  return sample_step


def _propagate_mean(mean, linop, dist):
  """Propagate a mean through linear Gaussian transformation."""
  return linop.matmul(mean) + dist.mean()[..., tf.newaxis]


def _propagate_cov(cov, linop, dist):
  """Propagate covariance through linear Gaussian transformation."""
  # For linop A and input cov P, returns `A P A' + dist.cov()`
  return linop.matmul(linop.matmul(cov), adjoint_arg=True) + dist.covariance()
