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
"""Local Level model."""
# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import linear_gaussian_ssm
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class LocalLevelStateSpaceModel(
    linear_gaussian_ssm.LinearGaussianStateSpaceModel):
  """State space model for a local level.

  A state space model (SSM) posits a set of latent (unobserved) variables that
  evolve over time with dynamics specified by a probabilistic transition model
  `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
  observation model conditioned on the current state, `p(x[t] | z[t])`. The
  special case where both the transition and observation models are Gaussians
  with mean specified as a linear function of the inputs, is known as a linear
  Gaussian state space model and supports tractable exact probabilistic
  calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
  details.

  The local level model is a special case of a linear Gaussian SSM, in which the
  latent state posits a `level` evolving via a Gaussian random walk:

  ```python
  level[t] = level[t-1] + Normal(0., level_scale)
  ```

  The latent state is `[level]` and `[level]` is observed (with noise) at each
  timestep.

  The parameters `level_scale` and `observation_noise_scale` are each (a batch
  of) scalars. The batch shape of this `Distribution` is the broadcast batch
  shape of these parameters and of the `initial_state_prior`.

  #### Mathematical Details

  The local level model implements a
  `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = 1` and
  `observation_size = 1`, following the transition model:

  ```
  transition_matrix = [[1.]]
  transition_noise ~ N(loc=0., scale=diag([level_scale]))
  ```

  which implements the evolution of `level` described above, and the observation
  model:

  ```
  observation_matrix = [[1.]]
  observation_noise ~ N(loc=0, scale=observation_noise_scale)
  ```

  #### Examples

  A simple model definition:

  ```python
  local_level_model = LocalLevelStateSpaceModel(
      num_timesteps=50,
      level_scale=0.5,
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1.]))

  y = local_level_model.sample() # y has shape [50, 1]
  lp = local_level_model.log_prob(y) # log_prob is scalar
  ```

  Passing additional parameter dimensions constructs a batch of models. The
  overall batch shape is the broadcast batch shape of the parameters:

  ```python
  local_level_model = LocalLevelStateSpaceModel(
      num_timesteps=50,
      level_scale=tf.ones([10]),
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([10, 10, 1])))

  y = local_level_model.sample(5) # y has shape [5, 10, 10, 50, 1]
  lp = local_level_model.log_prob(y) # has shape [5, 10, 10]
  ```
  """

  def __init__(self,
               num_timesteps,
               level_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """Build a state space model implementing a local level.

    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      level_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        level transitions.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states.  Must have
        event shape `[1]` (as `tfd.LinearGaussianStateSpaceModel` requires a
        rank-1 event shape).
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "LocalLevelStateSpaceModel".
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(name or 'LocalLevelStateSpaceModel') as name:
      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype

      level_scale = tf.convert_to_tensor(
          value=level_scale, name='level_scale', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      self._level_scale = level_scale
      self._observation_noise_scale = observation_noise_scale

      # Construct a linear Gaussian state space model implementing the
      # local level model. See "Mathematical Details" in the
      # class docstring for further explanation.
      super(LocalLevelStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=tf.constant([[1.]],
                                        dtype=dtype,
                                        name='transition_matrix'),
          transition_noise=mvn_diag.MultivariateNormalDiag(
              scale_diag=level_scale[..., tf.newaxis], name='transition_noise'),
          observation_matrix=tf.constant([[1.]],
                                         dtype=dtype,
                                         name='observation_matrix'),
          observation_noise=mvn_diag.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis],
              name='observation_noise'),
          initial_state_prior=initial_state_prior,
          name=name,
          **linear_gaussian_ssm_kwargs)
      self._parameters = parameters

  @property
  def level_scale(self):
    """Standard deviation of the level transitions."""
    return self._level_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale

  def _joint_sample_n(self, n, seed=None):
    """Draw a joint sample from the prior over latents and observations.

    This sampler is specific to LocalLevel models and is faster than the
    generic LinearGaussianStateSpaceModel implementation.

    Args:
      n: `int` `Tensor` number of samples to draw.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    Returns:
      latents: `float` `Tensor` of shape `concat([[n], self.batch_shape,
        [self.num_timesteps, self.latent_size]], axis=0)` representing samples
        of latent trajectories.
      observations: `float` `Tensor` of shape `concat([[n], self.batch_shape,
        [self.num_timesteps, self.observation_size]], axis=0)` representing
        samples of observed series generated from the sampled `latents`.
    """
    with tf.name_scope('joint_sample_n'):
      (initial_level_seed,
       level_jumps_seed,
       prior_observation_seed) = samplers.split_seed(
           seed, n=3, salt='LocalLevelStateSpaceModel_joint_sample_n')

      if self.batch_shape.is_fully_defined():
        batch_shape = self.batch_shape
      else:
        batch_shape = self.batch_shape_tensor()
      sample_and_batch_shape = ps.cast(
          ps.concat([[n], batch_shape], axis=0), tf.int32)

      # Sample the initial timestep from the prior.  Since we want
      # this sample to have full batch shape (not just the batch shape
      # of the self.initial_state_prior object which might in general be
      # smaller), we augment the sample shape to include whatever
      # extra batch dimensions are required.
      initial_level = self.initial_state_prior.sample(
          linear_gaussian_ssm._augment_sample_shape(  # pylint: disable=protected-access
              self.initial_state_prior,
              sample_and_batch_shape,
              self.validate_args),
          seed=initial_level_seed)

      # Sample the latent random walk and observed noise, more efficiently than
      # the generic loop in `LinearGaussianStateSpaceModel`.
      level_jumps = self.level_scale[..., tf.newaxis] * samplers.normal(
          ps.concat(
              [sample_and_batch_shape, [self.num_timesteps - 1]], axis=0),
          dtype=self.dtype, seed=level_jumps_seed)
      prior_level_sample = tf.cumsum(tf.concat(
          [initial_level, level_jumps], axis=-1), axis=-1)
      prior_observation_sample = prior_level_sample + (  # Sample noise.
          self.observation_noise_scale[..., tf.newaxis] *
          samplers.normal(ps.shape(prior_level_sample),
                          dtype=self.dtype,
                          seed=prior_observation_seed))

      return (prior_level_sample[..., tf.newaxis],
              prior_observation_sample[..., tf.newaxis])


class LocalLevel(StructuralTimeSeries):
  """Formal representation of a local level model.

  The local level model posits a `level` evolving via a Gaussian random walk:

  ```
  level[t] = level[t-1] + Normal(0., level_scale)
  ```

  The latent state is `[level]`. We observe a noisy realization of the current
  level: `f[t] = level[t] + Normal(0., observation_noise_scale)` at each
  timestep.
  """

  def __init__(self,
               level_scale_prior=None,
               initial_level_prior=None,
               observed_time_series=None,
               name=None):
    """Specify a local level model.

    Args:
      level_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `level_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      initial_level_prior: optional `tfd.Distribution` instance specifying a
        prior on the initial level. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series. Any `NaN`s
        are interpreted as missing observations; missingness may be also be
        explicitly specified by passing a `tfp.sts.MaskedTimeSeries` instance.
        Any priors not explicitly set will be given default values according to
        the scale of the observed time series (or batch of time series).
        Default value: `None`.
      name: the name of this model component.
        Default value: 'LocalLevel'.
    """
    init_parameters = dict(locals())
    with tf.name_scope(name or 'LocalLevel') as name:

      if observed_time_series is not None:
        _, observed_stddev, observed_initial = (
            sts_util.empirical_statistics(observed_time_series))
      else:
        observed_stddev, observed_initial = 1., 0.

      dtype = dtype_util.common_dtype([level_scale_prior,
                                       initial_level_prior,
                                       observed_stddev,
                                       observed_initial],
                                      dtype_hint=tf.float32)

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if level_scale_prior is None:
        level_scale_prior = lognormal.LogNormal(
            loc=tf.math.log(.05 * observed_stddev),
            scale=3.,
            name='level_scale_prior')
      if initial_level_prior is None:
        self._initial_state_prior = mvn_diag.MultivariateNormalDiag(
            loc=tf.convert_to_tensor(observed_initial)[..., tf.newaxis],
            scale_diag=(tf.abs(observed_initial) + observed_stddev)[...,
                                                                    tf.newaxis],
            name='initial_level_prior')
      else:
        self._initial_state_prior = mvn_diag.MultivariateNormalDiag(
            loc=initial_level_prior.mean()[..., tf.newaxis],
            scale_diag=initial_level_prior.stddev()[..., tf.newaxis])

      super(LocalLevel, self).__init__(
          parameters=[
              Parameter(
                  'level_scale', level_scale_prior,
                  chain.Chain([
                      scale.Scale(scale=observed_stddev),
                      softplus.Softplus(low=dtype_util.eps(dtype))
                  ])),
          ],
          latent_size=1,
          init_parameters=init_parameters,
          name=name)

  @property
  def initial_state_prior(self):
    """Prior distribution on the initial latent state (level and scale)."""
    return self._initial_state_prior

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              **linear_gaussian_ssm_kwargs):

    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior
    linear_gaussian_ssm_kwargs.update(param_map)
    return LocalLevelStateSpaceModel(
        num_timesteps,
        initial_state_prior=initial_state_prior,
        **linear_gaussian_ssm_kwargs)
