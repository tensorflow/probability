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
"""Sums of time-series models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries

tfl = tf.linalg


class AdditiveStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """A state space model representing a sum of component state space models.

  A state space model (SSM) posits a set of latent (unobserved) variables that
  evolve over time with dynamics specified by a probabilistic transition model
  `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
  observation model conditioned on the current state, `p(x[t] | z[t])`. The
  special case where both the transition and observation models are Gaussians
  with mean specified as a linear function of the inputs, is known as a linear
  Gaussian state space model and supports tractable exact probabilistic
  calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
  details.

  The `AdditiveStateSpaceModel` represents a sum of component state space
  models. Each of the `N` components describes a random process
  generating a distribution on observed time series `x1[t], x2[t], ..., xN[t]`.
  The additive model represents the sum of these
  processes, `y[t] = x1[t] + x2[t] + ... + xN[t] + eps[t]`, where
  `eps[t] ~ N(0, observation_noise_scale)` is an observation noise term.

  #### Mathematical Details

  The additive model concatenates the latent states of its component models.
  The generative process runs each component's dynamics in its own subspace of
  latent space, and then observes the sum of the observation models from the
  components.

  Formally, the transition model is linear Gaussian:

  ```
  p(z[t+1] | z[t]) ~ Normal(loc = transition_matrix.matmul(z[t]),
                            cov = transition_cov)
  ```

  where each `z[t]` is a latent state vector concatenating the component
  state vectors, `z[t] = [z1[t], z2[t], ..., zN[t]]`, so it has size
  `latent_size = sum([c.latent_size for c in components])`.

  The transition matrix is the block-diagonal composition of transition
  matrices from the component processes:

  ```
  transition_matrix =
    [[ c0.transition_matrix,  0.,                   ..., 0.                   ],
     [ 0.,                    c1.transition_matrix, ..., 0.                   ],
     [ ...                    ...                   ...                       ],
     [ 0.,                    0.,                   ..., cN.transition_matrix ]]
  ```

  and the noise covariance is similarly the block-diagonal composition of
  component noise covariances:

  ```
  transition_cov =
    [[ c0.transition_cov, 0.,                ..., 0.                ],
     [ 0.,                c1.transition_cov, ..., 0.                ],
     [ ...                ...                     ...               ],
     [ 0.,                0.,                ..., cN.transition_cov ]]
  ```

  The observation model is also linear Gaussian,

  ```
  p(y[t] | z[t]) ~ Normal(loc = observation_matrix.matmul(z[t]),
                          stddev = observation_noise_scale)
  ```

  This implementation assumes scalar observations, so
  `observation_matrix` has shape `[1, latent_size]`. The additive
  observation matrix simply concatenates the observation matrices from each
  component:

  ```
  observation_matrix =
    concat([c0.obs_matrix, c1.obs_matrix, ..., cN.obs_matrix], axis=-1)
  ```

  The effect is that each component observation matrix acts on the dimensions
  of latent state corresponding to that component, and the overall expected
  observation is the sum of the expected observations from each component.

  If `observation_noise_scale` is not explicitly specified, it is also computed
  by summing the noise variances of the component processes:

  ```
  observation_noise_scale = sqrt(sum([
    c.observation_noise_scale**2 for c in components]))
  ```

  #### Examples

  To construct an additive state space model combining a local linear trend
  and day-of-week seasonality component (note, the `StructuralTimeSeries`
  classes, e.g., `Sum`, provide a higher-level interface for this
  construction, which will likely be preferred by most users):

  ```
    num_timesteps = 30
    local_ssm = tfp.sts.LocalLinearTrendStateSpaceModel(
        num_timesteps=num_timesteps,
        level_scale=0.5,
        slope_scale=0.1,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=[0., 0.], scale_diag=[1., 1.]))
    day_of_week_ssm = tfp.sts.SeasonalStateSpaceModel(
        num_timesteps=num_timesteps,
        num_seasons=7,
        initial_state_prior=tfd.MultivariateNormalDiag(
            loc=tf.zeros([7]), scale_diag=tf.ones([7])))
    additive_ssm = tfp.sts.AdditiveStateSpaceModel(
        component_ssms=[local_ssm, day_of_week_ssm],
        observation_noise_scale=0.1)

    y = additive_ssm.sample()
    print(y.shape)
    # => []
  ```

  """

  # TODO(b/115656646): test the docstring example using
  # `tfp.sts.SeasonalStateSpaceModel` once that component is checked in.

  def __init__(self,
               component_ssms,
               constant_offset=0.,
               observation_noise_scale=None,
               initial_state_prior=None,
               initial_step=0,
               validate_args=False,
               allow_nan_stats=True,
               name=None):
    """Build a state space model representing the sum of component models.

    Args:
      component_ssms: Python `list` containing one or more
        `tfd.LinearGaussianStateSpaceModel` instances. The components
        will in general implement different time-series models, with possibly
        different `latent_size`, but they must have the same `dtype`, event
        shape (`num_timesteps` and `observation_size`), and their batch shapes
        must broadcast to a compatible batch shape.
      constant_offset: scalar `float` `Tensor`, or batch of scalars,
        specifying a constant value added to the sum of outputs from the
        component models. This allows the components to model the shifted series
        `observed_time_series - constant_offset`.
        Default value: `0.`
      observation_noise_scale: Optional scalar `float` `Tensor` indicating the
        standard deviation of the observation noise. May contain additional
        batch dimensions, which must broadcast with the batch shape of elements
        in `component_ssms`. If `observation_noise_scale` is specified for the
        `AdditiveStateSpaceModel`, the observation noise scales of component
        models are ignored. If `None`, the observation noise scale is derived
        by summing the noise variances of the component models, i.e.,
        `observation_noise_scale = sqrt(sum(
        [ssm.observation_noise_scale**2 for ssm in component_ssms]))`.
      initial_state_prior: Optional instance of `tfd.MultivariateNormal`
        representing a prior distribution on the latent state at time
        `initial_step`. If `None`, defaults to the independent priors from
        component models, i.e.,
        `[component.initial_state_prior for component in component_ssms]`.
        Default value: `None`.
      initial_step: Optional scalar `int` `Tensor` specifying the starting
        timestep.
        Default value: 0.
      validate_args: Python `bool`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are
        invalid, correct behavior is not guaranteed.
        Default value: `False`.
      allow_nan_stats: Python `bool`. If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member. If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
        Default value: `True`.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "AdditiveStateSpaceModel".

    Raises:
      ValueError: if components have different `num_timesteps`.
    """

    with tf.compat.v1.name_scope(
        name,
        'AdditiveStateSpaceModel',
        values=[observation_noise_scale, initial_step]) as name:
      # Check that all components have the same dtype
      dtype = tf.debugging.assert_same_float_dtype(component_ssms)

      constant_offset = tf.convert_to_tensor(value=constant_offset,
                                             name='constant_offset',
                                             dtype=dtype)
      assertions = []

      # Construct an initial state prior as a block-diagonal combination
      # of the component state priors.
      if initial_state_prior is None:
        initial_state_prior = sts_util.factored_joint_mvn(
            [ssm.initial_state_prior for ssm in component_ssms])
      dtype = initial_state_prior.dtype

      static_num_timesteps = [
          tf.get_static_value(ssm.num_timesteps)
          for ssm in component_ssms
          if tf.get_static_value(ssm.num_timesteps) is not None
      ]

      # If any components have a static value for `num_timesteps`, use that
      # value for the additive model. (and check that all other static values
      # match it).
      if static_num_timesteps:
        num_timesteps = static_num_timesteps[0]
        if not all([component_timesteps == num_timesteps
                    for component_timesteps in static_num_timesteps]):
          raise ValueError('Additive model components must all have the same '
                           'number of timesteps '
                           '(saw: {})'.format(static_num_timesteps))
      else:
        num_timesteps = component_ssms[0].num_timesteps
      if validate_args and len(static_num_timesteps) != len(component_ssms):
        assertions += [
            tf.compat.v1.assert_equal(
                num_timesteps,
                ssm.num_timesteps,
                message='Additive model components must all have '
                'the same number of timesteps.') for ssm in component_ssms
        ]

      # Define the transition and observation models for the additive SSM.
      # See the "mathematical details" section of the class docstring for
      # further information. Note that we define these as callables to
      # handle the fully general case in which some components have time-
      # varying dynamics.
      def transition_matrix_fn(t):
        return tfl.LinearOperatorBlockDiag(
            [ssm.get_transition_matrix_for_timestep(t)
             for ssm in component_ssms])

      def transition_noise_fn(t):
        return sts_util.factored_joint_mvn(
            [ssm.get_transition_noise_for_timestep(t)
             for ssm in component_ssms])

      # Build the observation matrix, concatenating (broadcast) observation
      # matrices from components. We also take this as an opportunity to enforce
      # any dynamic assertions we may have generated above.
      broadcast_batch_shape = tf.convert_to_tensor(
          value=sts_util.broadcast_batch_shape(
              [ssm.get_observation_matrix_for_timestep(initial_step)
               for ssm in component_ssms]), dtype=tf.int32)
      broadcast_obs_matrix = tf.ones(
          tf.concat([broadcast_batch_shape, [1, 1]], axis=0), dtype=dtype)
      if assertions:
        with tf.control_dependencies(assertions):
          broadcast_obs_matrix = tf.identity(broadcast_obs_matrix)

      def observation_matrix_fn(t):
        return tfl.LinearOperatorFullMatrix(
            tf.concat([ssm.get_observation_matrix_for_timestep(t).to_dense() *
                       broadcast_obs_matrix for ssm in component_ssms],
                      axis=-1))

      offset_vector = constant_offset[..., tf.newaxis]
      if observation_noise_scale is not None:
        observation_noise_scale = tf.convert_to_tensor(
            value=observation_noise_scale,
            name='observation_noise_scale',
            dtype=dtype)
        def observation_noise_fn(t):
          return tfd.MultivariateNormalDiag(
              loc=sum([ssm.get_observation_noise_for_timestep(t).mean()
                       for ssm in component_ssms]) + offset_vector,
              scale_diag=observation_noise_scale[..., tf.newaxis])
      else:
        def observation_noise_fn(t):
          return sts_util.sum_mvns(
              [tfd.MultivariateNormalDiag(
                  loc=offset_vector,
                  scale_diag=tf.zeros_like(offset_vector))] +
              [ssm.get_observation_noise_for_timestep(t)
               for ssm in component_ssms])

      super(AdditiveStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=transition_matrix_fn,
          transition_noise=transition_noise_fn,
          observation_matrix=observation_matrix_fn,
          observation_noise=observation_noise_fn,
          initial_state_prior=initial_state_prior,
          initial_step=initial_step,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)


class Sum(StructuralTimeSeries):
  """Sum of structural time series components.

  This class enables compositional specification of a structural time series
  model from basic components. Given a list of component models, it represents
  an additive model, i.e., a model of time series that may be decomposed into a
  sum of terms corresponding to the component models.

  Formally, the additive model represents a random process
  `g[t] = f1[t] + f2[t] + ... + fN[t] + eps[t]`, where the `f`'s are the
  random processes represented by the components, and
  `eps[t] ~ Normal(loc=0, scale=observation_noise_scale)` is an observation
  noise term. See the `AdditiveStateSpaceModel` documentation for mathematical
  details.

  This model inherits the parameters (with priors) of its components, and
  adds an `observation_noise_scale` parameter governing the level of noise in
  the observed time series.

  #### Examples

  To construct a model combining a local linear trend with a day-of-week effect:

  ```
    local_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_trend')
    day_of_week_effect = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week_effect')
    additive_model = tfp.sts.Sum(
        components=[local_trend, day_of_week_effect],
        observed_time_series=observed_time_series)

    print([p.name for p in additive_model.parameters])
    # => `[observation_noise_scale,
    #      local_trend_level_scale,
    #      local_trend_slope_scale,
    #      day_of_week_effect_drift_scale`]

    print(local_trend.latent_size,
          seasonal.latent_size,
          additive_model.latent_size)
    # => `2`, `7`, `9`
  ```

  """

  def __init__(self,
               components,
               constant_offset=None,
               observation_noise_scale_prior=None,
               observed_time_series=None,
               name=None):
    """Specify a structural time series model representing a sum of components.

    Args:
      components: Python `list` of one or more StructuralTimeSeries instances.
        These must have unique names.
      constant_offset: optional scalar `float` `Tensor`, or batch of scalars,
        specifying a constant value added to the sum of outputs from the
        component models. This allows the components to model the shifted series
        `observed_time_series - constant_offset`. If `None`, this is set to the
        mean of the provided `observed_time_series`.
        Default value: `None`.
      observation_noise_scale_prior: optional `tfd.Distribution` instance
        specifying a prior on `observation_noise_scale`. If `None`, a heuristic
        default prior is constructed based on the provided
        `observed_time_series`.
        Default value: `None`.
      observed_time_series: optional `float` `Tensor` of shape
        `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
        supported when `T > 1`), specifying an observed time series. This is
        used to set the constant offset, if not provided, and to construct a
        default heuristic `observation_noise_scale_prior` if not provided. May
        optionally be an instance of `tfp.sts.MaskedTimeSeries`, which includes
        a mask `Tensor` to specify timesteps with missing observations.
        Default value: `None`.
      name: Python `str` name of this model component; used as `name_scope`
        for ops created by this class.
        Default value: 'Sum'.

    Raises:
      ValueError: if components do not have unique names.
    """

    with tf.compat.v1.name_scope(
        name, 'Sum', values=[observed_time_series]) as name:
      if observed_time_series is not None:
        observed_mean, observed_stddev, _ = (
            sts_util.empirical_statistics(observed_time_series))
      else:
        observed_mean, observed_stddev = 0., 1.

      if observation_noise_scale_prior is None:
        observation_noise_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.01 * observed_stddev), scale=2.)

      if constant_offset is None:
        constant_offset = observed_mean

      # Check that components have unique names, to ensure that inherited
      # parameters will be assigned unique names.
      component_names = [c.name for c in components]
      if len(component_names) != len(set(component_names)):
        raise ValueError(
            'Components must have unique names: {}'.format(component_names))
      components_by_name = collections.OrderedDict(
          [(c.name, c) for c in components])

      # Build parameters list for the combined model, by inheriting parameters
      # from the component models in canonical order.
      parameters = [
          Parameter('observation_noise_scale', observation_noise_scale_prior,
                    tfb.Softplus()),
      ] + [Parameter(name='{}_{}'.format(component.name, parameter.name),
                     prior=parameter.prior,
                     bijector=parameter.bijector)
           for component in components for parameter in component.parameters]

      self._components = components
      self._components_by_name = components_by_name
      self._constant_offset = constant_offset

      super(Sum, self).__init__(
          parameters=parameters,
          latent_size=sum(
              [component.latent_size for component in components]),
          name=name)

  @property
  def components(self):
    """List of component `StructuralTimeSeries` models."""
    return self._components

  @property
  def components_by_name(self):
    """OrderedDict mapping component names to components."""
    return self._components_by_name

  @property
  def constant_offset(self):
    """Constant value subtracted from observed data."""
    return self._constant_offset

  def make_component_state_space_models(self,
                                        num_timesteps,
                                        param_vals,
                                        initial_step=0):
    """Build an ordered list of Distribution instances for component models.

    Args:
      num_timesteps: Python `int` number of timesteps to model.
      param_vals: a list of `Tensor` parameter values in order corresponding to
        `self.parameters`, or a dict mapping from parameter names to values.
      initial_step: optional `int` specifying the initial timestep to model.
        This is relevant when the model contains time-varying components,
        e.g., holidays or seasonality.

    Returns:
      component_ssms: a Python list of `LinearGaussianStateSpaceModel`
        Distribution objects, in order corresponding to `self.components`.
    """

    with tf.compat.v1.name_scope('make_component_state_space_models'):

      # List the model parameters in canonical order
      param_map = self._canonicalize_param_vals_as_map(param_vals)
      param_vals_list = [param_map[p.name] for p in self.parameters]

      # Build SSMs for each component model. We process the components in
      # canonical order, extracting the parameters for each component from the
      # (ordered) list of parameters.
      remaining_param_vals = param_vals_list[1:]
      component_ssms = []
      for component in self.components:
        num_parameters = len(component.parameters)
        component_param_vals = remaining_param_vals[:num_parameters]
        remaining_param_vals = remaining_param_vals[num_parameters:]

        component_ssms.append(
            component.make_state_space_model(
                num_timesteps,
                param_vals=component_param_vals,
                initial_step=initial_step))

    return component_ssms

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_step=0,
                              initial_state_prior=None,
                              constant_offset=None):

    # List the model parameters in canonical order
    param_vals_list = [param_map[p.name] for p in self.parameters]
    observation_noise_scale = param_vals_list[0]

    component_ssms = self.make_component_state_space_models(
        num_timesteps=num_timesteps,
        param_vals=param_map,
        initial_step=initial_step)

    if constant_offset is None:
      constant_offset = self.constant_offset

    return AdditiveStateSpaceModel(
        component_ssms=component_ssms,
        constant_offset=constant_offset,
        observation_noise_scale=observation_noise_scale,
        initial_state_prior=initial_state_prior,
        initial_step=initial_step)
