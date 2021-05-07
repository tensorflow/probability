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
"""Dynamic Linear Regression model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class DynamicLinearRegressionStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for a dynamic linear regression from provided covariates.

  A state space model (SSM) posits a set of latent (unobserved) variables that
  evolve over time with dynamics specified by a probabilistic transition model
  `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
  observation model conditioned on the current state, `p(x[t] | z[t])`. The
  special case where both the transition and observation models are Gaussians
  with mean specified as a linear function of the inputs, is known as a linear
  Gaussian state space model and supports tractable exact probabilistic
  calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
  details.

  The dynamic linear regression model is a special case of a linear Gaussian SSM
  and a generalization of typical (static) linear regression. The model
  represents regression `weights` with a latent state which evolves via a
  Gaussian random walk:

  ```
  weights[t] ~ Normal(weights[t-1], drift_scale)
  ```

  The latent state (the weights) has dimension `num_features`, while the
  parameters `drift_scale` and `observation_noise_scale` are each (a batch of)
  scalars. The batch shape of this `Distribution` is the broadcast batch shape
  of these parameters, the `initial_state_prior`, and the
  `design_matrix`. `num_features` is determined from the last dimension of
  `design_matrix` (equivalent to the number of columns in the design matrix in
  linear regression).

  #### Mathematical Details

  The dynamic linear regression model implements a
  `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size =
  num_features` and `observation_size = 1` following the transition model:

  ```
  transition_matrix = eye(num_features)
  transition_noise ~ Normal(0, diag([drift_scale]))
  ```

  which implements the evolution of `weights` described above. The observation
  model is:

  ```
  observation_matrix[t] = design_matrix[t]
  observation_noise ~ Normal(0, observation_noise_scale)
  ```

  #### Examples

  Given `series1`, `series2` as `Tensors` each of shape `[num_timesteps]`
  representing covariate time series, we create a dynamic regression model which
  conditions on these via the following:

  ```python
  dynamic_regression_ssm = DynamicLinearRegressionStateSpaceModel(
      num_timesteps=42,
      design_matrix=tf.stack([series1, series2], axis=-1),
      drift_scale=3.14,
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1., 2.]),
      observation_noise_scale=1.)

  y = dynamic_regression_ssm.sample()  # shape [42, 1]
  lp = dynamic_regression_ssm.log_prob(y)  # scalar
  ```

  Passing additional parameter and `initial_state_prior` dimensions constructs a
  batch of models, consider the following:

  ```python
  dynamic_regression_ssm = DynamicLinearRegressionStateSpaceModel(
      num_timesteps=42,
      design_matrix=tf.stack([series1, series2], axis=-1),
      drift_scale=[3.14, 1.],
      initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=[1., 2.]),
      observation_noise_scale=[1., 2.])

  y = dynamic_regression_ssm.sample(3)  # shape [3, 2, 42, 1]
  lp = dynamic_regression_ssm.log_prob(y)  # shape [3, 2]
  ```

  Which (effectively) constructs two independent state space models; the first
  with `drift_scale = 3.14` and `observation_noise_scale = 1.`, the second with
  `drift_scale = 1.` and `observation_noise_scale = 2.`. We then sample from
  each of the models three times and calculate the log probability of each of
  the samples under each of the models.

  Similarly, it is also possible to add batch dimensions via the
  `design_matrix`.

  """

  def __init__(self,
               num_timesteps,
               design_matrix,
               drift_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """State space model for a dynamic linear regression.

    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      design_matrix: float `Tensor` of shape `concat([batch_shape,
        [num_timesteps, num_features]])`.
      drift_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        latent state transitions.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states.  Must have
        event shape `[num_features]`.
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
        Default value: `0.`.
      name: Python `str` name prefixed to ops created by this class.
        Default value: 'DynamicLinearRegressionStateSpaceModel'.
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(
        name or 'DynamicLinearRegressionStateSpaceModel') as name:
      dtype = dtype_util.common_dtype(
          [design_matrix, drift_scale, initial_state_prior])

      design_matrix = tf.convert_to_tensor(
          value=design_matrix, name='design_matrix', dtype=dtype)
      design_matrix_with_time_in_first_dim = distribution_util.move_dimension(
          design_matrix, -2, 0)

      drift_scale = tf.convert_to_tensor(
          value=drift_scale, name='drift_scale', dtype=dtype)

      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      num_features = prefer_static.shape(design_matrix)[-1]

      def observation_matrix_fn(t):
        observation_matrix = tf.linalg.LinearOperatorFullMatrix(
            tf.gather(design_matrix_with_time_in_first_dim,
                      t)[..., tf.newaxis, :], name='observation_matrix')
        return observation_matrix

      self._drift_scale = drift_scale
      self._observation_noise_scale = observation_noise_scale

      super(DynamicLinearRegressionStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=tf.linalg.LinearOperatorIdentity(
              num_rows=num_features,
              dtype=dtype,
              name='transition_matrix'),
          transition_noise=tfd.MultivariateNormalDiag(
              scale_diag=(drift_scale[..., tf.newaxis] *
                          tf.ones([num_features], dtype=dtype)),
              name='transition_noise'),
          observation_matrix=observation_matrix_fn,
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis],
              name='observation_noise'),
          initial_state_prior=initial_state_prior,
          name=name,
          **linear_gaussian_ssm_kwargs)
      self._parameters = parameters

  @property
  def drift_scale(self):
    """Standard deviation of the drift in weights at each timestep."""
    return self._drift_scale

  @property
  def observation_noise_scale(self):
    """Standard deviation of the observation noise."""
    return self._observation_noise_scale


class DynamicLinearRegression(StructuralTimeSeries):
  """Formal representation of a dynamic linear regresson model.

  The dynamic linear regression model is a special case of a linear Gaussian SSM
  and a generalization of typical (static) linear regression. The model
  represents regression `weights` with a latent state which evolves via a
  Gaussian random walk:

  ```
  weights[t] ~ Normal(weights[t-1], drift_scale)
  ```

  The latent state has dimension `num_features`, while the parameters
  `drift_scale` and `observation_noise_scale` are each (a batch of) scalars. The
  batch shape of this `Distribution` is the broadcast batch shape of these
  parameters, the `initial_state_prior`, and the `design_matrix`. `num_features`
  is determined from the last dimension of `design_matrix` (equivalent to the
  number of columns in the design matrix in linear regression).

  """

  def __init__(self,
               design_matrix,
               drift_scale_prior=None,
               initial_weights_prior=None,
               observed_time_series=None,
               name=None):
    """Specify a dynamic linear regression.

    Args:
      design_matrix: float `Tensor` of shape `concat([batch_shape,
        [num_timesteps, num_features]])`.
      drift_scale_prior: instance of `tfd.Distribution` specifying a prior on
        the `drift_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      initial_weights_prior: instance of `tfd.MultivariateNormal` representing
        the prior distribution on the latent states (the regression weights).
        Must have event shape `[num_features]`. If `None`, a weakly-informative
        Normal(0., 10.) prior is used.
        Default value: `None`.
      observed_time_series: `float` `Tensor` of shape `batch_shape + [T, 1]`
        (omitting the trailing unit dimension is also supported when `T > 1`),
        specifying an observed time series. Any priors not explicitly set will
        be given default values according to the scale of the observed time
        series (or batch of time series). May optionally be an instance of
        `tfp.sts.MaskedTimeSeries`, which includes a mask `Tensor` to specify
        timesteps with missing observations.
        Default value: `None`.
      name: Python `str` for the name of this component.
        Default value: 'DynamicLinearRegression'.

    """

    with tf.name_scope(name or 'DynamicLinearRegression') as name:

      dtype = dtype_util.common_dtype(
          [design_matrix, drift_scale_prior, initial_weights_prior])

      num_features = prefer_static.shape(design_matrix)[-1]

      # Default to a weakly-informative Normal(0., 10.) for the initital state
      if initial_weights_prior is None:
        initial_weights_prior = tfd.MultivariateNormalDiag(
            scale_diag=10. * tf.ones([num_features], dtype=dtype))

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if drift_scale_prior is None:
        if observed_time_series is None:
          observed_stddev = tf.constant(1.0, dtype=dtype)
        else:
          _, observed_stddev, _ = sts_util.empirical_statistics(
              observed_time_series)

        drift_scale_prior = tfd.LogNormal(
            loc=tf.math.log(.05 * observed_stddev),
            scale=3.,
            name='drift_scale_prior')

      self._initial_state_prior = initial_weights_prior
      self._design_matrix = design_matrix

      super(DynamicLinearRegression, self).__init__(
          parameters=[
              Parameter('drift_scale', drift_scale_prior,
                        tfb.Chain([tfb.Scale(scale=observed_stddev),
                                   tfb.Softplus()]))
          ],
          latent_size=num_features,
          name=name)

  @property
  def initial_state_prior(self):
    """Prior distribution on the initial latent state (level and scale)."""
    return self._initial_state_prior

  @property
  def design_matrix(self):
    """Tensor representing the design matrix."""
    return self._design_matrix

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map,
                              initial_state_prior=None,
                              **linear_gaussian_ssm_kwargs):

    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior
    linear_gaussian_ssm_kwargs.update(param_map)
    return DynamicLinearRegressionStateSpaceModel(
        num_timesteps=num_timesteps,
        design_matrix=self.design_matrix,
        initial_state_prior=initial_state_prior,
        **linear_gaussian_ssm_kwargs)
