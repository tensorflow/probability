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
"""Autoregressive integrated moving average (ARIMA) model."""
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.bijectors import tanh
from tensorflow_probability.python.distributions import linear_gaussian_ssm as lgssm
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import mvn_diag
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps

from tensorflow_probability.python.sts.components.autoregressive_moving_average import AutoregressiveMovingAverageStateSpaceModel
from tensorflow_probability.python.sts.internal import missing_values_util
from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


__all__ = [
    'IntegratedStateSpaceModel',
    'AutoregressiveIntegratedMovingAverage'
]


def _pad_mvn_with_trailing_zeros(mvn, num_zeros):
  zeros = tf.zeros([num_zeros], dtype=mvn.dtype)
  return sts_util.factored_joint_mvn(
      [mvn, mvn_diag.MultivariateNormalDiag(loc=zeros, scale_diag=zeros)])


class IntegratedStateSpaceModel(lgssm.LinearGaussianStateSpaceModel):
  """Integrates (/cumsums) a noise-free state space model.

  The integrated model represents the cumulative sum of sequences sampled from
  an underlying state space model. In the absence of observation noise,
  this distribution is equivalent to `tfb.Cumsum()(original_ssm)`, but is
  represented explicitly as a linear Gaussian state space model allowing it to
  compose with other state space model components.

  The augmented transition model stores the expected output from step `t - 1`
  as additional dimension(s) in the latent state at step `t`. The observation
  model at each step is also then augmented to sum the current output of the
  underlying SSM with the previous (expected) output. Formally:

  ```python
  augmented_latent[t] = concat([original_latent[t], output[t-1]])

  augmented_transition_matrix = [
    [original_transition_matrix, zeros([latent_size, 1])],
    [original_observation_matrix, [1.]]
  ]
  augmented_observation_matrix = [original_observation_matrix, [1.]]
  ```

  """

  def __init__(self, original_ssm, name=None, **linear_gaussian_ssm_kwargs):
    parameters = dict(locals(), **linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']

    # Use settings from `original_ssm` if not otherwise specified.
    for arg_name in ('initial_step', 'experimental_parallelize'):
      if arg_name not in linear_gaussian_ssm_kwargs:
        linear_gaussian_ssm_kwargs[arg_name] = getattr(original_ssm, arg_name)
    if linear_gaussian_ssm_kwargs.get('initial_state_prior', None) is None:
      # Augment prior on latents with extra dims for the previous observation.
      linear_gaussian_ssm_kwargs['initial_state_prior'] = (
          _pad_mvn_with_trailing_zeros(
              original_ssm.initial_state_prior,
              original_ssm.observation_size_tensor()))

    self._original_ssm = original_ssm
    super().__init__(
        num_timesteps=original_ssm.num_timesteps,
        transition_matrix=self._get_cumsum_transition_matrix_for_timestep,
        transition_noise=self._get_cumsum_transition_noise_for_timestep,
        observation_matrix=self._get_cumsum_observation_matrix_for_timestep,
        observation_noise=original_ssm.observation_noise,
        name=name or 'cumsum_' + original_ssm.name,
        **linear_gaussian_ssm_kwargs)
    self._parameters = parameters

  @property
  def original_ssm(self):
    return self._original_ssm

  def _get_cumsum_transition_matrix_for_timestep(self, t):
    """Augmented transition that also stores the expected observation."""
    original_transition_matrix = (
        self.original_ssm.get_transition_matrix_for_timestep(t))
    original_observation_matrix = (
        self.original_ssm.get_observation_matrix_for_timestep(t))
    observation_size = self.original_ssm.observation_size_tensor()
    return tf.linalg.LinearOperatorBlockLowerTriangular(
        [[original_transition_matrix],
         [original_observation_matrix,
          tf.linalg.LinearOperatorIdentity(
              observation_size,
              dtype=original_observation_matrix.dtype)]])

  def _get_cumsum_transition_noise_for_timestep(self, t):
    """Augmented transition noise with the (noiseless) extra dimensions."""
    original_transition_noise = (
        self.original_ssm.get_transition_noise_for_timestep(t))
    observation_size = self.original_ssm.observation_size_tensor()
    return _pad_mvn_with_trailing_zeros(
        original_transition_noise, observation_size)

  def _get_cumsum_observation_matrix_for_timestep(self, t):
    """Augmented observation sums with the previous observation."""
    original_observation_matrix = (
        self.original_ssm.get_observation_matrix_for_timestep(t))
    observation_size = self.original_ssm.observation_size_tensor()
    # TODO(b/185968222): use a LinearOperatorBlock in place of dense concat.
    return tf.linalg.LinearOperatorFullMatrix(
        tf.concat([original_observation_matrix.to_dense(),
                   tf.eye(observation_size,
                          dtype=original_observation_matrix.dtype,
                          batch_shape=(
                              original_observation_matrix.batch_shape_tensor()))
                   ], axis=-1))


class AutoregressiveIntegratedMovingAverage(StructuralTimeSeries):
  """Represents an autoregressive integrated moving-average (ARIMA) model.

  An [autoregressive moving-average (ARMA)](
  https://en.wikipedia.org/wiki/Autoregressive_moving_average_model) process is
  defined by the recursion

  ```
  level[t + 1] = (
      level_drift
      + noise[t + 1]
      + sum(ar_coefficients * levels[t : t - order : -1])
      + sum(ma_coefficients * noise[t : t - order : -1]))
    noise[t + 1] ~ Normal(0., scale=level_scale)
    ```

  where `noise` is an iid noise process. An integrated ([ARIMA](
  https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average))
  process corresponds to an ARMA model of the
  `integration_degree`th-order differences of a sequence, or equivalently,
  taking `integration_degree` cumulative sums of an underlying ARMA process.
  """

  def __init__(self,
               ar_order,
               ma_order,
               integration_degree=0,
               ar_coefficients_prior=None,
               ma_coefficients_prior=None,
               level_drift_prior=None,
               level_scale_prior=None,
               initial_state_prior=None,
               ar_coefficient_constraining_bijector=None,
               ma_coefficient_constraining_bijector=None,
               observed_time_series=None,
               name=None):
    """Specifies an ARIMA(p=ar_order, d=integration_degree, q=ma_order) model.

    Args:
      ar_order: scalar Python positive `int` specifying the order of the
        autoregressive process (`p` in `ARIMA(p, d, q)`).
      ma_order: scalar Python positive `int` specifying the order of the
        moving-average process (`q` in `ARIMA(p, d, q)`).
      integration_degree: scalar Python positive `int` specifying the number
        of times to integrate an ARMA process. (`d` in `ARIMA(p, d, q)`).
        Default value: `0`.
      ar_coefficients_prior: optional `tfd.Distribution` instance specifying a
        prior on the `ar_coefficients` parameter. If `None`, a default standard
        normal (`tfd.MultivariateNormalDiag(scale_diag=tf.ones([ar_order]))`)
        prior is used.
        Default value: `None`.
      ma_coefficients_prior: optional `tfd.Distribution` instance specifying a
        prior on the `ma_coefficients` parameter. If `None`, a default standard
        normal (`tfd.MultivariateNormalDiag(scale_diag=tf.ones([ma_order]))`)
        prior is used.
        Default value: `None`.
      level_drift_prior: optional `tfd.Distribution` instance specifying a prior
        on the `level_drift` parameter. If `None`, the parameter is not inferred
        and is instead fixed to zero.
        Default value: `None`.
      level_scale_prior: optional `tfd.Distribution` instance specifying a prior
        on the `level_scale` parameter. If `None`, a heuristic default prior is
        constructed based on the provided `observed_time_series`.
        Default value: `None`.
      initial_state_prior: optional `tfd.Distribution` instance specifying a
        prior on the initial state, corresponding to the values of the process
        at a set of size `order` of imagined timesteps before the initial step.
        If `None`, a heuristic default prior is constructed based on the
        provided `observed_time_series`.
        Default value: `None`.
      ar_coefficient_constraining_bijector: optional `tfb.Bijector` instance
        representing a constraining mapping for the autoregressive coefficients.
        For example, `tfb.Tanh()` constrains the coefficients to lie in
        `(-1, 1)`, while `tfb.Softplus()` constrains them to be positive, and
        `tfb.Identity()` implies no constraint. If `None`, the default behavior
        constrains the coefficients to lie in `(-1, 1)` using a `Tanh` bijector.
        Default value: `None`.
      ma_coefficient_constraining_bijector: optional `tfb.Bijector` instance
        representing a constraining mapping for the moving average coefficients.
        For example, `tfb.Tanh()` constrains the coefficients to lie in
        `(-1, 1)`, while `tfb.Softplus()` constrains them to be positive, and
        `tfb.Identity()` implies no constraint. If `None`, the default behavior
        is to apply no constraint.
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
        Default value: 'ARIMA'.
    """
    init_parameters = dict(locals())
    with tf.name_scope(name or 'ARIMA') as name:
      masked_time_series = None
      if observed_time_series is not None:
        masked_time_series = (
            sts_util.canonicalize_observed_time_series_with_mask(
                observed_time_series))
      dtype = dtype_util.common_dtype(
          [(masked_time_series.time_series
            if masked_time_series is not None else None),
           ar_coefficients_prior,
           ma_coefficients_prior,
           level_scale_prior,
           initial_state_prior], dtype_hint=tf.float32)

      if observed_time_series is not None:
        for _ in range(integration_degree):
          # Compute statistics using `integration_order`-order differences.
          masked_time_series = (
              missing_values_util.differentiate_masked_time_series(
                  masked_time_series))
        _, observed_stddev, observed_initial = sts_util.empirical_statistics(
            masked_time_series)
      else:
        observed_stddev, observed_initial = (
            tf.convert_to_tensor(value=1., dtype=dtype),
            tf.convert_to_tensor(value=0., dtype=dtype))
      batch_ones = ps.ones(ps.concat([
          ps.shape(observed_initial),  # Batch shape
          [1]], axis=0), dtype=dtype)

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if ar_coefficients_prior is None:
        ar_coefficients_prior = mvn_diag.MultivariateNormalDiag(
            scale_diag=batch_ones * ps.ones([ar_order]))
      if ma_coefficients_prior is None:
        ma_coefficients_prior = mvn_diag.MultivariateNormalDiag(
            scale_diag=batch_ones * ps.ones([ma_order]))
      if level_scale_prior is None:
        level_scale_prior = lognormal.LogNormal(
            loc=tf.math.log(0.05 * observed_stddev), scale=3.)

      if (ar_coefficients_prior.event_shape.is_fully_defined() and
          ar_order != ar_coefficients_prior.event_shape[0]):
        raise ValueError(
            "Autoregressive prior dimension {} doesn't match order {}.".format(
                ar_coefficients_prior.event_shape[0], ar_order))
      if (ma_coefficients_prior.event_shape.is_fully_defined() and
          ma_order != ma_coefficients_prior.event_shape[0]):
        raise ValueError(
            "Moving average prior dimension {} doesn't match order {}.".format(
                ma_coefficients_prior.event_shape[0], ma_order))

      latent_size = ps.maximum(ar_order, ma_order + 1) + integration_degree
      if initial_state_prior is None:
        initial_state_prior = mvn_diag.MultivariateNormalDiag(
            loc=sts_util.pad_tensor_with_trailing_zeros(
                observed_initial[..., tf.newaxis] * batch_ones,
                num_zeros=latent_size - 1),
            scale_diag=sts_util.pad_tensor_with_trailing_zeros(
                (tf.abs(observed_initial) + observed_stddev)[..., tf.newaxis] *
                batch_ones,
                num_zeros=latent_size - 1))

      self._ar_order = ar_order
      self._ma_order = ma_order
      self._integration_degree = integration_degree
      self._ar_coefficients_prior = ar_coefficients_prior
      self._ma_coefficients_prior = ma_coefficients_prior
      self._level_scale_prior = level_scale_prior
      self._initial_state_prior = initial_state_prior

      parameters = []
      if ar_order > 0:
        parameters.append(
            Parameter('ar_coefficients', ar_coefficients_prior,
                      (ar_coefficient_constraining_bijector if
                       ar_coefficient_constraining_bijector else tanh.Tanh())))
      if ma_order > 0:
        parameters.append(
            Parameter('ma_coefficients', ma_coefficients_prior,
                      (ma_coefficient_constraining_bijector
                       if ma_coefficient_constraining_bijector else
                       identity.Identity())))
      if level_drift_prior is not None:
        parameters.append(
            Parameter(
                'level_drift', level_drift_prior,
                chain.Chain([
                    scale.Scale(scale=observed_stddev),
                    (level_drift_prior
                     .experimental_default_event_space_bijector())
                ])))
      super(AutoregressiveIntegratedMovingAverage, self).__init__(
          parameters=parameters + [
              Parameter(
                  'level_scale', level_scale_prior,
                  chain.Chain([
                      scale.Scale(scale=observed_stddev),
                      softplus.Softplus(low=dtype_util.eps(dtype))
                  ]))
          ],
          latent_size=latent_size,
          init_parameters=init_parameters,
          name=name)

  @property
  def initial_state_prior(self):
    return self._initial_state_prior

  @property
  def integration_degree(self):
    return self._integration_degree

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map=None,
                              initial_state_prior=None,
                              **linear_gaussian_ssm_kwargs):
    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior

    def maybe_make_dummy_prior(integration_degree):
      """"""
      integration_steps_remaining = self.integration_degree - integration_degree
      if integration_steps_remaining == 0:
        return initial_state_prior
      dtype = initial_state_prior.dtype
      return mvn_diag.MultivariateNormalDiag(
          loc=tf.zeros([self.latent_size - integration_steps_remaining],
                       dtype=dtype),
          scale_diag=tf.ones([self.latent_size - integration_steps_remaining],
                             dtype=dtype))

    arma_kwargs = dict(linear_gaussian_ssm_kwargs,
                       **(param_map if param_map else {}))
    arma_kwargs['ar_coefficients'] = arma_kwargs.get('ar_coefficients', [])
    arma_kwargs['ma_coefficients'] = arma_kwargs.get('ma_coefficients', [])

    current_integration_degree = 0
    ssm = AutoregressiveMovingAverageStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=maybe_make_dummy_prior(
            current_integration_degree),
        name=self.name,
        **arma_kwargs)
    for _ in range(self.integration_degree):
      current_integration_degree += 1
      ssm = IntegratedStateSpaceModel(
          ssm,
          name=self.name,
          initial_state_prior=maybe_make_dummy_prior(
              current_integration_degree),
          **linear_gaussian_ssm_kwargs)
    return ssm
