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
"""Autoregressive Moving Average model."""

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.sts.components.autoregressive import make_ar_transition_matrix
from tensorflow_probability.python.sts.internal import util as sts_util


class AutoregressiveMovingAverageStateSpaceModel(
    tfd.LinearGaussianStateSpaceModel):
  """State space model for an autoregressive moving average process.

    A state space model (SSM) posits a set of latent (unobserved) variables that
    evolve over time with dynamics specified by a probabilistic transition model
    `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
    observation model conditioned on the current state, `p(x[t] | z[t])`. The
    special case where both the transition and observation models are Gaussians
    with mean specified as a linear function of the inputs, is known as a linear
    Gaussian state space model and supports tractable exact probabilistic
    calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
    details.

    In an autoregressive moving average (ARMA) process, the expected level at
    each timestep is a linear function of previous levels, with added Gaussian
    noise, and a linear function of previous Gaussian noise:

    ```python
    level[t + 1] = (
      level_drift
      + noise[t + 1]
      + sum(ar_coefficients * levels[t:t-order:-1])
      + sum(ma_coefficients * noise[t:t-order:-1]))
    noise[t + 1] ~ Normal(0., scale=level_scale)
    ```

    The process is characterized by a vector `coefficients` whose size
    determines the order of the process (how many previous values it looks at),
    and by `level_scale`, the standard deviation of the noise added at each
    step.

    This is formulated as a state space model by letting the latent state encode
    the most recent values; see 'Mathematical Details' below.

    The parameters `level_scale` and `observation_noise_scale` are each (a batch
    of) scalars, and `coefficients` is a (batch) vector of size `[order]`. The
    batch shape of this `Distribution` is the broadcast batch
    shape of these parameters and of the `initial_state_prior`.

    #### Mathematical Details

    The Hamilton autoregressive moving average model implements a
    `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = order`
    and `observation_size = 1`. The latent state vector encodes the recent
    history of the process, with the current value in the topmost dimension. At
    each timestep, the transition sums the previous values to produce the new
    expected value, shifts all other values down by a dimension, and adds noise
    to the current value. This is formally encoded by the transition model:

    ```
    transition_matrix = [ ar_coefs[0],  ar_coefs[1],    ..., ar_coefs[p]
                          1.,           0,              ..., 0.
                          0.,           1,              ..., 0.
                          ...
                          0.,           0.,  ...,       1.,  0.   ]

    transition_noise ~ N(loc=level_drift / (1. + sum(ma_coefficients)),
                         scale=diag([level_scale, 0., 0., ..., 0.]))
    ```

    The observation model simply extracts the current (topmost) value,
    sums the previous noise and optionally adds independent noise at each step:

    ```
    observation_matrix = [1, ma_coefs[0], ma_coefs[1], ..., ma_coefs[p-1]]
    observation_noise ~ N(loc=0, scale=observation_noise_scale)
    ```

    Models with `observation_noise_scale = 0.` are ARMA(p, p-1) processes
    in the formal sense. Setting `observation_noise_scale` to a nonzero value
    corresponds to a latent ARMA(p, p-1) process observed under an iid noise
    model.

    #### References

    [1] James D. Hamilton. State-space models. __Handbook of Econometrics,
        Volume IV__ (1994): 3039-3080.
        http://web.pdx.edu/~crkl/readings/Hamilton94.pdf
  """

  def __init__(self,
               num_timesteps,
               ar_coefficients,
               ma_coefficients,
               level_scale,
               initial_state_prior,
               level_drift=0.,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """Builds a state space model implementing an ARMA(p, p - 1) process.

    Args:
        num_timesteps: Scalar `int` `Tensor` number of timesteps to model
          with this distribution.
        ar_coefficients: `float` `Tensor` of shape `concat(batch_shape,
          [order])` defining  the autoregressive coefficients. The
          ar_coefficients are defined
            backwards in time: `ar_coefficients[0] * level[t] +
              ar_coefficients[1] * level[t-1] + ... +
              ar_coefficients[order-1] * level[t-order+1]`.
        ma_coefficients: `float` `Tensor` of shape `concat(batch_shape,
          [order])` defining  the moving average coefficients. The
          ma_coefficients are defined
            backwards in time: `noise[t] + ma_coefficients[0] * noise[t-1] +
              ... + ma_coefficients[order-2] * noise[t-order+1]`.
        level_scale: Scalar (any additional dimensions are treated as batch
          dimensions) `float` `Tensor` indicating the standard deviation of
          the transition noise at each step.
        initial_state_prior: instance of `tfd.MultivariateNormal`
          representing the prior distribution on latent states.  Must have
          event shape `[order]`.
        level_drift: Scalar (any additional dimensions are
          treated as batch dimensions) `float` `Tensor` indicating a
          deterministic drift added to the level at each step.
          Default value: 0.
        observation_noise_scale: Scalar (any additional dimensions are
          treated as batch dimensions) `float` `Tensor` indicating the
          standard deviation of the observation noise.
            Default value: 0.
        name: Python `str` name prefixed to ops created by this class.
            Default value: "AutoregressiveStateSpaceModel".
        **linear_gaussian_ssm_kwargs: Optional additional keyword arguments
          to to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    Notes: This distribution is always represented as a ARMA(p, p - 1)
      process internally where q = p - 1 due to 'Mathematical Details'
      above. If q + 1 != p is desired, then either `ar_coefficients` or
      `ma_coefficients` will be automatically padded with zeros by the
      required amount to become a ARMA(p, p - 1) process.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(name or 'ARMAStateSpaceModel') as name:
      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype

      ar_coefficients = tf.convert_to_tensor(
          value=ar_coefficients, name='ar_coefficients', dtype=dtype)
      ma_coefficients = tf.convert_to_tensor(
          value=ma_coefficients, name='ma_coefficients', dtype=dtype)
      level_scale = tf.convert_to_tensor(
          value=level_scale, name='level_scale', dtype=dtype)
      level_drift = tf.convert_to_tensor(
          value=level_drift, name='level_drift', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale',
          dtype=dtype)

      # Canonicalize as ARMA[order, order - 1], where order = max(p, q + 1).
      ar_order = ps.shape(ar_coefficients)[-1]
      ma_order = ps.shape(ma_coefficients)[-1]
      order = ps.maximum(ar_order, ma_order + 1)
      ar_coefficients = sts_util.pad_tensor_with_trailing_zeros(
          ar_coefficients, order - ar_order)
      ma_coefficients = sts_util.pad_tensor_with_trailing_zeros(
          ma_coefficients, (order - 1) - ma_order)

      self._order = order
      self._ar_coefficients = ar_coefficients
      self._ma_coefficients = ma_coefficients
      self._level_scale = level_scale
      self._level_drift = level_drift
      self._observation_noise_scale = observation_noise_scale

      # Ensure the prior's shape matches the padded order.
      prior_event_dimension = tf.compat.dimension_value(
          initial_state_prior.event_shape[-1])
      if (prior_event_dimension is not None and
          prior_event_dimension != self.order):
        raise ValueError('prior event dimension needs to match max(p, q + 1). '
                         f'prior event dimension: {prior_event_dimension}, '
                         f'max(p, q + 1): {self.order}.')

      # Incorporate drift in the latent process, divided by `1 + sum(ma_coefs)`
      # to preserve the magnitude of the effect on the observed series.
      # TODO(b/211794069): Handle `level_drift` stably when sum(ma_coefs) ~= -1.
      ma_factor = 1. + tf.reduce_sum(ma_coefficients, axis=-1)
      latent_level_drift = level_drift / tf.where(
          # Prevent blowup in the drift-free case.
          tf.equal(level_drift, 0.), tf.ones_like(ma_factor), ma_factor)
      super(AutoregressiveMovingAverageStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=make_ar_transition_matrix(ar_coefficients),
          transition_noise=tfd.MultivariateNormalDiag(
              loc=sts_util.pad_tensor_with_trailing_zeros(
                  latent_level_drift[..., tf.newaxis],
                  self.order - 1),
              scale_diag=sts_util.pad_tensor_with_trailing_zeros(
                  level_scale[..., tf.newaxis], self.order - 1)),
          observation_matrix=make_ma_observation_matrix(ma_coefficients),
          observation_noise=tfd.MultivariateNormalDiag(
              scale_diag=observation_noise_scale[..., tf.newaxis]),
          initial_state_prior=initial_state_prior,
          name=name,
          **linear_gaussian_ssm_kwargs)
      self._parameters = parameters

  @property
  def order(self):
    return self._order

  @property
  def ar_coefficients(self):
    return self._ar_coefficients

  @property
  def ma_coefficients(self):
    return self._ma_coefficients

  @property
  def level_drift(self):
    return self._level_drift

  @property
  def level_scale(self):
    return self._level_scale

  @property
  def observation_noise_scale(self):
    return self._observation_noise_scale


def make_ma_observation_matrix(coefficients):
  """Build observation matrix for an moving average StateSpaceModel.

  When applied in the observation equation, this row vector extracts the
  current (topmost) value and then takes a linear combination of previous
  noise values that were added during previous recursive steps:

  ```
  observation_matrix = [1, theta[0], theta[1], ..., theta[p-1]]
  ```

  To ensure broadcasting with the transition matrix, we return a shape of:
  `concat([batch_shape, [1, order])`

  Args:
    coefficients: float `Tensor` of shape `concat([batch_shape, [order - 1])`.
  Returns:
    ma_matrix: float `Tensor` with shape `concat([batch_shape, [1, order])`.
  """
  batch_shape = ps.shape(coefficients)[:-1]
  top_entry = tf.ones(ps.concat([batch_shape, [1, 1]], axis=0),
                      dtype=coefficients.dtype)
  return tf.concat([top_entry, coefficients[..., tf.newaxis, :]], axis=-1)
