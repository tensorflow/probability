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
"""Autoregressive model."""
# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.internal import distribution_util as dist_util
from tensorflow_probability.python.internal import dtype_util

from tensorflow_probability.python.sts.internal import util as sts_util
from tensorflow_probability.python.sts.structural_time_series import Parameter
from tensorflow_probability.python.sts.structural_time_series import StructuralTimeSeries


class AutoregressiveStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for an autoregressive process.

  A state space model (SSM) posits a set of latent (unobserved) variables that
  evolve over time with dynamics specified by a probabilistic transition model
  `p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
  observation model conditioned on the current state, `p(x[t] | z[t])`. The
  special case where both the transition and observation models are Gaussians
  with mean specified as a linear function of the inputs, is known as a linear
  Gaussian state space model and supports tractable exact probabilistic
  calculations; see `tfp.distributions.LinearGaussianStateSpaceModel` for
  details.

  In an autoregressive process, the expected level at each timestep is a linear
  function of previous levels, with added Gaussian noise:

  ```python
  level[t+1] = (sum(coefficients * levels[t:t-order:-1]) +
                Normal(0., level_scale))
  ```

  The process is characterized by a vector `coefficients` whose size determines
  the order of the process (how many previous values it looks at), and by
  `level_scale`, the standard deviation of the noise added at each step.

  This is formulated as a state space model by letting the latent state encode
  the most recent values; see 'Mathematical Details' below.

  The parameters `level_scale` and `observation_noise_scale` are each (a batch
  of) scalars, and `coefficients` is a (batch) vector of size `[order]`. The
  batch shape of this `Distribution` is the broadcast batch
  shape of these parameters and of the `initial_state_prior`.

  #### Mathematical Details

  The autoregressive model implements a
  `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = order`
  and `observation_size = 1`. The latent state vector encodes the recent history
  of the process, with the current value in the topmost dimension. At each
  timestep, the transition sums the previous values to produce the new expected
  value, shifts all other values down by a dimension, and adds noise to the
  current value. This is formally encoded by the transition model:

  ```
  transition_matrix = [ coefs[0], coefs[1], ..., coefs[order]
                        1.,       0 ,       ..., 0.
                        0.,       1.,       ..., 0.
                        ...
                        0.,       0.,  ...,  1.,  0.            ]
  transition_noise ~ N(loc=0., scale=diag([level_scale, 0., 0., ..., 0.]))
  ```

  The observation model simply extracts the current (topmost) value, and
  optionally adds independent noise at each step:

  ```
  observation_matrix = [[1., 0., ..., 0.]]
  observation_noise ~ N(loc=0, scale=observation_noise_scale)
  ```

  Models with `observation_noise_scale = 0.` are AR processes in the formal
  sense. Setting `observation_noise_scale` to a nonzero value corresponds to a
  latent AR process observed under an iid noise model.

  #### Examples

  A simple model definition:

  ```python
  ar_model = AutoregressiveStateSpaceModel(
      num_timesteps=50,
      coefficients=[0.8, -0.1],
      level_scale=0.5,
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=[1., 1.]))

  y = ar_model.sample() # y has shape [50, 1]
  lp = ar_model.log_prob(y) # log_prob is scalar
  ```

  Passing additional parameter dimensions constructs a batch of models. The
  overall batch shape is the broadcast batch shape of the parameters:

  ```python
  ar_model = AutoregressiveStateSpaceModel(
      num_timesteps=50,
      coefficients=[0.8, -0.1],
      level_scale=tf.ones([10]),
      initial_state_prior=tfd.MultivariateNormalDiag(
        scale_diag=tf.ones([10, 10, 2])))

  y = ar_model.sample(5) # y has shape [5, 10, 10, 50, 1]
  lp = ar_model.log_prob(y) # has shape [5, 10, 10]
  ```

  """

  def __init__(self,
               num_timesteps,
               coefficients,
               level_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
    """Build a state space model implementing an autoregressive process.

    Args:
      num_timesteps: Scalar `int` `Tensor` number of timesteps to model
        with this distribution.
      coefficients: `float` `Tensor` of shape `concat(batch_shape, [order])`
        defining  the autoregressive coefficients. The coefficients are defined
        backwards in time: `coefficients[0] * level[t] + coefficients[1] *
        level[t-1] + ... + coefficients[order-1] * level[t-order+1]`.
      level_scale: Scalar (any additional dimensions are treated as batch
        dimensions) `float` `Tensor` indicating the standard deviation of the
        transition noise at each step.
      initial_state_prior: instance of `tfd.MultivariateNormal`
        representing the prior distribution on latent states.  Must have
        event shape `[order]`.
      observation_noise_scale: Scalar (any additional dimensions are
        treated as batch dimensions) `float` `Tensor` indicating the standard
        deviation of the observation noise.
        Default value: 0.
      name: Python `str` name prefixed to ops created by this class.
        Default value: "AutoregressiveStateSpaceModel".
      **linear_gaussian_ssm_kwargs: Optional additional keyword arguments to
        to the base `tfd.LinearGaussianStateSpaceModel` constructor.
    """
    parameters = dict(locals())
    parameters.update(linear_gaussian_ssm_kwargs)
    del parameters['linear_gaussian_ssm_kwargs']
    with tf.name_scope(name or 'AutoregressiveStateSpaceModel') as name:

      # The initial state prior determines the dtype of sampled values.
      # Other model parameters must have the same dtype.
      dtype = initial_state_prior.dtype

      coefficients = tf.convert_to_tensor(
          value=coefficients, name='coefficients', dtype=dtype)
      level_scale = tf.convert_to_tensor(
          value=level_scale, name='level_scale', dtype=dtype)
      observation_noise_scale = tf.convert_to_tensor(
          value=observation_noise_scale,
          name='observation_noise_scale', dtype=dtype)

      order = tf.compat.dimension_value(coefficients.shape[-1])
      if order is None:
        raise ValueError('Autoregressive coefficients must have static shape.')

      self._order = order
      self._coefficients = coefficients
      self._level_scale = level_scale

      super(AutoregressiveStateSpaceModel, self).__init__(
          num_timesteps=num_timesteps,
          transition_matrix=make_ar_transition_matrix(coefficients),
          transition_noise=tfd.MultivariateNormalDiag(
              scale_diag=tf.stack([level_scale] +
                                  [tf.zeros_like(level_scale)] * (
                                      self.order - 1), axis=-1)),
          observation_matrix=tf.concat([tf.ones([1, 1], dtype=dtype),
                                        tf.zeros([1, self.order - 1],
                                                 dtype=dtype)],
                                       axis=-1),
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
  def coefficients(self):
    return self._coefficients

  @property
  def level_scale(self):
    return self._level_scale


def make_ar_transition_matrix(coefficients):
  """Build transition matrix for an autoregressive StateSpaceModel.

  When applied to a vector of previous values, this matrix computes
  the expected new value (summing the previous states according to the
  autoregressive coefficients) in the top dimension of the state space,
  and moves all previous values down by one dimension, 'forgetting' the
  final (least recent) value. That is, it looks like this:

  ```
  ar_matrix = [ coefs[0], coefs[1], ..., coefs[order]
                1.,       0 ,       ..., 0.
                0.,       1.,       ..., 0.
                ...
                0.,       0.,  ..., 1.,  0.            ]
  ```

  Args:
    coefficients: float `Tensor` of shape `concat([batch_shape, [order]])`.

  Returns:
    ar_matrix: float `Tensor` with shape `concat([batch_shape,
    [order, order]])`.
  """

  top_row = tf.expand_dims(coefficients, -2)
  coef_shape = dist_util.prefer_static_shape(coefficients)
  batch_shape, order = coef_shape[:-1], coef_shape[-1]
  remaining_rows = tf.concat([
      tf.eye(order - 1, dtype=coefficients.dtype, batch_shape=batch_shape),
      tf.zeros(tf.concat([batch_shape, (order - 1, 1)], axis=0),
               dtype=coefficients.dtype)
  ], axis=-1)
  ar_matrix = tf.concat([top_row, remaining_rows], axis=-2)
  return ar_matrix


class Autoregressive(StructuralTimeSeries):
  """Formal representation of an autoregressive model.

  An autoregressive (AR) model posits a latent `level` whose value at each step
  is a noisy linear combination of previous steps:

  ```python
  level[t+1] = (sum(coefficients * levels[t:t-order:-1]) +
                Normal(0., level_scale))
  ```

  The latent state is `levels[t:t-order:-1]`. We observe a noisy realization of
  the current level: `f[t] = level[t] + Normal(0., observation_noise_scale)` at
  each timestep.

  If `coefficients=[1.]`, the AR process is a simple random walk, equivalent to
  a `LocalLevel` model. However, a random walk's variance increases with time,
  while many AR processes (in particular, any first-order process with
  `abs(coefficient) < 1`) are *stationary*, i.e., they maintain a constant
  variance over time. This makes AR processes useful models of uncertainty.

  See the [Wikipedia article](
  https://en.wikipedia.org/wiki/Autoregressive_model#Definition) for details on
  stationarity and other mathematical properties of autoregressive processes.
  """

  def __init__(self,
               order,
               coefficients_prior=None,
               level_scale_prior=None,
               initial_state_prior=None,
               coefficient_constraining_bijector=None,
               observed_time_series=None,
               name=None):
    """Specify an autoregressive model.

    Args:
      order: scalar Python positive `int` specifying the number of past
        timesteps to regress on.
      coefficients_prior: optional `tfd.Distribution` instance specifying a
        prior on the `coefficients` parameter. If `None`, a default standard
        normal (`tfd.MultivariateNormalDiag(scale_diag=tf.ones([order]))`) prior
        is used.
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
      coefficient_constraining_bijector: optional `tfb.Bijector` instance
        representing a constraining mapping for the autoregressive coefficients.
        For example, `tfb.Tanh()` constrains the coefficients to lie in
        `(-1, 1)`, while `tfb.Softplus()` constrains them to be positive, and
        `tfb.Identity()` implies no constraint. If `None`, the default behavior
        constrains the coefficients to lie in `(-1, 1)` using a `Tanh` bijector.
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
        Default value: 'Autoregressive'.
    """
    init_parameters = dict(locals())
    with tf.name_scope(name or 'Autoregressive') as name:
      masked_time_series = None
      if observed_time_series is not None:
        masked_time_series = (
            sts_util.canonicalize_observed_time_series_with_mask(
                observed_time_series))

      dtype = dtype_util.common_dtype(
          [(masked_time_series.time_series
            if masked_time_series is not None else None),
           coefficients_prior,
           level_scale_prior,
           initial_state_prior], dtype_hint=tf.float32)

      if observed_time_series is not None:
        _, observed_stddev, observed_initial = sts_util.empirical_statistics(
            masked_time_series)
      else:
        observed_stddev, observed_initial = (
            tf.convert_to_tensor(value=1., dtype=dtype),
            tf.convert_to_tensor(value=0., dtype=dtype))
      batch_ones = tf.ones(tf.concat([
          tf.shape(observed_initial),  # Batch shape
          [order]], axis=0), dtype=dtype)

      # Heuristic default priors. Overriding these may dramatically
      # change inference performance and results.
      if coefficients_prior is None:
        coefficients_prior = tfd.MultivariateNormalDiag(
            scale_diag=batch_ones)
      if level_scale_prior is None:
        level_scale_prior = tfd.LogNormal(
            loc=tf.math.log(0.05 *  observed_stddev), scale=3.)

      if (coefficients_prior.event_shape.is_fully_defined() and
          order != coefficients_prior.event_shape[0]):
        raise ValueError("Prior dimension {} doesn't match order {}.".format(
            coefficients_prior.event_shape[0], order))

      if initial_state_prior is None:
        initial_state_prior = tfd.MultivariateNormalDiag(
            loc=observed_initial[..., tf.newaxis] * batch_ones,
            scale_diag=(tf.abs(observed_initial) +
                        observed_stddev)[..., tf.newaxis] * batch_ones)

      self._order = order
      self._coefficients_prior = coefficients_prior
      self._level_scale_prior = level_scale_prior
      self._initial_state_prior = initial_state_prior

      if coefficient_constraining_bijector is None:
        coefficient_constraining_bijector = tfb.Tanh()
      super(Autoregressive, self).__init__(
          parameters=[
              Parameter('coefficients',
                        coefficients_prior,
                        coefficient_constraining_bijector),
              Parameter('level_scale', level_scale_prior,
                        tfb.Chain([tfb.Scale(scale=observed_stddev),
                                   tfb.Softplus(low=dtype_util.eps(dtype))]))
          ],
          latent_size=order,
          init_parameters=init_parameters,
          name=name)

  @property
  def initial_state_prior(self):
    return self._initial_state_prior

  def _make_state_space_model(self,
                              num_timesteps,
                              param_map=None,
                              initial_state_prior=None,
                              **linear_gaussian_ssm_kwargs):

    if initial_state_prior is None:
      initial_state_prior = self.initial_state_prior

    if param_map:
      linear_gaussian_ssm_kwargs.update(param_map)

    return AutoregressiveStateSpaceModel(
        num_timesteps=num_timesteps,
        initial_state_prior=initial_state_prior,
        name=self.name,
        **linear_gaussian_ssm_kwargs)
