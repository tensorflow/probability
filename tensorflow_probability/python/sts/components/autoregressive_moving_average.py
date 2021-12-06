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
"""Autoregressive Moving Average model."""
# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.sts.components.autoregressive import make_ar_transition_matrix


class AutoregressiveMovingAverageStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
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
    level[t+1] = (sum(coefficients * levels[t:t-order:-1]) +
    Normal(0., level_scale) +
    sum(coefficients * noise[t:t-order:-1]))
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

    The Hamilton autoregressive moving average model implements a
    `tfp.distributions.LinearGaussianStateSpaceModel` with `latent_size = order`
    and `observation_size = 1`. The latent state vector encodes the recent history
    of the process, with the current value in the topmost dimension. At each
    timestep, the transition sums the previous values to produce the new expected
    value, shifts all other values down by a dimension, and adds noise to the
    current value. This is formally encoded by the transition model:

    ```
    transition_matrix = [ phi[0], phi[1],     ..., phi[p]
                          1.,       0 ,       ..., 0.
                          0.,       1.,       ..., 0.
                          ...
                          0.,       0.,  ...,  1.,  0.   ]

    transition_noise ~ N(loc=0., scale=diag([level_scale, 0., 0., ..., 0.]))
    ```

    The observation model simply extracts the current (topmost) value,
    sums the previous noise and optionally adds independent noise at each step:

    ```
    observation_matrix = [1, theta[0], theta[1], ..., theta[p-1]]
    observation_noise ~ N(loc=0, scale=observation_noise_scale)
    ```

    Models with `observation_noise_scale = 0.` are ARMA(p, p-1) processes
    in the formal sense. Setting `observation_noise_scale` to a nonzero value
    corresponds to a latent ARMA(p, p-1) process observed under an iid noise model.

    See the [Wikipedia article](http://web.pdx.edu/~crkl/readings/Hamilton94.pdf)
    for details on the Hamilton state space formulation for ARMA(p, p-1) processes.
    """

    def __init__(self,
                 num_timesteps,
                 ar_coefficients,
                 ma_coefficients,
                 level_scale,
                 initial_state_prior,
                 observation_noise_scale=0.,
                 name=None,
                 **linear_gaussian_ssm_kwargs):
        """ Build a state space model implementing an ARMA(p, p - 1) process.

        Args:
            num_timesteps: Scalar `int` `Tensor` number of timesteps to model
                with this distribution.
            ar_coefficients: `float` `Tensor` of shape `concat(batch_shape, [order])`
                defining  the autoregressive coefficients. The coefficients are defined
                backwards in time: `coefficients[0] * level[t] + coefficients[1] *
                level[t-1] + ... + coefficients[order-1] * level[t-order+1]`.
            ma_coefficients: `float` `Tensor` of shape `concat(batch_shape, [order])`
                defining  the moving average coefficients. The coefficients are defined
                backwards in time: `coefficients[0] * noise[t] + coefficients[1] *
                noise[t-1] + ... + coefficients[order-1] * noise[t-order+1]`.
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

        Notes:
            This representation is a ARMA(p, p - 1) process where q = p - 1.
            If q + 1 != p, then either `ar_coefficients` or `ma_coefficients`
            will be padded with zeros by the required amount to become a
            ARMA(p, p - 1) process.
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
            observation_noise_scale = tf.convert_to_tensor(
                value=observation_noise_scale,
                name='observation_noise_scale', dtype=dtype)

            p = tf.compat.dimension_value(ar_coefficients.shape[-1])
            q = tf.compat.dimension_value(ma_coefficients.shape[-1])

            if p is None or q is None:
                raise ValueError('coefficients must have static shape.')

            # if p != q + 1, then pad either of the arguments to ensure
            # we end up fitting an ARMA(p, p-1) process.
            if p > q + 1:
                pad = tf.zeros([p - q - 1], dtype=dtype)
                ma_coefficients = tf.concat([ma_coefficients, pad], axis=-1)
            elif q + 1 > p:
                pad = tf.zeros([q + 1 - p], dtype=dtype)
                ar_coefficients = tf.concat([ar_coefficients, pad], axis=-1)

            order = max(p, q + 1)
            if order is None:
                raise ValueError('coefficients must have static shape.')

            self._order = order
            self._ar_coefficients = ar_coefficients
            self._ma_coefficients = ma_coefficients
            self._level_scale = level_scale

            super(AutoregressiveMovingAverageStateSpaceModel, self).__init__(
                num_timesteps=num_timesteps,
                transition_matrix=make_ar_transition_matrix(ar_coefficients),
                transition_noise=tfd.MultivariateNormalDiag(
                    scale_diag=tf.stack([level_scale] +
                                        [tf.zeros_like(level_scale)] * (
                                                self.order - 1), axis=-1)),
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

    @order.setter
    def order(self, order):
        self._order = order

    @property
    def ar_coefficients(self):
        return self._ar_coefficients

    @property
    def ma_coefficients(self):
        return self._ma_coefficients

    @property
    def level_scale(self):
        return self._level_scale


def make_ma_observation_matrix(coefficients):
    """Build observation matrix for an moving average StateSpaceModel.

    When applied in the observation equation, this row vector extracts the
    current (topmost) value and then takes a linear combination of previous
    noise values that were added during previous recursive steps:

    ```
    observation_matrix = [1, theta[0], theta[1], ..., theta[p-1]]
    ```

    Args:
        coefficients: float `Tensor` of shape `concat([batch_shape, [order - 1])`.

    Returns:
        ma_matrix: float `Tensor` with shape `concat([batch_shape, [order])`.
    """
    top_entry = tf.ones([1, 1], dtype=coefficients.dtype)
    ma_matrix = tf.concat(
        [top_entry, coefficients[..., tf.newaxis, :]], axis=-1
    )
    return ma_matrix
