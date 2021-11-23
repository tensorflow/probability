"""Autoregressive Moving Average model."""
# Dependency imports
import tensorflow.compat.v2 as tf
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.sts.components.autoregressive import make_ar_transition_matrix



class ARMAStateSpaceModel(tfd.LinearGaussianStateSpaceModel):
  """State space model for an autoregressive moving average process."""

  def __init__(self,
               num_timesteps,
               ar_coefficients,
               ma_coefficients,
               level_scale,
               initial_state_prior,
               observation_noise_scale=0.,
               name=None,
               **linear_gaussian_ssm_kwargs):
      """Build a state space model implementing an ARMA(p, p - 1) process.

      References:
          http://web.pdx.edu/~crkl/readings/Hamilton94.pdf

      Hamilton Autoregressive Moving Average (ARMA) State Space Model (SSM):
          (obs eqn)   y_t = Z * a_t
          (state eqn) a_t+1 = T * a_t + e_t

          T = [ phi[0], phi[1],     ..., phi[r]
                1.,       0 ,       ..., 0.
                0.,       1.,       ..., 0.
                ...
                0.,       0.,  ...,  1.,  0.   ]
          Z = [1, theta[0], theta[1], ..., theta[r-1]]
          e_t ~ N(loc=0., scale=diag([level_scale, 0., 0., ..., 0.])

      Notes:
          This representation is a ARMA(p, p - 1) process where q = p - 1.
          Expects len(ar_coefficients) == len(ma_coefficients) + 1

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

          if p != q + 1:
              raise ValueError("Only ARMA(p, p-1) representation supported.")

          self._order = p
          self._ar_coefficients = ar_coefficients
          self._ma_coefficients = ma_coefficients
          self._level_scale = level_scale

          super(ARMAStateSpaceModel, self).__init__(
              num_timesteps=num_timesteps,
              transition_matrix=make_ar_transition_matrix(ar_coefficients),
              transition_noise=tfd.MultivariateNormalDiag(
                  scale_diag=tf.stack([level_scale] +
                                      [tf.zeros_like(level_scale)] * (
                                              self.order - 1), axis=-1)),
              observation_matrix=tf.concat(
                  [tf.ones([1, 1], dtype=dtype),
                   tf.reshape(ma_coefficients, (-1, 1))], axis=-1),
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