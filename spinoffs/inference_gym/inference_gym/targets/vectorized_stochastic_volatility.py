# Lint as: python3
# Copyright 2020 The TensorFlow Probability Authors.
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
"""Vectorized stochastic volatility model."""

import collections

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps
from inference_gym.internal import data
from inference_gym.targets import bayesian_model
from inference_gym.targets import model
from inference_gym.targets.ground_truth import stochastic_volatility_log_sp500
from inference_gym.targets.ground_truth import stochastic_volatility_log_sp500_small
from inference_gym.targets.ground_truth import stochastic_volatility_sp500
from inference_gym.targets.ground_truth import stochastic_volatility_sp500_small

tfd = tfp.distributions
tfb = tfp.bijectors

__all__ = [
    'VectorizedStochasticVolatility',
    'VectorizedStochasticVolatilityLogSP500',
    'VectorizedStochasticVolatilityLogSP500Small',
    'VectorizedStochasticVolatilitySP500',
    'VectorizedStochasticVolatilitySP500Small',
]


def _complex_real(real):
  return tf.complex(real, tf.zeros_like(real))


def _fft_conv(x, y):
  x = _complex_real(x)
  y = _complex_real(y)
  x = tf.concat([x, tf.zeros_like(x)], -1)
  y = tf.concat([y, tf.zeros_like(y)], -1)
  return tf.math.real(tf.signal.ifft(tf.signal.fft(x) *
                                     tf.signal.fft(y)))[..., :-1]


def _drive_coroutine(gen, values):
  """Iterates through a coroutine while feeding it values."""
  next(gen)
  for value in values:
    try:
      gen.send(value)
    except StopIteration as e:
      return e.value


def _fft_conv_center(noncentered, persistence_of_volatility):
  """Transform from noncentered to centered parameterization.

  Returns a centered version of `noncentered` such that
  `centered[0] == noncentered[0] / tf.sqrt(1 - persistence_of_volatility**2)`
  and for i > 0
  `centered[i] == noncentered[i] + persistence_of_volatility *
  noncentered[i-1]`.

  Rather than implement the recursion directly (which involves expensive
  sequential computation), this function implements it using FFT-based
  convolution.

  Args:
    noncentered: A (possibly batched) series of independent normal samples. The
      last dimension is taken to represent time.
    persistence_of_volatility: Scalar between -1 and 1 indicating how strong the
      dependence between time steps should be.

  Returns:
    centered: The centered time series.
  """
  noncentered = tf.convert_to_tensor(noncentered)
  persistence_of_volatility = tf.convert_to_tensor(persistence_of_volatility)
  persistence_of_volatility = persistence_of_volatility[..., tf.newaxis]
  last_dim = int(tuple(noncentered.shape)[-1])

  kernel = tf.concat([
      tf.ones_like(persistence_of_volatility), persistence_of_volatility**
      tf.range(1, last_dim, dtype=tf.float32)
  ], -1)
  centered_at_0 = tf.concat([
      noncentered[..., :1] / tf.sqrt(1 - persistence_of_volatility**2),
      noncentered[..., 1:]
  ], -1)
  return _fft_conv(centered_at_0, kernel)[..., :last_dim]


class VectorizedStochasticVolatility(bayesian_model.BayesianModel):
  """Stochastic model of asset price volatility over time.

  Compared to the `StochasticVolatility` model, this uses vectorized math and is
  significantly faster by default. The model assumes that asset prices follow a
  random walk, in which the volatility---the variance of the day-to-day
  returns---is itself changing over time. The volatility is assumed to evolve as
  an AR(1) process with unknown coefficient and white noise shock scale.

  ```none

  persistence_of_volatility ~ 2 * Beta(concentration1=20,
                                       concentration0=1.5) - 1.
  mean_log_volatility ~ Cauchy(loc=0, scale=5)
  white_noise_shock_scale ~ HalfCauchy(loc=0, scale=2)

  # Order-1 autoregressive process on log volatility.
  log_volatility[0] ~ Normal(
      loc=0,
      scale=white_noise_shock_scale / sqrt(1 - persistence_of_volatility**2))
  for t in range(1, num_timesteps):
    log_volatility[t] ~ Normal(
        loc=persistence_of_volatility * log_volatility[t - 1],
        scale=white_noise_shock_scale)

  for t in range(num_timesteps):
    centered_returns[t] ~ Normal(
      loc=0, scale=sqrt(exp(mean_log_volatility + log_volatility[t])))
  ```

  This class allows two different parameterizations of the `log_volatility`
  random variable, controlled by the `centered` argument to the initializer. The
  `centered=True` variant is as described in the block above, while the
  `centered=False` variant replaces the `log_volatility` random variable with
  `std_log_volatility` which has an standard isotropic multivariate normal
  prior. Realizations of that variable are thereafter transformed.

  The `centered=False` variant is significantly simpler to do inference on due
  to a more favorable posterior.

  #### References

  [1] <https://mc-stan.org/docs/2_23/stan-users-guide/
      stochastic-volatility-models.html>
  """

  def __init__(
      self,
      centered_returns,
      centered=False,
      use_fft=True,
      name='vectorized_stochastic_volatility',
      pretty_name='Stochastic Volatility',
  ):
    """Construct the stochastic volatility model.

    We use the parameterization from [1].


    Args:
      centered_returns: Float `Tensor` of shape `[num_timesteps]` giving the
        mean-adjusted return (change in asset price, minus the average change)
        observed at each step.
      centered: Whether or not to use the centered parameterization for log time
        volatility. The outputs of the sample transformations and their ground
        truth values are independent of this choice, this merely affects how
        easy the model is for inference algorithms.
      use_fft: Whether or not to use FFT-based convolution with the noncentered
        parameterization. Turning this off may use fewer FLOPs, and may result
        in slightly better numerics, but is dramatically slower because it uses
        scan instead of vectorized ops.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.

    Raises:
      ValueError: If the dataset has non-scalar entries.

    #### References

    1: Kim, Sangjoon, Neil Shephard, and Siddhartha Chib. 1998. “Stochastic
       Volatility: Likelihood Inference and Comparison with ARCH Models.” Review
       of Economic Studies 65: 361–93.
    """
    with tf.name_scope(name):
      if len(centered_returns.shape) > 1:
        raise ValueError('This model only supports series with scalar entires.')

      num_timesteps = centered_returns.shape[0]

      root = tfd.JointDistributionCoroutine.Root

      def log_volatility_centered_fn(white_noise_shock_scale,
                                     persistence_of_volatility):
        """Centered parameterization of log_volatility random variable."""

        def bijector_fn(std_log_volatility):
          """Bijector function to set up the autoregressive dependence."""
          shift = tf.concat([
              tf.zeros(std_log_volatility.shape[:-1])[..., tf.newaxis],
              persistence_of_volatility[..., tf.newaxis] *
              std_log_volatility[..., :-1]
          ], -1)

          scale0 = white_noise_shock_scale / tf.sqrt(
              1 - persistence_of_volatility**2)

          scale_rest = (
              white_noise_shock_scale[..., tf.newaxis] * tf.ones(
                  ps.concat([
                      ps.shape(white_noise_shock_scale),
                      ps.shape(std_log_volatility)[-1:] - 1
                  ], 0)))

          scale = tf.concat([scale0[..., tf.newaxis], scale_rest], -1)

          return tfb.Shift(shift)(tfb.Scale(scale))

        b = tfb.MaskedAutoregressiveFlow(bijector_fn=bijector_fn)

        # Interesting syntax oddity, 'return yield' is a syntax error.
        return (yield root(
            tfd.TransformedDistribution(
                tfd.Sample(tfd.Normal(0., 1.), num_timesteps),
                b,
                name='log_volatility',
            )))

      def log_volatility_noncentered_fn(white_noise_shock_scale,
                                        persistence_of_volatility):
        """Noncentered parameterization of log_volatility random variable."""
        # The non-centered parameterization for log_volatility improves geometry
        # but is slower (catastrophically so if FFT is not used).
        std_log_volatility = yield root(
            tfd.Sample(
                tfd.Normal(0., 1.),
                num_timesteps,
                name='std_log_volatility',
            ))

        if use_fft:
          return (
              white_noise_shock_scale[..., tf.newaxis] *
              _fft_conv_center(std_log_volatility, persistence_of_volatility))
        else:
          log_volatility = (
              std_log_volatility * white_noise_shock_scale[..., tf.newaxis])

          log_volatility_0 = (
              log_volatility[..., 0] /
              tf.sqrt(1 - persistence_of_volatility**2))

          # Make the time axis be first, for scan to work.
          log_volatility = distribution_util.move_dimension(
              log_volatility, -1, 0)
          # I.e.
          # log_volatility[t] += (persistence_of_volatility *
          #     log_volatility[t-1])
          log_volatility = tf.concat(
              [
                  log_volatility_0[tf.newaxis],
                  tf.scan(
                      lambda v_prev, v: persistence_of_volatility * v_prev + v,
                      log_volatility[1:], log_volatility_0)
              ],
              axis=0,
          )

          return distribution_util.move_dimension(log_volatility, 0, -1)

      def prior_fn():
        """Model definition."""
        persistence_of_volatility = yield root(
            tfd.TransformedDistribution(
                tfd.Beta(concentration1=20., concentration0=1.5),
                tfb.Shift(-1.)(tfb.Scale(2.)),
                name='persistence_of_volatility'))
        mean_log_volatility = yield root(
            tfd.Cauchy(loc=0., scale=5., name='mean_log_volatility'))
        white_noise_shock_scale = yield root(
            tfd.HalfCauchy(loc=0., scale=2., name='white_noise_shock_scale'))

        if centered:
          log_volatility = yield from log_volatility_centered_fn(
              white_noise_shock_scale, persistence_of_volatility)
        else:
          log_volatility = yield from log_volatility_noncentered_fn(
              white_noise_shock_scale, persistence_of_volatility)

        return tf.exp(0.5 *
                      (log_volatility + mean_log_volatility[..., tf.newaxis]))

      self._prior_dist = tfd.JointDistributionCoroutine(prior_fn)

      def log_likelihood_fn(value):
        observation_scales = _drive_coroutine(prior_fn(), value)
        observation_dist = tfd.Independent(
            tfd.Normal(0., observation_scales), 1)
        res = observation_dist.log_prob(centered_returns)
        return res

      self._log_likelihood_fn = log_likelihood_fn

      def _ext_identity(params):
        """Function to extract params in a centered parameterization."""
        if centered:
          log_volatility = params.log_volatility
        else:
          log_volatility = _drive_coroutine(
              log_volatility_noncentered_fn(params.white_noise_shock_scale,
                                            params.persistence_of_volatility),
              [params.std_log_volatility])
        res = collections.OrderedDict()
        res['persistence_of_volatility'] = params.persistence_of_volatility
        res['mean_log_volatility'] = params.mean_log_volatility
        res['white_noise_shock_scale'] = params.white_noise_shock_scale
        res['log_volatility'] = log_volatility + params.mean_log_volatility[
            ..., tf.newaxis]
        return res

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=_ext_identity,
                  pretty_name='Identity',
                  dtype=collections.OrderedDict(
                      persistence_of_volatility=tf.float32,
                      mean_log_volatility=tf.float32,
                      white_noise_shock_scale=tf.float32,
                      log_volatility=tf.float32))
      }

    if centered:
      log_volatility_bijector = {'log_volatility': tfb.Identity()}
    else:
      log_volatility_bijector = {'std_log_volatility': tfb.Identity()}

    super(VectorizedStochasticVolatility, self).__init__(
        default_event_space_bijector=type(self._prior_dist.dtype)(
            persistence_of_volatility=tfb.Sigmoid(-1., 1.),
            mean_log_volatility=tfb.Identity(),
            white_noise_shock_scale=tfb.Softplus(),
            **log_volatility_bijector,
        ),
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def _log_likelihood(self, value):
    return self._log_likelihood_fn(value)


class VectorizedStochasticVolatilitySP500(VectorizedStochasticVolatility):
  """Stochastic volatility model of ten years of S&P500 returns."""

  GROUND_TRUTH_MODULE = stochastic_volatility_sp500

  def __init__(
      self,
      centered=False,
      use_fft=True,
  ):
    dataset = data.sp500_returns()
    super(VectorizedStochasticVolatilitySP500, self).__init__(
        name='vectorized_stochastic_volatility_sp500',
        pretty_name='Stochastic volatility model of S&P500 returns.',
        **dataset)


class VectorizedStochasticVolatilitySP500Small(VectorizedStochasticVolatility):
  """Stochastic volatility model of 100 days of S&P500 returns."""

  GROUND_TRUTH_MODULE = stochastic_volatility_sp500_small

  def __init__(
      self,
      centered=False,
      use_fft=True,
  ):
    dataset = data.sp500_returns(num_points=100)
    super(VectorizedStochasticVolatilitySP500Small, self).__init__(
        name='vectorized_stochasticic_volatility_sp500_small',
        pretty_name='Smaller stochastic volatility model of S&P500 returns.',
        **dataset)


class VectorizedStochasticVolatilityLogSP500(VectorizedStochasticVolatility):
  """Stochastic volatility model of ten years of S&P500 log returns."""

  GROUND_TRUTH_MODULE = stochastic_volatility_log_sp500

  def __init__(
      self,
      centered=False,
      use_fft=True,
  ):
    dataset = data.sp500_log_returns()
    super(VectorizedStochasticVolatilityLogSP500, self).__init__(
        name='vectorized_stochastic_volatility_log_sp500',
        pretty_name='Stochastic volatility model of S&P500 log returns.',
        **dataset)


class VectorizedStochasticVolatilityLogSP500Small(VectorizedStochasticVolatility
                                                 ):
  """Stochastic volatility model of 100 days of S&P500 log returns."""

  GROUND_TRUTH_MODULE = stochastic_volatility_log_sp500_small

  def __init__(
      self,
      centered=False,
      use_fft=True,
  ):
    dataset = data.sp500_log_returns(num_points=100)
    super(VectorizedStochasticVolatilityLogSP500Small, self).__init__(
        name='vectorized_stochasticic_volatility_log_sp500_small',
        pretty_name=(
            'Smaller stochastic volatility model of S&P500 log returns.'),
        **dataset)
