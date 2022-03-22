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
"""Stochastic volatility model."""

import collections
import functools

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import prefer_static as ps
from inference_gym.internal import data
from inference_gym.targets import bayesian_model
from inference_gym.targets import model
from inference_gym.targets.ground_truth import stochastic_volatility_sp500
from inference_gym.targets.ground_truth import stochastic_volatility_sp500_small

tfb = tfp.bijectors
tfd = tfp.distributions

__all__ = [
    'StochasticVolatility'
]

Root = tfd.JointDistributionCoroutine.Root


def autoregressive_series_fn(num_timesteps, mean, noise_scale, persistence):
  """Generative process for an order-1 autoregressive process."""
  x_t = yield Root(tfd.Normal(
      loc=mean,
      scale=noise_scale / tf.math.sqrt(tf.ones([]) - persistence**2),
      name='x_{:06d}'.format(0)))
  for t in range(1, num_timesteps):
    # The 'centered' representation used here is challenging for inference
    # algorithms. A noncentered representation would be easier, but no one ever
    # achieved greatness by doing only easy things.
    x_t = yield tfd.Normal(
        loc=mean + persistence * (x_t -  mean),
        scale=noise_scale,
        name='x_{:06d}'.format(t))


def autoregressive_series_markov_chain(num_timesteps, mean, noise_scale,
                                       persistence, name):
  return tfd.MarkovChain(
      initial_state_prior=tfd.Normal(
          loc=mean,
          scale=noise_scale / tf.math.sqrt(tf.ones([]) - persistence**2)),
      transition_fn=lambda _, x_t: tfd.Normal(  # pylint: disable=g-long-lambda
          loc=persistence * (x_t -  mean) + mean,
          scale=noise_scale),
      num_steps=num_timesteps,
      name=name)


def stochastic_volatility_prior_fn(num_timesteps, use_markov_chain):
  """Generative process for the stochastic volatility model."""
  persistence_of_volatility = yield Root(
      tfd.TransformedDistribution(
          tfd.Beta(concentration1=20.,
                   concentration0=1.5),
          tfb.Shift(-1.)(tfb.Scale(2.)),
          name='persistence_of_volatility'))
  mean_log_volatility = yield Root(tfd.Cauchy(loc=0., scale=5.,
                                              name='mean_log_volatility'))
  white_noise_shock_scale = yield Root(tfd.HalfCauchy(
      loc=0., scale=2., name='white_noise_shock_scale'))

  if use_markov_chain:
    yield autoregressive_series_markov_chain(
        num_timesteps=num_timesteps,
        mean=mean_log_volatility,
        noise_scale=white_noise_shock_scale,
        persistence=persistence_of_volatility,
        name='log_volatility')
  else:
    yield tfd.JointDistributionCoroutine(
        functools.partial(autoregressive_series_fn,
                          num_timesteps=num_timesteps,
                          mean=mean_log_volatility,
                          noise_scale=white_noise_shock_scale,
                          persistence=persistence_of_volatility),
        name='log_volatility')


def stochastic_volatility_log_likelihood_fn(
    values, centered_returns, use_markov_chain):
  """Likelihood of observed returns under the hypothesized volatilities."""
  log_volatility = (values[-1] if use_markov_chain
                    else tf.stack(values[-1], axis=-1))
  likelihood = tfd.Normal(loc=0., scale=tf.exp(log_volatility / 2.))
  return tf.reduce_sum(likelihood.log_prob(centered_returns), axis=-1)


class StochasticVolatility(bayesian_model.BayesianModel):
  """Stochastic model of asset price volatility over time.

  The model assumes that asset prices follow a random walk, in which the
  volatility---the variance of the day-to-day returns---is itself changing
  over time. The volatility is assumed to evolve as an AR(1) process with
  unknown coefficient and white noise shock scale.

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

  #### References

  [1] <https://mc-stan.org/docs/2_23/stan-users-guide/
      stochastic-volatility-models.html>
  """

  def __init__(
      self,
      centered_returns,
      use_markov_chain=False,
      name='stochastic_volatility',
      pretty_name='Stochastic Volatility'):
    """Construct the stochastic volatility model.

    Args:
      centered_returns: Float `Tensor` of shape `[num_timesteps]` giving the
        mean-adjusted return (change in asset price, minus the average change)
        observed at each step.
      use_markov_chain: Python `bool` indicating whether to use the
        `MarkovChain` distribution in place of separate random variables for
        each time step. The default of `False` is for backwards compatibility;
        setting this to `True` should significantly improve performance.
      name: Python `str` name prefixed to Ops created by this class.
      pretty_name: A Python `str`. The pretty name of this model.
    """
    with tf.name_scope(name):
      num_timesteps = ps.size0(centered_returns)
      if tf.is_tensor(num_timesteps):
        raise ValueError('Returns series length must be static, but saw '
                         'shape {}.'.format(centered_returns.shape))

      self._prior_dist = tfd.JointDistributionCoroutine(
          functools.partial(stochastic_volatility_prior_fn,
                            num_timesteps=num_timesteps,
                            use_markov_chain=use_markov_chain))

      self._log_likelihood_fn = functools.partial(
          stochastic_volatility_log_likelihood_fn,
          centered_returns=centered_returns,
          use_markov_chain=use_markov_chain)

      def _ext_identity(params):
        res = collections.OrderedDict()
        res['persistence_of_volatility'] = params.persistence_of_volatility
        res['mean_log_volatility'] = params.mean_log_volatility
        res['white_noise_shock_scale'] = params.white_noise_shock_scale
        res['log_volatility'] = (params.log_volatility
                                 if use_markov_chain
                                 else tf.stack(params.log_volatility, axis=-1))
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
                      log_volatility=tf.float32)
              )
      }

    if use_markov_chain:
      log_volatility_bijector = tfb.Identity()
    else:
      log_volatility_bijector = type(self._prior_dist.dtype[-1])(*(
          (tfb.Identity(),) * num_timesteps))
    super(StochasticVolatility, self).__init__(
        default_event_space_bijector=type(self._prior_dist.dtype)(
            tfb.Sigmoid(-1., 1.),
            tfb.Identity(),
            tfb.Softplus(),
            log_volatility_bijector),
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


class StochasticVolatilitySP500(StochasticVolatility):
  """Stochastic volatility model of ten years of S&P500 returns."""

  GROUND_TRUTH_MODULE = stochastic_volatility_sp500

  def __init__(self, use_markov_chain=False):
    dataset = data.sp500_returns()
    super(StochasticVolatilitySP500, self).__init__(
        name='stochastic_volatility_sp500',
        pretty_name='Stochastic volatility model of S&P500 returns.',
        use_markov_chain=use_markov_chain,
        **dataset)


class StochasticVolatilitySP500Small(StochasticVolatility):
  """Stochastic volatility model of 100 days of S&P500 returns."""

  GROUND_TRUTH_MODULE = stochastic_volatility_sp500_small

  def __init__(self, use_markov_chain=False):
    dataset = data.sp500_returns(num_points=100)
    super(StochasticVolatilitySP500Small, self).__init__(
        name='stochastic_volatility_sp500_small',
        pretty_name='Smaller stochastic volatility model of S&P500 returns.',
        use_markov_chain=use_markov_chain,
        **dataset)
