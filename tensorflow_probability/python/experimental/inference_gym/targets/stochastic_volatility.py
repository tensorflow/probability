# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd

from tensorflow_probability.python.experimental.inference_gym.internal import data
from tensorflow_probability.python.experimental.inference_gym.targets import bayesian_model
from tensorflow_probability.python.experimental.inference_gym.targets import model
from tensorflow_probability.python.internal import prefer_static as ps

__all__ = [
    'StochasticVolatility'
]

Root = tfd.JointDistributionCoroutine.Root


def autoregressive_series_fn(num_timesteps,
                             mean,
                             noise_scale,
                             persistence):
  """Generative process for an order-1 autoregressive process."""
  x_t = yield Root(tfd.Normal(
      loc=mean,
      scale=noise_scale / tf.math.sqrt(1 - persistence**2),
      name='x_{:06d}'.format(0)))
  for t in range(1, num_timesteps):
    # The 'centered' representation used here is challenging for inference
    # algorithms. A noncentered representation would be easier, but no one ever
    # achieved greatness by doing only easy things.
    x_t = yield tfd.Normal(
        loc=mean + persistence * (x_t -  mean),
        scale=noise_scale,
        name='x_{:06d}'.format(t))


def stochastic_volatility_prior_fn(num_timesteps):
  """Generative process for the stochastic volatility model."""
  persistence_of_volatility = yield Root(
      tfb.Shift(-1.)(
          tfb.Scale(2.)(
              tfd.Beta(concentration1=20.,
                       concentration0=1.5,
                       name='persistence_of_volatility'))))
  mean_log_volatility = yield Root(tfd.Cauchy(loc=0., scale=5.,
                                              name='mean_log_volatility'))
  white_noise_shock_scale = yield Root(tfd.HalfCauchy(
      loc=0., scale=2., name='white_noise_shock_scale'))

  _ = yield tfd.JointDistributionCoroutine(
      functools.partial(autoregressive_series_fn,
                        num_timesteps=num_timesteps,
                        mean=mean_log_volatility,
                        noise_scale=white_noise_shock_scale,
                        persistence=persistence_of_volatility),
      name='log_volatility')


def stochastic_volatility_log_likelihood_fn(values, centered_returns):
  """Likelihood of observed returns under the hypothesized volatilities."""
  log_volatility = tf.stack(values[-1], axis=-1)
  likelihood = tfd.Normal(loc=0., scale=tf.exp(log_volatility / 2.))
  return tf.reduce_sum(likelihood.log_prob(centered_returns), axis=-1)


class StochasticVolatility(bayesian_model.BayesianModel):
  # pylint: disable=line-too-long
  """Stochastic model of asset price volatility over time.

  The model assumes that asset prices follow a random walk, in which the
  volatility---the variance of the day-to-day returns---is itself changing
  over time. The volatility is assumed to evolve as an AR(1) process with
  unknown coefficient and white noise shock scale.

  ```none

  persistence_of_volatility ~ 2. * Beta(concentration1=20.,
                                        concentration0=1.5) - 1.
  mean_log_volatility ~ Cauchy(loc=0., scale=5.)
  white_noise_shock_scale ~ HalfCauchy(loc=0., scale=2.)

  # Order-1 autoregressive process on log volatility.
  log_volatility[0] ~ Normal(
      loc=0.,
      scale=white_noise_shock_scale / sqrt(1. - persistence_of_volatility**2))
  for t in range(1, num_timesteps):
    log_volatility[t] ~ Normal(
        loc=persistence_of_volatility * log_volatility[t - 1],
        scale=white_noise_shock_scale)

  centered_returns ~ Normal(
    loc=0., scale=sqrt(exp(mean_log_volatility + log_volatility)))
  ```

  #### References

  [1] https://mc-stan.org/docs/2_23/stan-users-guide/stochastic-volatility-models.html
  """
  # pylint: enable=line-too-long

  def __init__(
      self,
      centered_returns,
      name='stochastic_volatility',
      pretty_name='Stochastic Volatility'):
    """Construct the stochastic volatility model.

    Args:
      centered_returns: Float `Tensor` of shape `[num_timesteps]` giving the
        mean-adjusted return (change in asset price, minus the average change)
        observed at each step.
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
                            num_timesteps=num_timesteps))

      self._log_likelihood_fn = functools.partial(
          stochastic_volatility_log_likelihood_fn,
          centered_returns=centered_returns)

      def _ext_identity(params):
        res = collections.OrderedDict()
        res['persistence_of_volatility'] = params[0]
        res['mean_log_volatility'] = params[1]
        res['white_noise_shock_scale'] = params[2]
        res['log_volatility'] = tf.stack(params[3], axis=-1)
        return res

      sample_transformations = {
          'identity':
              model.Model.SampleTransformation(
                  fn=_ext_identity,
                  pretty_name='Identity',
              )
      }

    super(StochasticVolatility, self).__init__(
        default_event_space_bijector=(tfb.Sigmoid(-1., 1.),
                                      tfb.Identity(),
                                      tfb.Softplus()) + ((
                                          tfb.Identity(),) * num_timesteps,),
        event_shape=self._prior_dist.event_shape,
        dtype=self._prior_dist.dtype,
        name=name,
        pretty_name=pretty_name,
        sample_transformations=sample_transformations,
    )

  def _prior_distribution(self):
    return self._prior_dist

  def log_likelihood(self, value):
    return self._log_likelihood_fn(value)


class StochasticVolatilitySP500(StochasticVolatility):
  """Stochastic volatility model of ten years of S&P500 returns."""

  def __init__(self):
    dataset = data.sp500_closing_prices()
    super(StochasticVolatilitySP500, self).__init__(
        name='stochastic_volatility_sp500',
        pretty_name='Stochastic volatility model of S&P500 returns.',
        **dataset)


class StochasticVolatilitySP500Small(StochasticVolatility):
  """Stochastic volatility model of 100 days of S&P500 returns."""

  def __init__(self):
    dataset = data.sp500_closing_prices(num_points=100)
    super(StochasticVolatilitySP500Small, self).__init__(
        name='stochastic_volatility_sp500_small',
        pretty_name='Smaller stochastic volatility model of S&P500 returns.',
        **dataset)
