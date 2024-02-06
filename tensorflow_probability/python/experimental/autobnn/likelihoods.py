# Copyright 2023 The TensorFlow Probability Authors.
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
"""Likelihood models for Bayesian Neural Networks."""

import dataclasses
from typing import Any
import jax
from tensorflow_probability.substrates.jax.bijectors import softplus as softplus_lib
from tensorflow_probability.substrates.jax.distributions import distribution as distribution_lib
from tensorflow_probability.substrates.jax.distributions import inflated as inflated_lib
from tensorflow_probability.substrates.jax.distributions import logistic as logistic_lib
from tensorflow_probability.substrates.jax.distributions import lognormal as lognormal_lib
from tensorflow_probability.substrates.jax.distributions import negative_binomial as negative_binomial_lib
from tensorflow_probability.substrates.jax.distributions import normal as normal_lib
from tensorflow_probability.substrates.jax.distributions import transformed_distribution as transformed_distribution_lib


@dataclasses.dataclass
class LikelihoodModel:
  """A class that knows how to compute the likelihood of some data."""

  def dist(self, params, nn_out) -> distribution_lib.Distribution:
    """Return the distribution underlying the likelihood."""
    raise NotImplementedError()

  def sample(self, params, nn_out, seed, sample_shape=None) -> jax.Array:
    """Sample from the likelihood."""
    return self.dist(params, nn_out).sample(
        seed=seed, sample_shape=sample_shape
    )

  def num_outputs(self):
    """The number of outputs from the neural network the model needs."""
    return 1

  def distributions(self):
    """Like BayesianModule::distributions but for the model's parameters."""
    return {}

  @jax.named_call
  def log_likelihood(
      self, params, nn_out: jax.Array, observations: jax.Array
  ) -> jax.Array:
    return self.dist(params, nn_out).log_prob(observations)


@dataclasses.dataclass
class DummyLikelihoodModel(LikelihoodModel):
  """A likelihood model that only knows how many outputs it has."""
  num_outs: int

  def num_outputs(self):
    return self.num_outs


class NormalLikelihoodFixedNoise(LikelihoodModel):
  """Abstract base class for observations = N(nn_out, noise_scale)."""

  def dist(self, params, nn_out):
    return normal_lib.Normal(loc=nn_out, scale=params['noise_scale'])


@dataclasses.dataclass
class NormalLikelihoodLogisticNoise(NormalLikelihoodFixedNoise):
  noise_min: float = 0.0
  log_noise_scale: float = 1.0

  def distributions(self):
    noise_scale = transformed_distribution_lib.TransformedDistribution(
        logistic_lib.Logistic(0.0, self.log_noise_scale),
        softplus_lib.Softplus(low=self.noise_min),
    )
    return {'noise_scale': noise_scale}


@dataclasses.dataclass
class BoundedNormalLikelihoodLogisticNoise(NormalLikelihoodLogisticNoise):
  lower_bound: float = 0.0

  def dist(self, params, nn_out):
    return softplus_lib.Softplus(low=self.lower_bound)(
        normal_lib.Normal(loc=nn_out, scale=params['noise_scale'])
    )


@dataclasses.dataclass
class NormalLikelihoodLogNormalNoise(NormalLikelihoodFixedNoise):
  log_noise_mean: float = -2.0
  log_noise_scale: float = 1.0

  def distributions(self):
    return {
        'noise_scale': lognormal_lib.LogNormal(
            loc=self.log_noise_mean, scale=self.log_noise_scale
        )
    }


class NormalLikelihoodVaryingNoise(LikelihoodModel):

  def num_outputs(self):
    return 2

  def dist(self, params, nn_out):
    # TODO(colcarroll): Add a prior to constrain the scale (`nn_out[..., [1]]`)
    # separately before it goes into the likelihood.
    return normal_lib.Normal(
        loc=nn_out[..., [0]], scale=jax.nn.softplus(nn_out[..., [1]])
    )


class NegativeBinomial(LikelihoodModel):
  """observations = NB(total_count = nn_out[0], logits = nn_out[1])."""

  def num_outputs(self):
    return 2

  def dist(self, params, nn_out):
    return negative_binomial_lib.NegativeBinomial(
        total_count=nn_out[..., [0]],
        logits=nn_out[..., [1]],
        require_integer_total_count=False,
    )


class ZeroInflatedNegativeBinomial(LikelihoodModel):
  """observations = NB(total_count = nn_out[0], logits = nn_out[1])."""

  def num_outputs(self):
    return 3

  def dist(self, params, nn_out):
    return inflated_lib.ZeroInflatedNegativeBinomial(
        total_count=nn_out[..., [0]],
        logits=nn_out[..., [1]],
        inflated_loc_logits=nn_out[..., [2]],
        require_integer_total_count=False,
    )


NAME_TO_LIKELIHOOD_MODEL = {
    'normal_likelihood_logistic_noise': NormalLikelihoodLogisticNoise,
    'bounded_normal_likelihood_logistic_noise': (
        BoundedNormalLikelihoodLogisticNoise
    ),
    'normal_likelihood_lognormal_noise': NormalLikelihoodLogNormalNoise,
    'normal_likelihood_varying_noise': NormalLikelihoodVaryingNoise,
    'negative_binomial': NegativeBinomial,
    'zero_inflated_negative_binomial': ZeroInflatedNegativeBinomial,
}


def get_likelihood_model(
    likelihood_model: str, likelihood_parameters: dict[str, Any]
) -> Any:
  # Actually returns a Likelihood model, but pytype thinks it returns a
  # Union[NegativeBinomial, ...].
  m = NAME_TO_LIKELIHOOD_MODEL[likelihood_model]()
  for k, v in likelihood_parameters.items():
    if hasattr(m, k):
      setattr(m, k, v)
  return m
