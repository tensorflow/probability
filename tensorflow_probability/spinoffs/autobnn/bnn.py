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
"""Base class for Bayesian Neural Networks."""

import dataclasses

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from tensorflow_probability.spinoffs.autobnn import likelihoods
from tensorflow_probability.substrates.jax.distributions import distribution as distribution_lib


@jax.named_call
def log_prior_of_parameters(params, distributions) -> Float:
  """Return the prior of the parameters according to the distributions."""
  if 'params' in params:
    params = params['params']
  # We can't use jax.tree_util.tree_map here because params is allowed to
  # have extra things (like bnn_0, ... for a BnnOperator) that aren't in
  # distributions.
  lp = 0.0
  for k, v in distributions.items():
    p = params[k]
    if isinstance(v, distribution_lib.Distribution):
      lp += jnp.sum(v.log_prob(p))
    else:
      lp += log_prior_of_parameters(p, v)
  return lp


class BayesianModule(nn.Module):
  """A linen.Module with distributions over its parameters.

  Example usage:
    class MyModule(BayesianModule):

      def distributions(self):
        return {'dense': {'kernel': tfd.Normal(loc=0, scale=1),
                          'bias': tfd.Normal(loc=0, scale=1)},
                'amplitude': tfd.LogNormal(loc=0, scale=1)}

      def setup(self):
        self.dense = nn.Dense(50)
        super().setup()   # <-- Very important, do not forget!

      def __call__(self, inputs):
        return self.amplitude * self.dense(inputs)


    my_bnn = MyModule()
    params = my_bnn.init(jax.random.PRNGKey(0), jnp.zeros(10))
    lp = my_bnn.log_prior(params)

  Note that in this example, self.amplitude will be initialized using
  the given tfd.LogNormal distribution, but the self.dense's parameters
  will be initialized using the nn.Dense's default initializers.  However,
  the log_prior score will take into account all of the parameters.
  """

  def distributions(self):
    """Return a nested dictionary of distributions for the model's params.

    The nested dictionary should have the same structure as the
    variables returned by the init() method, except all leaves should
    be tensorflow probability Distributions.
    """
    # TODO(thomaswc): Consider having this optionally also be able to
    # return a tfd.JointNamedDistribution, so as to support dependencies
    # between the subdistributions.
    raise NotImplementedError('Subclasses of BNN must define this.')

  def setup(self):
    """Children classes must call this from their setup() !"""

    def make_sample_func(dist):
      def sample_func(key, shape):
        return dist.sample(sample_shape=shape, seed=key)

      return sample_func

    for k, v in self.distributions().items():
      # Create a variable for every distribution that doesn't already
      # have one.  If you define a variable in your setup, we assume
      # you initialize it correctly.
      if not hasattr(self, k):
        try:
          setattr(self, k, self.param(k, make_sample_func(v), 1))
        except flax.errors.NameInUseError:
          # Sometimes subclasses will have parameters where the
          # parameter name doesn't exactly correspond to the name of
          # the object field.  This can happen with arrays of parameters
          # (like PolynomialBBN's hidden parameters.) for example.  I
          # don't know of any way to detect this beforehand except by
          # trying to call self.params and having it fail with NameInUseError.
          # (For example, self.variables doesn't exist at setup() time.)
          pass

  def log_prior(self, params) -> float:
    """Return the log probability of the params according to the prior."""
    return log_prior_of_parameters(params, self.distributions())

  def shortname(self) -> str:
    """Return the class name, minus any BNN suffix."""
    return type(self).__name__.removesuffix('BNN')

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    return self.shortname()


class BNN(BayesianModule):
  """A Bayesian Neural Network.

  A BNN's __call__ method must accept a tensor of shape (..., num_features)
  and return a tensor of shape (..., likelihood_model.num_outputs()).
  Given that, it provides log_likelihood and log_prob methods based
  on the provided likelihood_model.
  """

  likelihood_model: likelihoods.LikelihoodModel = dataclasses.field(
      default_factory=likelihoods.NormalLikelihoodLogisticNoise
  )

  def distributions(self):
    # Children classes must call super().distributions() to include this!
    return self.likelihood_model.distributions()

  def set_likelihood_model(self, likelihood_model: likelihoods.LikelihoodModel):
    self.likelihood_model = likelihood_model

  def log_likelihood(
      self,
      params: PyTree,
      data: Float[Array, 'time features'],
      observations: Float[Array, 'time'],
  ) -> Float[Array, '']:
    """Return the likelihood of the data given the model."""
    nn_out = self.apply(params, data)
    if 'params' in params:
      params = params['params']
    # Sum over all axes here - user should use `vmap` for batching.
    return jnp.sum(
        self.likelihood_model.log_likelihood(params, nn_out, observations)
    )

  def log_prob(
      self,
      params: PyTree,
      data: Float[Array, 'time features'],
      observations: Float[Array, 'time'],
  ) -> Float[Array, '']:
    return self.log_prior(params) + self.log_likelihood(
        params, data, observations
    )

  def get_all_distributions(self):
    return self.distributions()
