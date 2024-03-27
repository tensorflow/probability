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
"""Flax.linen modules for combining BNNs."""

import functools
from typing import Optional
from flax import linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.spinoffs.autobnn import bnn
from tensorflow_probability.spinoffs.autobnn import likelihoods
from tensorflow_probability.substrates.jax.bijectors import chain as chain_lib
from tensorflow_probability.substrates.jax.bijectors import scale as scale_lib
from tensorflow_probability.substrates.jax.bijectors import shift as shift_lib
from tensorflow_probability.substrates.jax.distributions import beta as beta_lib
from tensorflow_probability.substrates.jax.distributions import dirichlet as dirichlet_lib
from tensorflow_probability.substrates.jax.distributions import half_normal as half_normal_lib
from tensorflow_probability.substrates.jax.distributions import normal as normal_lib
from tensorflow_probability.substrates.jax.distributions import transformed_distribution as transformed_distribution_lib


Array = jnp.ndarray


class BnnOperator(bnn.BNN):
  """Base class for BNNs that are made from other BNNs."""
  bnns: tuple[bnn.BNN, ...] = tuple()

  def setup(self):
    assert self.bnns, 'Forgot to pass `bnns` keyword argument?'
    super().setup()

  def set_likelihood_model(self, likelihood_model: likelihoods.LikelihoodModel):
    super().set_likelihood_model(likelihood_model)
    # We need to set the likelihood models on the component
    # bnns so that they will know how many outputs they are
    # supposed to have.  BUT:  we also don't want to accidentally
    # create any additional variables, distributions or parameters
    # in them.  So we set them all to having a dummy likelihood
    # model that only knows how many outputs it has.
    dummy_ll_model = likelihoods.DummyLikelihoodModel(
        num_outs=likelihood_model.num_outputs()
    )
    for b in self.bnns:
      b.set_likelihood_model(dummy_ll_model)

  @jax.named_call
  def log_prior(self, params):
    if 'params' in params:
      params = params['params']
    # params for bnns[i] are stored in params['bnns_{i}'].
    lp = bnn.log_prior_of_parameters(params, self.distributions())
    for i, b in enumerate(self.bnns):
      params_field = f'bnns_{i}'
      if params_field in params:
        lp += b.log_prior(params[params_field])
    return lp

  def get_all_distributions(self):
    distributions = self.distributions()
    for idx, sub_bnn in enumerate(self.bnns):
      d = sub_bnn.get_all_distributions()
      if d:
        distributions[f'bnns_{idx}'] = d
    return distributions

  def summary_join_string(self, params) -> str:
    """String to use when joining the component summaries."""
    raise NotImplementedError()

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    params = params or {}
    if 'params' in params:
      params = params['params']

    names = [
        b.summarize(params.get(f'bnns_{i}'), full)
        for i, b in enumerate(self.bnns)
    ]

    return f'({self.summary_join_string(params).join(names)})'


class MultipliableBnnOperator(BnnOperator):
  """Abstract base class for a BnnOperator that can be multiplied."""
  # Ideally, this would just inherit from both BnnOperator and
  # kernels.MultipliableBNN, but pytype gets really confused by that.
  going_to_be_multiplied: bool = False

  def setup(self):
    if self.going_to_be_multiplied:
      for b in self.bnns:
        assert b.going_to_be_multiplied
    else:
      for b in self.bnns:
        assert not getattr(b, 'going_to_be_multiplied', False)
    super().setup()

  def penultimate(self, inputs):
    raise NotImplementedError(
        'Subclasses of MultipliableBnnOperator must define this.')


class Add(MultipliableBnnOperator):
  """Add two or more BNNs."""

  @functools.partial(jax.named_call, name='Add::penultimate')
  def penultimate(self, inputs):
    penultimates = [b.penultimate(inputs) for b in self.bnns]
    return jnp.sum(jnp.stack(penultimates, axis=-1), axis=-1)

  def __call__(self, inputs, deterministic=True):
    return jnp.sum(
        jnp.stack([b(inputs) for b in self.bnns], axis=-1),
        axis=-1)

  def summary_join_string(self, params) -> str:
    return '#'


class WeightedSum(MultipliableBnnOperator):
  """Add two or more BNNs, with weights taken from a Dirichlet prior."""

  # `alpha=1` is a uniform prior on mixing weights, higher values will favor
  # weights like `1/n`, and lower weights will favor sparsity.
  alpha: float = 1.0
  num_outputs: int = 1

  def distributions(self):
    bnn_concentrations = [1.0 if isinstance(b, BnnOperator) else 1.5
                          for b in self.bnns]
    if self.going_to_be_multiplied:
      concentration = self.alpha * jnp.array(bnn_concentrations)
    else:
      concentration = self.alpha * jnp.array(
          [bnn_concentrations for _ in range(self.num_outputs)])
    return super().distributions() | {
        'bnn_weights': dirichlet_lib.Dirichlet(concentration=concentration)
    }

  @functools.partial(jax.named_call, name='WeightedSum::penultimate')
  def penultimate(self, inputs):
    penultimates = [
        b.penultimate(inputs) * self.bnn_weights[0, i]
        for i, b in enumerate(self.bnns)
    ]
    return jnp.sum(jnp.stack(penultimates, axis=-1), axis=-1)

  @functools.partial(jax.named_call, name='WeightedSum::__call__')
  def __call__(self, inputs, deterministic=True):
    return jnp.sum(
        jnp.stack(
            [
                b(inputs) * self.bnn_weights[0, :, i]
                for i, b in enumerate(self.bnns)
            ],
            axis=-1,
        ),
        axis=-1,
    )

  def summarize(self, params=None, full: bool = False) -> str:
    """Return a string summarizing the structure of the BNN."""
    params = params or {}
    if 'params' in params:
      params = params['params']

    names = [
        b.summarize(params.get(f'bnns_{i}'), full)
        for i, b in enumerate(self.bnns)
    ]

    def pretty_print(w):
      try:
        s = f'{jnp.array_str(jnp.array(w), precision=3)}'
      except Exception:  # pylint: disable=broad-exception-caught
        try:
          s = f'{w:.3f}'
        except Exception:  # pylint: disable=broad-exception-caught
          s = f'{w}'
      return s.replace('\n', ' ')

    weights = params.get('bnn_weights')
    if weights is not None:
      weights = jnp.array(weights)[0].T.squeeze()
      names = [
          f'{pretty_print(w)} {n}'
          for w, n in zip(weights, names)
          if full or jnp.max(w) > 0.04
      ]

    return f'({"+".join(names)})'


class Multiply(BnnOperator):
  """Multiply two or more BNNs."""

  def setup(self):
    self.dense = nn.Dense(self.likelihood_model.num_outputs())
    for b in self.bnns:
      assert hasattr(b, 'penultimate')
      assert b.going_to_be_multiplied, 'Forgot to set going_to_be_multiplied?'
    super().setup()

  def distributions(self):
    return super().distributions() | {
        'dense': {
            'kernel': normal_lib.Normal(loc=0, scale=1.0),
            'bias': normal_lib.Normal(loc=0, scale=1.0),
        }
    }

  @functools.partial(jax.named_call, name='Multiply::__call__')
  def __call__(self, inputs, deterministic=True):
    penultimates = [b.penultimate(inputs) for b in self.bnns]
    return self.dense(jnp.prod(jnp.stack(penultimates, axis=-1), axis=-1))

  def summary_join_string(self, params) -> str:
    return '*'


class ChangePoint(BnnOperator):
  """Switch from one BNN to another based on a time point."""
  change_point: float = 0.0
  slope: float = 1.0
  change_index: int = 0

  def setup(self):
    assert len(self.bnns) == 2
    super().setup()

  @jax.named_call
  def __call__(self, inputs, deterministic=True):
    time = inputs[..., self.change_index, jnp.newaxis]
    y = (time - self.change_point) / self.slope
    return nn.sigmoid(y) * self.bnns[1](inputs) + nn.sigmoid(
        -y) * self.bnns[0](inputs)

  def summary_join_string(self, params) -> str:
    return f'<[{self.change_point}]'


class LearnableChangePoint(BnnOperator):
  """Switch from one BNN to another based on a time point."""
  time_series_xs: Optional[Array] = None
  change_index: int = 0

  def distributions(self):
    assert self.time_series_xs is not None
    lo = jnp.min(self.time_series_xs)
    hi = jnp.max(self.time_series_xs)
    # We want change_slope_scale to be the average value of
    # time_series_xs[i+1] - time_series_xs[i]
    change_slope_scale = (hi - lo) / self.time_series_xs.size

    # this distribution puts a lower density at the endpoints, and a reasonably
    # flat distribution near the middle of the timeseries.
    bij = chain_lib.Chain([shift_lib.Shift(lo), scale_lib.Scale(hi - lo)])
    dist = transformed_distribution_lib.TransformedDistribution(
        distribution=beta_lib.Beta(1.5, 1.5), bijector=bij
    )
    return super().distributions() | {
        'change_point': dist,
        'change_slope': half_normal_lib.HalfNormal(scale=change_slope_scale),
    }

  def setup(self):
    assert len(self.bnns) == 2
    assert len(self.time_series_xs) >= 2
    super().setup()

  @functools.partial(jax.named_call, name='LearnableChangePoint::__call__')
  def __call__(self, inputs, deterministic=True):
    time = inputs[..., self.change_index, jnp.newaxis]
    y = (time - self.change_point) / self.change_slope
    return nn.sigmoid(y) * self.bnns[1](inputs) + nn.sigmoid(-y) * self.bnns[0](
        inputs
    )

  def summary_join_string(self, params) -> str:
    params = params or {}
    if 'params' in params:
      params = params['params']
    change_point = params.get('change_point')
    cp_str = ''
    if change_point is not None:
      cp_str = f'[{jnp.array_str(change_point, precision=2)}]'
    return f'<{cp_str}'
