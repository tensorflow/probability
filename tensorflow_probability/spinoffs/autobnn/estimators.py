# Copyright 2024 The TensorFlow Probability Authors.
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
"""Estimator classes for training BNN models using Bayeux."""

from typing import Any, Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree  # pylint: disable=g-importing-member,g-multiple-import
from tensorflow_probability.spinoffs.autobnn import bnn
from tensorflow_probability.spinoffs.autobnn import likelihoods
from tensorflow_probability.spinoffs.autobnn import models
from tensorflow_probability.spinoffs.autobnn import training_util


class _AutoBnnEstimator:
  """Estimator class based on scikit-learn estimators.

  See https://scikit-learn.org/stable/developers/develop.html
  """

  def __init__(
      self,
      model_or_name: Union[str, bnn.BNN],
      likelihood_model: str,
      seed: jax.Array,
      width: int = 50,
      periods: Sequence[ArrayLike] = (12.0,),
      likelihood_kwargs: Optional[Mapping[str, Any]] = None,
  ):
    self.model_or_name = model_or_name
    self.likelihood_model = likelihood_model
    self.width = width
    self.periods = periods
    self.seed = seed
    if likelihood_kwargs is not None:
      self.likelihood_kwargs = likelihood_kwargs
    else:
      self.likelihood_kwargs = {}

    self.net_: Optional[bnn.BNN] = None
    self.params_: Optional[PyTree] = None
    self.diagnostics_: Optional[dict[str, jax.Array]] = None
    self.fit_seed_: Optional[jax.Array] = None
    self.predict_quantiles_seed_: Optional[jax.Array] = None
    self.likelihood_: Optional[likelihoods.LikelihoodModel] = None
    self.kwargs = {}

  def check_is_fitted(self) -> bool:
    # The model is fit if any of the real variables (those with trailing
    # underscores) have non-None values. Note that scikit-learn does not
    # initialize these values at all, but we do to help type hinting.
    for v in vars(self):
      if (
          v.endswith('_')
          and not v.startswith('__')
          and (getattr(self, v) is not None)
      ):
        return True
    return False

  def _fit(
      self, net, seed, x_train, y_train, **kwargs
  ) -> tuple[PyTree, dict[str, jax.Array]]:
    raise NotImplementedError()

  def fit(self, X: jax.Array, y: jax.Array) -> '_AutoBnnEstimator':  # pylint: disable=invalid-name
    """Fit the model according to the given training data.

    Args:
      X: Training array, where the rows are samples, and the column are the
        number of features (assumed to be 1).
      y: Target vector relative to X.

    Returns:
      A fitted estimator.
    """
    self.likelihood_ = likelihoods.get_likelihood_model(
        self.likelihood_model, self.likelihood_kwargs
    )
    self.net_ = models.make_model(
        model_name=self.model_or_name,
        likelihood_model=self.likelihood_,
        time_series_xs=X,
        width=self.width,
        periods=self.periods,
    )
    self.fit_seed_, self.predict_quantiles_seed_ = jax.random.split(self.seed)
    self.params_, self.diagnostics_ = self._fit(
        net=self.net_, seed=self.seed, x_train=X, y_train=y, **self.kwargs
    )
    return self

  def predict(self, X: jax.Array) -> jax.Array:  # pylint: disable=invalid-name
    if not self.check_is_fitted():
      raise RuntimeError(
          'Model is not yet fit! Call `AutoBNN.fit(X, y)` first.'
      )

    # Not every distribution has an analytic mean, and `self.net_.apply` does
    # not always return a mean.
    return self.predict_quantiles(X, q=50.0, axis=())

  def fit_predict(self, X: jax.Array, y: jax.Array) -> jax.Array:  # pylint: disable=invalid-name
    if not self.check_is_fitted():
      raise RuntimeError(
          'Model is not yet fit! Call `AutoBNN.fit(X, y)` first.'
      )
    self.fit(X, y)
    return self.predict(X)

  def summary(self) -> str:
    if self.net_ is None:
      return ''
    if self.params_ is None:
      return self.net_.summarize(None)
    params_per_particle = training_util.debatchify_params(self.params_)
    summaries = [self.net_.summarize(p) for p in params_per_particle]
    return '\n'.join(summaries)

  @jax.named_call
  def predict_quantiles(
      self, X: jax.Array, q=(2.5, 50.0, 97.5), axis: tuple[int, ...] = (0,)  # pylint: disable=invalid-name
  ) -> jax.Array:
    """Predict quantiles over the time points in X.

    This uses the fit params_ in this class, which has a single batch dimension.
    By default, the function will compute a quantiles over this batch dimension,
    but can compute per-particle quantiles by passing `axis=()`.

    Args:
      X: Training array, where the rows are samples, and the column are the
        number of features (assumed to be 1).
      q: Quantiles in [0, 100] to compute.
      axis: Tuple of dimensions to reduce over.

    Raises:
      RuntimeError: If model is not fit.

    Returns:
      Array with a row for each element of `q`.
    """
    if not self.check_is_fitted():
      raise RuntimeError(
          'Model is not yet fit! Call `AutoBNN.fit(X, y)` first.'
      )
    loc = training_util.make_predictions(self.params_, net=self.net_, x_test=X)
    assert hasattr(self.likelihood_, 'sample')

    # We are doing all quantiles via sampling currently, so the fact that we
    # also reduce over the new batch axis is an implementation detail.
    axis = (0,) + tuple(j + 1 for j in axis)
    draws = self.likelihood_.sample(
        self.params_['params'],
        jnp.squeeze(loc),
        seed=self.predict_quantiles_seed_,
        sample_shape=100,
    )
    # TODO(ursk): return these in a dictionary rather than as a tuple
    return jnp.percentile(draws, jnp.array(q), axis=axis)


class AutoBnnMapEstimator(_AutoBnnEstimator):
  """Implementation of a MAP estimator for the BNN.

  Example usage:

  estimator = estimators.AutoBnnMapEstimator(
        model_or_name='linear_plus_periodic',
        likelihood_model='normal_likelihood_logistic_noise',
        seed=jax.random.PRNGKey(42),
        width=25,
        num_particles=32,
        num_iters=1000,
  )
  estimator.fit(x_train, y_train)
  low, mid, high = estimator.predict_quantiles(x_train)

  Or:

  estimator = estimators.AutoBnnMapEstimator(
        model_or_name=operators.Add(
            bnns=(kernels.LinearBNN(width=50),
                  kernels.PeriodicBNN(width=50, period=12))),
        likelihood_model='normal_likelihood_lognormal_noise',
        seed=jax.random.PRNGKey(123))
  """

  def __init__(
      self,
      model_or_name: Union[str, bnn.BNN],
      likelihood_model: str,
      seed: jax.Array,
      width: int = 50,
      periods: Sequence[ArrayLike] = (12.0,),
      num_iters: int = 5_000,
      num_particles: int = 8,
      learning_rate: float = 0.01,
      likelihood_kwargs: Optional[Mapping[str, Any]] = None,
      **unused_kwargs,
  ):
    super().__init__(
        model_or_name=model_or_name,
        likelihood_model=likelihood_model,
        seed=seed,
        width=width,
        periods=periods,
        likelihood_kwargs=likelihood_kwargs,
    )
    self.num_iters = num_iters
    self.num_particles = num_particles
    self.learning_rate = learning_rate
    self.kwargs = {
        'num_iters': num_iters,
        'num_particles': num_particles,
        'learning_rate': learning_rate,
    }

    self.diagnostics_ = None
    self._fit = training_util.fit_bnn_map


class AutoBnnMCMCEstimator(_AutoBnnEstimator):
  """Implementation of an MCMC estimator for the BNN."""

  def __init__(
      self,
      model_or_name: Union[str, bnn.BNN],
      likelihood_model: str,
      seed: jax.Array,
      width: int = 50,
      periods: Sequence[ArrayLike] = (12.0,),
      num_chains: int = 128,
      num_draws: int = 8,
      likelihood_kwargs: Optional[Mapping[str, Any]] = None,
      **unused_kwargs,
  ):
    super().__init__(
        model_or_name=model_or_name,
        likelihood_model=likelihood_model,
        seed=seed,
        width=width,
        periods=periods,
        likelihood_kwargs=likelihood_kwargs,
    )
    self.num_chain = num_chains
    self.num_draws = num_draws
    self.kwargs = {'num_chains': num_chains, 'num_draws': num_draws}

    self.diagnostics_ = None
    self._fit = training_util.fit_bnn_mcmc


class AutoBnnVIEstimator(_AutoBnnEstimator):
  """Implementation of a VI estimator for the BNN."""

  def __init__(
      self,
      model_or_name: Union[str, bnn.BNN],
      likelihood_model: str,
      seed: jax.Array,
      width: int = 50,
      periods: Sequence[ArrayLike] = (12.0,),
      likelihood_kwargs: Optional[Mapping[str, Any]] = None,
      batch_size: int = 16,
      num_draws: int = 128,
      **unused_kwargs,
  ):
    super().__init__(
        model_or_name=model_or_name,
        likelihood_model=likelihood_model,
        seed=seed,
        width=width,
        periods=periods,
        likelihood_kwargs=likelihood_kwargs,
    )
    self.batch_size = batch_size
    self.num_draws = num_draws
    self.kwargs = {'batch_size': batch_size, 'num_draws': num_draws}

    self.diagnostics_ = None
    self._fit = training_util.fit_bnn_vi


NAME_TO_ESTIMATOR = {
    'map': AutoBnnMapEstimator,
    'mcmc': AutoBnnMCMCEstimator,
    'vi': AutoBnnVIEstimator,
}


def get_estimator(estimator_name: str, params) -> _AutoBnnEstimator:
  return NAME_TO_ESTIMATOR[estimator_name](**params)
