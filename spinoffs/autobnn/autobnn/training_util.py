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
"""Utilities for training BNNs."""

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import PyTree  # pylint: disable=g-importing-member
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from autobnn import bnn
from autobnn import util


def smape(y, yhat):
  """Return the symmetric mean absolute percentage error.

  Args:
    y: An array containing the true values.
    yhat: An array containing the predicted values.

  Returns:
    The scalar SMAPE.
  """
  # https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
  assert len(yhat) == len(y)
  h = len(y)
  errors = np.abs(y - yhat) / (np.abs(y) + np.abs(yhat)) * 100
  return 2/h * np.sum(errors)


def mase(y, yhat, y_obs, m):
  """Return the mean absolute scaled error.

  Args:
      y: An array containing the true values.
      yhat: An array containing the predicted values.
      y_obs: An array containing the training values.
      m: The season length.

  Returns:
    The scalar MASE.
  """
  # https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
  assert len(yhat) == len(y)
  n = len(y_obs)
  h = len(y)
  assert 0 < m < len(y_obs)
  numer = np.sum(np.abs(y - yhat))
  denom = np.sum(np.abs(y_obs[m:] - y_obs[:-m])) / (n - m)
  return (1 / h) * (numer / denom)


def msis(y, yhat_lower, yhat_upper, y_obs, m, a=0.05):
  """Return the mean scaled interval score.

  Args:
    y: An array containing the true values.
    yhat_lower: An array containing the a% quantile of the predicted
      distribution.
    yhat_upper: An array containing the (1-a)% quantile of the
      predicted distribution.
    y_obs: An array containing the training values.
    m: The season length.
    a: A scalar in [0, 1] specifying the quantile window to evaluate.

  Returns:
    The scalar MSIS.
  """
  # https://www.uber.com/blog/m4-forecasting-competition/
  assert len(y) == len(yhat_lower) == len(yhat_upper)
  n = len(y_obs)
  h = len(y)
  numer = np.sum(
      (yhat_upper - yhat_lower)
      + (2 / a) * (yhat_lower - y) * (y < yhat_lower)
      + (2 / a) * (y - yhat_upper) * (yhat_upper < y))
  denom = np.sum(np.abs(y_obs[m:] - y_obs[:-m])) / (n - m)
  return (1 / h) * (numer / denom)


def _make_bayeux_model(
    net: bnn.BNN,
    seed: jax.Array,
    x_train: jax.Array,
    y_train: jax.Array,
    num_particles: int = 8,
    for_vi: bool = False,
):
  """Use a MAP estimator to fit a BNN."""
  # We can't import bayeux at the file level because it would create a
  # circular dependency:  autobnn imports bayeux imports tfp:jax
  # which in turn imports (through __init__.py files) autobnn.
  import bayeux as bx  # pylint:disable=g-bad-import-order,g-import-not-at-top

  transform, inverse_transform, ildj = util.make_transforms(net)

  @jax.jit
  def _init(rand_seed):
    return net.init(rand_seed, x_train)

  initial_state = jax.vmap(_init)(jax.random.split(seed, num_particles))
  # It is okay to reuse the initial_state[0] as the test point, as Bayeux
  # only uses it to figure out the treedef.
  test_point = jax.tree_map(lambda t: t[0], initial_state)

  if for_vi:

    def log_density(params, *, seed=None):
      # The TFP VI machinery tries passing in a `seed` parameter, splats
      # dictionaries that are used for arguments, and adds a batch dimension
      # of size [1] to the start, so we undo all of that.
      del seed
      return net.log_prob(
          {'params': jax.tree_map(lambda x: x[0, ...], params)},
          data=x_train,
          observations=y_train)

  else:
    log_density = functools.partial(
        net.log_prob, data=x_train, observations=y_train)
  return bx.Model(
      log_density=jax.jit(log_density),
      test_point=test_point,
      initial_state=initial_state,
      transform_fn=transform,
      inverse_transform_fn=inverse_transform,
      inverse_log_det_jacobian=ildj)


@jax.named_call
def fit_bnn_map(
    net: bnn.BNN,
    seed: jax.Array,
    x_train: jax.Array,
    y_train: jax.Array,
    num_particles: int = 8,
    **optimizer_kwargs,
) -> Tuple[PyTree, dict[str, jax.Array]]:
  """Use a MAP estimator to fit a BNN."""
  optimizer_kwargs['num_particles'] = num_particles
  model_seed, optimization_seed = jax.random.split(seed)
  model = _make_bayeux_model(net, model_seed, x_train, y_train, num_particles)
  res = model.optimize.optax_adam(seed=optimization_seed, **optimizer_kwargs)  # pytype: disable=attribute-error
  params = res.params
  loss = res.loss
  return params, {'loss': loss}


def _filter_stuck_chains(params):
  """Rough heuristic for stuck MCMC parameters.

  1. Compute the z scores of the noise_scale `variances`.
  2. Compute the z score of 0.
  3. Filter parameters with z score more than halfway to the score for 0.

  If there are 0 or 1 chains left, just return the two with the biggest
  variances. These might be stuck!

  Args:
    params: Nested dictionary with a `noise_scale` key.

  Returns:
    A dictionary with the same structure, but only leading dimensions with
    a reasonable amount of variance in the noise parameter.
  """
  # TODO(colcarroll): Use a better heuristic here for filtering stuck chains.
  if 'noise_scale' not in params['params']:
    return params
  stds = jnp.std(params['params']['noise_scale'].squeeze(), axis=1)
  stds_mu, stds_scale = jnp.mean(stds), jnp.std(stds)

  z_scores = (stds - stds_mu) / stds_scale
  halfway_to_zero = -0.5 * stds_mu / stds_scale
  unstuck = jnp.where(z_scores > halfway_to_zero)[0]
  if unstuck.shape[0] > 2:
    return jax.tree_map(lambda x: x[unstuck], params)
  best_two = jnp.argsort(stds)[-2:]
  return jax.tree_map(lambda x: x[best_two], params)


@jax.named_call
def fit_bnn_vi(
    net: bnn.BNN,
    seed: jax.Array,
    x_train: jax.Array,
    y_train: jax.Array,
    batch_size: int = 16,
    num_draws: int = 128,
    **vi_kwargs,
) -> Tuple[PyTree, dict[str, jax.Array]]:
  """Use a MAP estimator to fit a BNN."""
  vi_kwargs['batch_size'] = batch_size
  vi_kwargs['num_samples'] = num_draws
  model_seed, vi_seed, draw_seed = jax.random.split(seed, num=3)
  model = _make_bayeux_model(
      net, model_seed, x_train, y_train, batch_size, for_vi=True)
  surrogate_dist, loss = model.vi.tfp_factored_surrogate_posterior(  # pytype: disable=attribute-error
      seed=vi_seed, **vi_kwargs)
  params = surrogate_dist.sample(seed=draw_seed, sample_shape=num_draws)

  params = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), params)
  return params, {'loss': loss}


@jax.named_call
def fit_bnn_mcmc(
    net: bnn.BNN,
    seed: jax.Array,
    x_train: jax.Array,
    y_train: jax.Array,
    num_chains: int = 128,
    num_draws: int = 8,
    **sampler_kwargs,
) -> Tuple[PyTree, dict[str, jax.Array]]:
  """Use a MAP estimator to fit a BNN."""
  sampler_kwargs['num_chains'] = num_chains
  sampler_kwargs['num_samples'] = num_draws
  sampler_kwargs['return_pytree'] = True
  model_seed, mcmc_seed = jax.random.split(seed)
  model = _make_bayeux_model(net, model_seed, x_train, y_train, num_chains)
  params = model.mcmc.numpyro_nuts(seed=mcmc_seed, **sampler_kwargs)  # pytype: disable=attribute-error

  # TODO(colcarroll): This function should instead reliably return `params``
  # with shape (num_chains * num_samples, ...), but looking at per-chain metrics
  # is the easiest way to determine where "stuck chains" occur, and it is
  # nice to return parameters with a single batch dimension.
  params = _filter_stuck_chains(params)
  params = jax.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), params)
  return params, {'noise_scale': params['params'].get('noise_scale', None)}


def _plot_loss_fn(losses, ax=None, log_scale=True) -> plt.Figure:
  """Plot losses from optimization."""
  if ax is None:
    fig, ax = plt.subplots(figsize=(16, 4), constrained_layout=True)
  else:
    fig = ax.figure
  flat_losses = losses.reshape((-1, losses.shape[-1])).T
  step = jnp.arange(flat_losses.shape[0])
  ax.plot(step, flat_losses, '-', alpha=0.5)

  x_val = int(step.max() * 0.75)

  xlim = (x_val, step.max())
  ylim = (0.95 * flat_losses.min(), 1.05 * flat_losses[x_val:].max())
  axins = ax.inset_axes(
      [0.5, 0.5, 0.47, 0.4],  # This is in axis units, not data
      xlim=xlim,
      ylim=ylim,
      xticklabels=[],
  )
  axins.plot(step[x_val:], flat_losses[x_val:], '-', alpha=0.8)
  if log_scale:
    axins.set_yscale('log')
  ax.indicate_inset_zoom(axins, edgecolor='black')
  if log_scale:
    ax.set_yscale('log')
  ax.set_title('Loss')
  return fig


def make_predictions(params, net: bnn.BNN, x_test: jax.Array) -> jax.Array:
  """Use a (batch of) parameters to make a prediction on x_test data."""
  f = jax.jit(lambda p: net.apply(p, x_test))
  return jax.vmap(f)(params)


def make_results_dataframe(
    predictions: jax.Array,
    y_test: jax.Array,
    y_train: jax.Array,
    p2_5: jax.Array,
    p90: jax.Array,
    p97_5: jax.Array,
) -> pd.core.frame.DataFrame:
  """Compute metrics and put into a dataframe for serialization.

  Note that all data is expected to be untransformed.

  Args:
    predictions: Unscaled predictions of the data.
    y_test: Unscaled testing data.
    y_train: Unscaled training data.
    p2_5: 2.5th percentile prediction.
    p90: 90th percentile prediction.
    p97_5: 97.5th percentile prediction.

  Returns:
    Dataframe with predictions and metrics.
  """
  y_test = y_test.squeeze()
  y_train = y_train.squeeze()
  n_test = len(y_test)

  # Write metrics to dataframe
  # TODO(ursk): Compute the correct metrics for each dataset here, i.e.
  # 'm3': [smape, mase, msis],
  # 'traffic': [wmppl, wmape],
  # 'm5': [wrmsse, wspl]
  smapes = np.array([smape(y_test[:i], predictions[:i])
                     for i in range(1, n_test+1)])
  mases = np.array([mase(y_test[:i], predictions[:i], y_train, 12)
                    for i in range(1, n_test+1)])
  msises = np.array(
      [msis(y_test[:i], p2_5[:i], p97_5[:i], y_train, 12)
       for i in range(1, n_test + 1)])
  return pd.DataFrame(
      data=np.array([predictions, p2_5, p90, p97_5,
                     y_test, smapes, mases, msises]).T,
      columns=['yhat', 'yhat_lower', 'p90', 'yhat_upper',
               'y', 'smape', 'mase', 'msis'])


def _plot_noise_fn(
    noise_scale: jax.Array, ax_t: plt.Axes, ax_h: plt.Axes
) -> plt.Figure:
  ax_t.plot(noise_scale.squeeze())
  ax_t.set_xlim(0, noise_scale.shape[0])
  ax_t.set_ylim(0)

  ax_h.hist(noise_scale.squeeze(), orientation='horizontal')
  ax_h.set_ylim(0)
  ax_h.axes.get_yaxis().set_visible(False)
  ax_h.axes.get_xaxis().set_visible(False)
  return ax_t.figure


def plot_results(
    dates_preds: Union[Sequence[np.datetime64], jax.Array],
    preds: jax.Array,
    *,
    dates_test: Union[Sequence[np.datetime64], jax.Array, None] = None,
    y_test: Optional[jax.Array] = None,
    p2_5: Optional[jax.Array] = None,
    p50: Optional[jax.Array] = None,
    p97_5: Optional[jax.Array] = None,
    dates_train: Union[Sequence[np.datetime64], jax.Array, None] = None,
    y_train: Optional[jax.Array] = None,
    diagnostics: Optional[Dict[str, jax.Array]] = None,
    log_scale: bool = False,
    show_particles: bool = True,
    left_limit: int = 24*7*2,
    right_limit: int = 24*7*2,
) -> plt.Figure:
  """Plot the results of `fit_bnn_map`."""
  if diagnostics is None:
    diagnostics = {}
  if diagnostics.get('loss') is not None:
    fig, (aux_ax, res_ax) = plt.subplots(
        figsize=(16, 6), nrows=2, constrained_layout=True
    )
    _plot_loss_fn(diagnostics['loss'], ax=aux_ax, log_scale=log_scale)
  elif diagnostics.get('noise_scale') is not None:
    fig = plt.figure(figsize=(16, 6), constrained_layout=True)
    axes = fig.subplot_mosaic('tttth;rrrrr')
    _plot_noise_fn(diagnostics['noise_scale'], axes['t'], axes['h'])
    res_ax = axes['r']
  else:
    fig, res_ax = plt.subplots(figsize=(16, 3), constrained_layout=True)

  if show_particles:
    for idx, p in enumerate(preds):
      res_ax.plot(
          dates_preds,
          p,
          'k-',
          alpha=0.1,
          label='Particle predictions' if idx == 0 else None,
      )

  color = 'steelblue'
  if p50 is not None:
    res_ax.plot(
        dates_preds, p50, '-', lw=2.5, color=color, label='Prediction')
  if p97_5 is not None and p2_5 is not None:
    res_ax.plot(dates_preds, p97_5, '-',
                lw=1.5, color=color, label='Upper/lower bound')
    res_ax.plot(dates_preds, p2_5, '-', lw=1.5, color=color)
    res_ax.fill_between(
        dates_preds, p2_5, p97_5, color=color, alpha=0.2
    )

  if dates_train is not None and y_train is not None:
    res_ax.plot(
        dates_train,
        y_train,
        'k-',
        label='Ground truth data',
        )
  if dates_test is not None and y_test is not None:
    res_ax.plot(
        dates_test,
        y_test,
        'k-',
        )
  res_ax.set_title('Predictions')
  res_ax.legend()
  left_limit = min(len(dates_preds) - len(dates_test), left_limit)
  right_limit = min(len(dates_test) - 1, right_limit)
  first_test_point = np.where(dates_preds == dates_test[0])[0][0]
  # TODO(ursk): Rather than modifying xlim, don't plot invisible points at all.
  res_ax.set_xlim([dates_preds[first_test_point-left_limit],
                   dates_preds[first_test_point+right_limit]])
  res_ax.axvline(dates_preds[first_test_point], linestyle='--')
  return fig


def get_params_batch_length(params: PyTree) -> int:
  """Get the batch length from a params dictionary."""
  return jax.tree_util.tree_leaves(params)[0].shape[0]


def debatchify_params(params: PyTree) -> List[Dict[str, Any]]:
  """Nested dict of rank n tensors -> a list of nested dicts of rank n-1's."""
  n = get_params_batch_length(params)
  def get_item(i):
    return jax.tree_map(lambda x: x[i, ...], params)

  return [get_item(i) for i in range(n)]
