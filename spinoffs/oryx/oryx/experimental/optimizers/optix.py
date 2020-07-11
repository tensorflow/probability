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
# ============================================================================
# Lint as: python3
"""Reimplementation of jax.experimental.optix.

There are a few key differences:
1. We use Oryx's state API to handle variables.
2. We include the params as an input to the optimizer so we can build optimizers
like LARS and LAMB.
"""
import itertools

import jax
from jax import lax
from jax import random
from jax import tree_util
import jax.numpy as np

from oryx.core import primitive
from oryx.core import state

tree_map = tree_util.tree_multimap
tree_leaves = tree_util.tree_leaves
tree_structure = tree_util.tree_structure
tree_unflatten = tree_util.tree_unflatten

__all__ = [
    'clip',
    'global_norm',
    'clip_by_global_norm',
    'trace',
    'scale_by_rms',
    'scale_by_stddev',
    'scale_by_adam',
    'scale',
    'scale_by_schedule',
    'add_noise',
    'apply_every',
    'chain',
    'sgd',
    'noisy_sgd',
    'adam',
    'rmsprop',
    'gradient_descent',
    'optimize'
]


def clip(max_delta):

  def update(params, updates):
    del params
    return tree_map(lambda g: np.clip(updates, -max_delta, max_delta), updates)

  return update


def global_norm(items):
  return np.sqrt(np.sum([np.sum(x**2) for x in tree_leaves(items)]))


def clip_by_global_norm(max_norm):
  """Returns a function that clips updates to a provided max norm."""

  def update(params, updates):
    del params
    g_norm = global_norm(updates)
    trigger = g_norm < max_norm
    updates = tree_map(lambda t: np.where(trigger, t, t * (max_norm / g_norm)),
                       updates)
    return updates

  return update


def trace(decay, nesterov):
  """Returns a function that combines updates with a running state."""

  def update(params, updates, init_key=None):
    del params
    if init_key is None:
      raise ValueError('`init_key` cannot be `None`.')
    tr = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates),
        key=init_key,
        name='trace')
    f = lambda g, t: g + decay * t
    update_trace = state.assign(tree_map(f, updates, tr), name='trace')
    updates = tree_map(f, updates, update_trace) if nesterov else update_trace
    return updates

  return update


def _update_moment(updates, moments, decay, order):
  return tree_map(lambda g, t: (1 - decay) * (g**order) + decay * t, updates,
                  moments)


def scale_by_rms(decay=0.9, eps=1e-8):
  """Returns a function that scales updates by the RMS of the updates."""
  def update(params, updates, init_key=None):
    del params
    if init_key is None:
      raise ValueError('`init_key` cannot be `None`.')
    nu = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates), key=init_key, name='nu')
    nu = state.assign(_update_moment(updates, nu, decay, 2), name='nu')
    updates = tree_map(lambda g, n: g / (np.sqrt(n + eps)), updates, nu)
    return updates

  return update


def scale_by_stddev(decay=0.9, eps=1e-8):
  """Returns a function that scales updates by their standard deviation."""

  def update(params, updates, init_key=None):
    del params
    if init_key is None:
      raise ValueError('`init_key` cannot be `None`.')
    mu_key, nu_key = random.split(init_key)
    mu = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates), key=mu_key, name='mu')
    nu = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates), key=nu_key, name='nu')
    mu = state.assign(_update_moment(updates, mu, decay, 1), name='mu')
    nu = state.assign(_update_moment(updates, nu, decay, 2), name='nu')
    updates = tree_map(lambda g, m, n: g / np.sqrt(n - m**2 + eps), updates, mu,
                       nu)
    return updates

  return update


def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8):
  """Scales updates according to Adam update rules."""

  def update(params, updates, init_key=None):
    del params
    if init_key is None:
      raise ValueError('`init_key` cannot be `None`.')
    count_key, mu_key, nu_key = random.split(init_key, 3)
    count = state.variable(0., key=count_key, name='count')
    mu = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates), key=mu_key, name='mu')
    nu = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates), key=nu_key, name='nu')
    mu = state.assign(_update_moment(updates, mu, b1, 1), name='mu')
    nu = state.assign(_update_moment(updates, nu, b2, 2), name='nu')
    count = state.assign(count + 1., name='count', key=updates)
    mu_hat = tree_map(lambda t: t / (1 - b1**count), mu)
    nu_hat = tree_map(lambda t: t / (1 - b2**count), nu)
    updates = tree_map(lambda m, v: m / (np.sqrt(v) + eps), mu_hat, nu_hat)
    return updates

  return update


def scale(step_size):

  def update(params, updates):
    del params
    return tree_map(lambda g: step_size * g, updates)

  return update


def scale_by_schedule(step_size_fn):
  """Returns a function that scales updates according to an input schedule."""

  def update(params, updates, init_key=None):
    del params
    if init_key is None:
      raise ValueError('`init_key` cannot be `None`.')
    count = state.variable(0., key=init_key, name='count')
    updates = tree_map(lambda g: step_size_fn(count) * g, updates)
    updates, count = primitive.tie_all(
        updates, state.assign(count + 1., name='count', key=updates))
    return updates

  return update


def add_noise(eta, gamma, seed):
  """Returns a function that adds noise to updates."""

  def update(params, updates, init_key=None):
    del params
    count_key, seed_key = random.split(init_key)
    count = state.variable(0., key=count_key, name='count')
    rng_key = state.variable(random.PRNGKey(seed), key=seed_key, name='rng_key')
    num_vars = len(tree_leaves(updates))
    treedef = tree_structure(updates)
    variance = eta / (1 + count)**gamma
    all_keys = random.split(rng_key, num_vars + 1)
    noise = tree_map(lambda g, k: random.normal(k, shape=g.shape), updates,
                     tree_unflatten(treedef, all_keys[1:]))
    updates = tree_map(lambda g, n: g + variance * n, updates, noise)
    updates, count, rng_key = primitive.tie_all(
        updates, state.assign(count + 1., name='count', key=updates),
        state.assign(all_keys[0], name='rng_key', key=updates))
    return updates

  return update


def apply_every(k=1):
  """Returns a function that accumulates updates and applies them all at once."""

  def update(params, updates, init_key=None):
    del params
    count_key, grad_acc_key = random.split(init_key)
    count = state.variable(0., key=count_key, name='count')
    grad_acc = state.variable(
        tree_map(lambda g: np.zeros(g.shape), updates),
        key=grad_acc_key,
        name='grad_acc')

    c = count % k
    acc = c != 0
    grad_acc = state.assign(
        tree_map(lambda g, ga: acc * ga + g, updates, grad_acc),
        name='grad_acc')
    emit = c == (k - 1)
    updates = tree_map(lambda ga: emit * ga, grad_acc)
    updates, count = primitive.tie_all(
        updates, state.assign(count + 1., name='count', key=updates))
    return updates

  return update


def chain(*args, **kwargs):
  """Composes update functions together serially."""
  def update(params, updates, init_key=None):
    keys = random.split(init_key, len(args) + len(kwargs))
    names = [f'update_{i}' for i in range(len(args))] + list(kwargs.keys())
    for (name, key, update_fn) in zip(
        names, keys, itertools.chain(args, kwargs.values())):
      step = state.init(update_fn, name=name)(key, params, updates)
      updates = step(params, updates)
    return updates

  return update


def sgd(learning_rate, momentum=0., nesterov=False):
  return chain(trace(decay=momentum, nesterov=nesterov), scale(-learning_rate))


def noisy_sgd(learning_rate, eta=0.01, gamma=0.55, seed=42):
  return chain(
      trace(decay=0., nesterov=False), scale(-learning_rate),
      add_noise(eta, gamma, seed))


def adam(learning_rate, b1=0.9, b2=0.999, eps=1e-8):
  return chain(scale_by_adam(b1=b1, b2=b2, eps=eps), scale(-learning_rate))


def rmsprop(learning_rate, decay=0.9, eps=1e-8, centered=False):
  if not centered:
    return chain(scale_by_rms(decay=decay, eps=eps), scale(-learning_rate))
  else:
    return chain(scale_by_stddev(decay=decay, eps=eps), scale(-learning_rate))


def apply_updates(params, updates):
  return tree_map(lambda p, u: p + u, params, updates)


def gradient_descent(update, objective):

  def step(params, *args, init_key=None):
    out, updates = jax.value_and_grad(objective)(params, *args)
    updates = primitive.tie_in(out, update(params, updates, init_key=init_key))
    return apply_updates(params, updates)

  return step


def optimize(objective, update, num_iters):
  """Runs several iterations of optimization and returns the result."""

  def run(params, init_key=None):
    opt = state.init(
        gradient_descent(update, objective), name='opt')(init_key, params)

    def body(carry, _):
      opt, params = carry
      params, opt = opt.call_and_update(params)
      return (opt, params), ()

    opt, params = lax.scan(body, (opt, params), np.arange(num_iters))[0]
    opt, params = primitive.tie_all(state.assign(opt, name='opt'), params)
    return params

  return run
