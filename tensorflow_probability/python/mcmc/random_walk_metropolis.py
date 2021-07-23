# Copyright 2018 The TensorFlow Probability Authors.
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
"""Random Walk Metropolis (RWM) Transition Kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'random_walk_normal_fn',
    'random_walk_uniform_fn',
    'RandomWalkMetropolis',
    'UncalibratedRandomWalk',
]


class UncalibratedRandomWalkResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedRandomWalkResults',
        [
            'log_acceptance_correction',
            'target_log_prob',        # For "next_state".
            'seed',
        ])
    ):
  """Internal state and diagnostics for Random Walk MH."""
  __slots__ = ()


def random_walk_normal_fn(scale=1., name=None):
  """Returns a callable that adds a random normal perturbation to the input.

  This function returns a callable that accepts a Python `list` of `Tensor`s of
  any shapes and `dtypes`  representing the state parts of the `current_state`
  and a random seed. The supplied argument `scale` must be a `Tensor` or Python
  `list` of `Tensor`s representing the scale of the generated
  proposal. `scale` must broadcast with the state parts of `current_state`.
  The callable adds a sample from a zero-mean normal distribution with the
  supplied scales to each state part and returns a same-type `list` of `Tensor`s
  as the state parts of `current_state`.

  Args:
    scale: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
      controlling the scale of the normal proposal distribution.
    name: Python `str` name prefixed to Ops created by this function.
        Default value: 'random_walk_normal_fn'.

  Returns:
    random_walk_normal_fn: A callable accepting a Python `list` of `Tensor`s
      representing the state parts of the `current_state` and an `int`
      representing the random seed to be used to generate the proposal. The
      callable returns the same-type `list` of `Tensor`s as the input and
      represents the proposal for the RWM algorithm.
  """
  def _fn(state_parts, seed, experimental_shard_axis_names=None):
    """Adds a normal perturbation to the input state.

    Args:
      state_parts: A list of `Tensor`s of any shape and real dtype representing
        the state parts of the `current_state` of the Markov chain.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.

    Returns:
      perturbed_state_parts: A Python `list` of The `Tensor`s. Has the same
        shape and type as the `state_parts`.

    Raises:
      ValueError: if `scale` does not broadcast with `state_parts`.
    """
    with tf.name_scope(name or 'random_walk_normal_fn'):
      scales = scale if mcmc_util.is_list_like(scale) else [scale]
      if len(scales) == 1:
        scales *= len(state_parts)
      if len(state_parts) != len(scales):
        raise ValueError('`scale` must broadcast with `state_parts`.')

      part_seeds = list(samplers.split_seed(seed, n=len(state_parts)))
      part_seeds = distribute_lib.fold_in_axis_index(
          part_seeds, experimental_shard_axis_names)

      next_state_parts = [
          samplers.normal(  # pylint: disable=g-complex-comprehension
              mean=state_part,
              stddev=scale_part,
              shape=ps.shape(state_part),
              dtype=dtype_util.base_dtype(state_part.dtype),
              seed=seed_part)
          for scale_part, state_part, seed_part
          in zip(scales, state_parts, part_seeds)
      ]

      return next_state_parts
  return _fn


def random_walk_uniform_fn(scale=1., name=None):
  """Returns a callable that adds a random uniform perturbation to the input.

  For more details on `random_walk_uniform_fn`, see
  `random_walk_normal_fn`. `scale` might
  be a `Tensor` or a list of `Tensor`s that should broadcast with state parts
  of the `current_state`. The generated uniform perturbation is sampled as a
  uniform point on the rectangle `[-scale, scale]`.

  Args:
    scale: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
      controlling the upper and lower bound of the uniform proposal
      distribution.
    name: Python `str` name prefixed to Ops created by this function.
        Default value: 'random_walk_uniform_fn'.

  Returns:
    random_walk_uniform_fn: A callable accepting a Python `list` of `Tensor`s
      representing the state parts of the `current_state` and an `int`
      representing the random seed used to generate the proposal. The callable
      returns the same-type `list` of `Tensor`s as the input and represents the
      proposal for the RWM algorithm.
  """
  def _fn(state_parts, seed, experimental_shard_axis_names=None):
    """Adds a uniform perturbation to the input state.

    Args:
      state_parts: A list of `Tensor`s of any shape and real dtype representing
        the state parts of the `current_state` of the Markov chain.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.

    Returns:
      perturbed_state_parts: A Python `list` of The `Tensor`s. Has the same
        shape and type as the `state_parts`.

    Raises:
      ValueError: if `scale` does not broadcast with `state_parts`.
    """
    with tf.name_scope(name or 'random_walk_uniform_fn'):
      scales = scale if mcmc_util.is_list_like(scale) else [scale]
      if len(scales) == 1:
        scales *= len(state_parts)
      if len(state_parts) != len(scales):
        raise ValueError('`scale` must broadcast with `state_parts`.')

      part_seeds = list(samplers.split_seed(seed, n=len(state_parts)))
      part_seeds = distribute_lib.fold_in_axis_index(
          part_seeds, experimental_shard_axis_names)

      next_state_parts = [
          samplers.uniform(  # pylint: disable=g-complex-comprehension
              minval=state_part - scale_part,
              maxval=state_part + scale_part,
              shape=tf.shape(state_part),
              dtype=dtype_util.base_dtype(state_part.dtype),
              seed=seed_part)
          for scale_part, state_part, seed_part
          in zip(scales, state_parts, part_seeds)
      ]
      return next_state_parts
  return _fn


class RandomWalkMetropolis(kernel_base.TransitionKernel):
  """Runs one step of the RWM algorithm with symmetric proposal.

  Random Walk Metropolis is a gradient-free Markov chain Monte Carlo
  (MCMC) algorithm. The algorithm involves a proposal generating step
  `proposal_state = current_state + perturb` by a random
  perturbation, followed by Metropolis-Hastings accept/reject step. For more
  details see [Section 2.1 of Roberts and Rosenthal (2004)](
  http://dx.doi.org/10.1214/154957804100000024).

  Current class implements RWM for normal and uniform proposals. Alternatively,
  the user can supply any custom proposal generating function.

  The function `one_step` can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics
  are governed by `target_log_prob_fn(*current_state)`. (The number of
  independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

  #### Examples:

  ##### Sampling from the Standard Normal Distribution.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dtype = np.float32

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  samples = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=dtype(1),
    kernel=tfp.mcmc.RandomWalkMetropolis(target.log_prob),
    num_burnin_steps=500,
    trace_fn=None,
    seed=42)

  sample_mean = tf.math.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
      tf.math.reduce_mean(
          tf.math.squared_difference(samples, sample_mean),
          axis=0))

  print('Estimated mean: {}'.format(sample_mean))
  print('Estimated standard deviation: {}'.format(sample_std))
  ```

  ##### Sampling from a 2-D Normal Distribution.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dtype = np.float32
  true_mean = dtype([0, 0])
  true_cov = dtype([[1, 0.5],
                    [0.5, 1]])
  num_results = 500
  num_chains = 100

  # Target distribution is defined through the Cholesky decomposition `L`:
  L = tf.linalg.cholesky(true_cov)
  target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=L)

  # Initial state of the chain
  init_state = np.ones([num_chains, 2], dtype=dtype)

  # Run Random Walk Metropolis with normal proposal for `num_results`
  # iterations for `num_chains` independent chains:
  samples = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target.log_prob),
      num_burnin_steps=200,
      num_steps_between_results=1,  # Thinning.
      trace_fn=None,
      seed=54)

  sample_mean = tf.math.reduce_mean(samples, axis=0)
  x = tf.squeeze(samples - sample_mean)
  sample_cov = tf.matmul(tf.transpose(x, [1, 2, 0]),
                         tf.transpose(x, [1, 0, 2])) / num_results

  mean_sample_mean = tf.math.reduce_mean(sample_mean)
  mean_sample_cov = tf.math.reduce_mean(sample_cov, axis=0)
  x = tf.reshape(sample_cov - mean_sample_cov, [num_chains, 2 * 2])
  cov_sample_cov = tf.reshape(tf.matmul(x, x, transpose_a=True) / num_chains,
                              shape=[2 * 2, 2 * 2])

  print('Estimated mean: {}'.format(mean_sample_mean))
  print('Estimated avg covariance: {}'.format(mean_sample_cov))
  print('Estimated covariance of covariance: {}'.format(cov_sample_cov))
  ```

  ##### Sampling from the Standard Normal Distribution using Cauchy proposal.

  ```python
  import numpy as np
  import tensorflow.compat.v2 as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  dtype = np.float32
  num_burnin_steps = 500
  num_chain_results = 1000

  def cauchy_new_state_fn(scale, dtype):
    cauchy = tfd.Cauchy(loc=dtype(0), scale=dtype(scale))
    def _fn(state_parts, seed):
      next_state_parts = []
      part_seeds = tfp.random.split_seed(
          seed, n=len(state_parts), salt='rwmcauchy')
      for sp, ps in zip(state_parts, part_seeds):
        next_state_parts.append(sp + cauchy.sample(
          sample_shape=sp.shape, seed=ps))
      return next_state_parts
    return _fn

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  samples = tfp.mcmc.sample_chain(
      num_results=num_chain_results,
      num_burnin_steps=num_burnin_steps,
      current_state=dtype(1),
      kernel=tfp.mcmc.RandomWalkMetropolis(
          target.log_prob,
          new_state_fn=cauchy_new_state_fn(scale=0.5, dtype=dtype)),
      trace_fn=None,
      seed=42)

  sample_mean = tf.math.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
      tf.math.reduce_mean(
          tf.math.squared_difference(samples, sample_mean),
          axis=0))

  print('Estimated mean: {}'.format(sample_mean))
  print('Estimated standard deviation: {}'.format(sample_std))
  ```

  """

  def __init__(self,
               target_log_prob_fn,
               new_state_fn=None,
               experimental_shard_axis_names=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it's a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      new_state_fn: Python callable which takes a list of state parts and a
        seed; returns a same-type `list` of `Tensor`s, each being a perturbation
        of the input state parts. The perturbation distribution is assumed to be
        a symmetric distribution centered at the input state part.
        Default value: `None` which is mapped to
          `tfp.mcmc.random_walk_normal_fn()`.
      experimental_shard_axis_names: A structure of string names indicating how
        members of the state are sharded.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'rwm_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `scale` or a list with same length as
        `current_state`.
    """
    if new_state_fn is None:
      new_state_fn = random_walk_normal_fn()

    self._impl = metropolis_hastings.MetropolisHastings(
        inner_kernel=UncalibratedRandomWalk(
            target_log_prob_fn=target_log_prob_fn,
            new_state_fn=new_state_fn,
            name=name)).experimental_with_shard_axes(
                experimental_shard_axis_names)
    self._parameters = self._impl.inner_kernel.parameters.copy()

  @property
  def target_log_prob_fn(self):
    return self._impl.inner_kernel.target_log_prob_fn

  @property
  def new_state_fn(self):
    return self._impl.inner_kernel.new_state_fn

  @property
  def name(self):
    return self._impl.inner_kernel.name

  @property
  def is_calibrated(self):
    return True

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._impl.inner_kernel.parameters

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Runs one iteration of Random Walk Metropolis with normal proposal.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions index
        independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `scale` or a list with same length as
        `current_state`.
    """
    return self._impl.one_step(current_state, previous_kernel_results,
                               seed=seed)

  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    return self._impl.bootstrap_results(init_state)

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axes):
    return self.copy(experimental_shard_axis_names=shard_axes)


class UncalibratedRandomWalk(kernel_base.TransitionKernel):
  """Generate proposal for the Random Walk Metropolis algorithm.

  Warning: this kernel will not result in a chain which converges to the
  `target_log_prob`. To get a convergent MCMC, use
  `tfp.mcmc.RandomWalkMetropolisNormal(...)` or
  `tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedRandomWalk(...))`.

  For more details on `UncalibratedRandomWalk`, see
  `RandomWalkMetropolis`.
  """

  @mcmc_util.set_doc(RandomWalkMetropolis.__init__.__doc__)
  def __init__(self,
               target_log_prob_fn,
               new_state_fn=None,
               experimental_shard_axis_names=None,
               name=None):
    if new_state_fn is None:
      new_state_fn = random_walk_normal_fn()

    self._target_log_prob_fn = target_log_prob_fn
    self._name = name
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        new_state_fn=new_state_fn,
        experimental_shard_axis_names=experimental_shard_axis_names,
        name=name)

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def new_state_fn(self):
    return self._parameters['new_state_fn']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Return `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return False

  @mcmc_util.set_doc(RandomWalkMetropolis.one_step.__doc__)
  def one_step(self, current_state, previous_kernel_results, seed=None):
    with tf.name_scope(mcmc_util.make_name(self.name, 'rwm', 'one_step')):
      with tf.name_scope('initialize'):
        if mcmc_util.is_list_like(current_state):
          current_state_parts = list(current_state)
        else:
          current_state_parts = [current_state]
        current_state_parts = [
            tf.convert_to_tensor(s, name='current_state')
            for s in current_state_parts
        ]

      seed = samplers.sanitize_seed(seed)  # Retain for diagnostics.
      state_fn_kwargs = {}
      if self.experimental_shard_axis_names is not None:
        state_fn_kwargs['experimental_shard_axis_names'] = (
            self.experimental_shard_axis_names)

      next_state_parts = self.new_state_fn(  # pylint: disable=not-callable
          current_state_parts, seed, **state_fn_kwargs)

      # User should be using a new_state_fn that does not alter the state size.
      # This will fail noisily if that is not the case.
      for next_part, current_part in zip(next_state_parts, current_state_parts):
        tensorshape_util.set_shape(next_part, current_part.shape)

      # Compute `target_log_prob` so its available to MetropolisHastings.
      next_target_log_prob = self.target_log_prob_fn(*next_state_parts)  # pylint: disable=not-callable

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      return [
          maybe_flatten(next_state_parts),
          UncalibratedRandomWalkResults(
              log_acceptance_correction=tf.zeros_like(next_target_log_prob),
              target_log_prob=next_target_log_prob,
              seed=seed,
          ),
      ]

  @mcmc_util.set_doc(RandomWalkMetropolis.bootstrap_results.__doc__)
  def bootstrap_results(self, init_state):
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'rwm', 'bootstrap_results')):
      if not mcmc_util.is_list_like(init_state):
        init_state = [init_state]
      init_state = [tf.convert_to_tensor(x) for x in init_state]
      init_target_log_prob = self.target_log_prob_fn(*init_state)  # pylint:disable=not-callable
      return UncalibratedRandomWalkResults(
          log_acceptance_correction=tf.zeros_like(init_target_log_prob),
          target_log_prob=init_target_log_prob,
          # Allow room for one_step's seed.
          seed=samplers.zeros_seed())

  @property
  def experimental_shard_axis_names(self):
    return self._parameters['experimental_shard_axis_names']

  def experimental_with_shard_axes(self, shard_axis_names):
    return self.copy(experimental_shard_axis_names=shard_axis_names)


def _maybe_call_fn(fn,
                   fn_arg_list,
                   fn_result=None,
                   description='target_log_prob'):
  """Helper which computes `fn_result` if needed."""
  if mcmc_util.is_list_like(fn_arg_list):
    fn_arg_list = list(fn_arg_list)
  else:
    fn_arg_list = [fn_arg_list]

  if fn_result is None:
    fn_result = fn(*fn_arg_list)
  if not dtype_util.is_floating(fn_result.dtype):
    raise TypeError('`{}` must be a `Tensor` with `float` `dtype`.'.format(
        description))
  return fn_result
