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
"""Elliptic Slice sampler transition kernel."""

import collections
# Dependency imports

import numpy as np

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc.internal import util as mcmc_util


__all__ = [
    'EllipticalSliceSampler',
]


def _rotate_on_ellipse(state_parts, vectors, angle):
  new_state_parts = []
  padded_angle = _right_pad_with_ones(angle, tf.rank(state_parts[0]))
  for state, vector in zip(state_parts, vectors):
    new_state_parts.append(
        state * tf.cos(padded_angle) + vector * tf.sin(padded_angle))
  return new_state_parts


def  _right_pad_with_ones(x, target_rank):
  static_target_rank = tf.get_static_value(target_rank)
  if x.shape.is_fully_defined() and static_target_rank is not None:
    return tf.reshape(x, x.shape.concatenate([1] * (
        int(static_target_rank) - x.shape.ndims)))
  return tf.reshape(
      x, tf.concat(
          [tf.shape(x),
           tf.ones(
               [target_rank - tf.rank(x)], dtype=target_rank.dtype)], axis=0))


class EllipticalSliceSamplerKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'EllipticalSliceSamplerKernelResults',
        [
            'log_likelihood',  # For "next_state".
            'angle',  # Angle previous state was rotated by.
            'normal_samples',  # Normal samples used to generate the next state.
            'seed',  # The seed used by `one_step`.
        ])):
  """Internal state and diagnostics for Elliptical slice sampling."""
  __slots__ = ()


class EllipticalSliceSampler(kernel_base.TransitionKernel):
  """Runs one step of the elliptic slice sampler.

  Elliptical Slice Sampling is a Markov Chain Monte Carlo (MCMC) algorithm
  based, as stated in [Murray, 2010][1].

  Given `log_likelihood_fn` and `normal_sampler_fn`, the goal of Elliptical
  Slice Sampling is to sample from:

  ```none
  p(f) = N(f; 0, Sigma)L(f) / Z
  ```

  where:
    * `L = log_likelihood_fn`
    * `Sigma` is a covariance matrix.
    * Samples from `normal_sampler_fn` are distributed as `N(f; 0, Sigma)`.
    * `Z` is a normalizing constant.

  In other words, sampling from a posterior distribution that is proportional
  to a multivariate gaussian prior multiplied by some likelihood function.

  The `one_step` function can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `log_likelihood_fn(*current_state)` should sum log-probabilities across all
  event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`.
  These semantics are governed both by `log_likelihood_fn(*current_state)` and
  `normal_sampler_fn`.

  Note that the sampler only supports states where all components have a common
  dtype.

  ### Examples:

  #### Simple chain with warm-up.

  In this example we have the following model.

  ```none
    p(loc | loc0, scale0) ~ N(loc0, scale0)
    p(x | loc, sigma) ~ N(mu, sigma)
  ```

  What we would like to do is sample from `p(loc | x, loc0, scale0)`. In other
  words, given some data, we would like to infer the posterior distribution
  of the mean that generated that data point.

  We can use elliptical slice sampling here.

  ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    import numpy as np

    tfd = tfp.distributions

    dtype = np.float64

    # loc0 = 0, scale0 = 1
    normal_sampler_fn = lambda seed: return tfd.Normal(
        loc=dtype(0), scale=dtype(1)).sample(seed=seed)

    # We saw the following data.
    data_points = np.random.randn(20)

    # scale = 2.
    log_likelihood_fn = lambda state: return tf.reduce_sum(
        tfd.Normal(state, dtype(2.)).log_prob(data_points))

    kernel = tfp.mcmc.EllipticalSliceSampler(
        normal_sampler_fn=normal_sampler_fn,
        log_likelihood_fn=log_likelihood_fn,
        seed=1234)

    samples = tfp.mcmc.sample_chain(
        num_results=int(3e5),
        current_state=dtype(1),
        kernel=kernel,
        num_burnin_steps=1000,
        trace_fn=None,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_std = tf.sqrt(
      tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                     axis=0))

    with tf.Session() as sess:
      [sample_mean, sample_std] = sess.run([sample_mean, sample_std])

    print("Sample mean: ", sample_mean)
    print("Sample Std: ", sample_std)
  ```

  ### References

  [1]: Ian Murray, Ryan P. Adams, David J.C. MacKay. Elliptical slice sampling.
       proceedings.mlr.press/v9/murray10a/murray10a.pdf
  """

  def __init__(self,
               normal_sampler_fn,
               log_likelihood_fn,
               name=None):
    """Initializes this transition kernel.

    Args:
      normal_sampler_fn: Python callable that takes in a seed and returns a
        sample from a multivariate normal distribution. Note that the shape of
        the samples must agree with `log_likelihood_fn`.
      log_likelihood_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it is a list) and returns its
        (possibly unnormalized) log-likelihood.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'slice_sampler_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """
    self._parameters = dict(
        normal_sampler_fn=normal_sampler_fn,
        log_likelihood_fn=log_likelihood_fn,
        name=name)

  @property
  def normal_sampler_fn(self):
    return self._parameters['normal_sampler_fn']

  @property
  def log_likelihood_fn(self):
    return self._parameters['log_likelihood_fn']

  @property
  def name(self):
    return self._parameters['name']

  @property
  def parameters(self):
    """Returns `dict` of ``__init__`` arguments and their values."""
    return self._parameters

  @property
  def is_calibrated(self):
    return True

  def one_step(self, current_state, previous_kernel_results, seed=None):
    """Runs one iteration of the Elliptical Slice Sampler.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions
        index independent chains,
        `r = tf.rank(log_likelihood_fn(*normal_sampler_fn()))`.
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
      TypeError: if `not log_likelihood.dtype.is_floating`.
    """
    with tf.name_scope(
        mcmc_util.make_name(self.name, 'elliptical_slice', 'one_step')):
      with tf.name_scope('initialize'):
        [
            init_state_parts,
            init_log_likelihood
        ] = _prepare_args(
            self.log_likelihood_fn,
            current_state,
            previous_kernel_results.log_likelihood)

      seed = samplers.sanitize_seed(seed)  # Unsalted, for kernel results.
      normal_seed, u_seed, angle_seed, loop_seed = samplers.split_seed(
          seed, n=4, salt='elliptical_slice_sampler')
      normal_samples = self.normal_sampler_fn(normal_seed)  # pylint: disable=not-callable
      normal_samples = list(normal_samples) if mcmc_util.is_list_like(
          normal_samples) else [normal_samples]
      u = samplers.uniform(
          shape=tf.shape(init_log_likelihood),
          seed=u_seed,
          dtype=init_log_likelihood.dtype.base_dtype,
      )
      threshold = init_log_likelihood + tf.math.log(u)

      starting_angle = samplers.uniform(
          shape=tf.shape(init_log_likelihood),
          minval=0.,
          maxval=2 * np.pi,
          name='angle',
          seed=angle_seed,
          dtype=init_log_likelihood.dtype.base_dtype,
      )
      starting_angle_min = starting_angle - 2 * np.pi
      starting_angle_max = starting_angle

      starting_state_parts = _rotate_on_ellipse(
          init_state_parts, normal_samples, starting_angle)
      starting_log_likelihood = self.log_likelihood_fn(*starting_state_parts)  # pylint: disable=not-callable

      def chain_not_done(
          seed,
          angle,
          angle_min,
          angle_max,
          current_state_parts,
          current_log_likelihood):
        del seed, angle, angle_min, angle_max, current_state_parts
        return tf.reduce_any(current_log_likelihood < threshold)

      def sample_next_angle(
          seed,
          angle,
          angle_min,
          angle_max,
          current_state_parts,
          current_log_likelihood):
        """Slice sample a new angle, and rotate init_state by that amount."""
        angle_seed, next_seed = samplers.split_seed(seed)
        chain_not_done = current_log_likelihood < threshold
        # Box in on angle. Only update angles for which we haven't generated a
        # point that beats the threshold.
        angle_min = tf.where(
            (angle < 0) & chain_not_done,
            angle,
            angle_min)
        angle_max = tf.where(
            (angle >= 0) & chain_not_done,
            angle,
            angle_max)
        new_angle = samplers.uniform(
            shape=tf.shape(current_log_likelihood),
            minval=angle_min,
            maxval=angle_max,
            seed=angle_seed,
            dtype=angle.dtype.base_dtype
        )
        angle = tf.where(chain_not_done, new_angle, angle)
        next_state_parts = _rotate_on_ellipse(
            init_state_parts, normal_samples, angle)

        new_state_parts = []
        broadcasted_chain_not_done = _right_pad_with_ones(
            chain_not_done, tf.rank(next_state_parts[0]))
        for n_state, c_state in zip(next_state_parts, current_state_parts):
          new_state_part = tf.where(
              broadcasted_chain_not_done, n_state, c_state)
          new_state_parts.append(new_state_part)

        return (
            next_seed,
            angle,
            angle_min,
            angle_max,
            new_state_parts,
            self.log_likelihood_fn(*new_state_parts)  # pylint: disable=not-callable
        )

      [
          _,
          next_angle,
          _,
          _,
          next_state_parts,
          next_log_likelihood,
      ] = tf.while_loop(
          cond=chain_not_done,
          body=sample_next_angle,
          loop_vars=[
              loop_seed,
              starting_angle,
              starting_angle_min,
              starting_angle_max,
              starting_state_parts,
              starting_log_likelihood
          ])

      return [
          next_state_parts if mcmc_util.is_list_like(
              current_state) else next_state_parts[0],
          EllipticalSliceSamplerKernelResults(
              log_likelihood=next_log_likelihood,
              angle=next_angle,
              normal_samples=normal_samples,
              seed=seed,
          ),
      ]

  def bootstrap_results(self, init_state):
    with tf.name_scope(mcmc_util.make_name(
        self.name, 'elliptical_slice', 'bootstrap_results')):
      if not mcmc_util.is_list_like(init_state):
        init_state = [init_state]
      init_state = [tf.convert_to_tensor(x) for x in init_state]
      init_log_likelihood = self.log_likelihood_fn(*init_state)  # pylint:disable=not-callable
      return EllipticalSliceSamplerKernelResults(
          log_likelihood=init_log_likelihood,
          angle=tf.zeros_like(init_log_likelihood),
          normal_samples=[tf.zeros_like(x) for x in init_state],
          seed=samplers.zeros_seed(),
      )


def _maybe_call_fn(fn,
                   fn_arg_list,
                   fn_result=None,
                   description='log_likelihood'):
  """Helper which computes `fn_result` if needed."""
  fn_arg_list = (list(fn_arg_list) if mcmc_util.is_list_like(fn_arg_list)
                 else [fn_arg_list])
  if fn_result is None:
    fn_result = fn(*fn_arg_list)
  if not fn_result.dtype.is_floating:
    raise TypeError('`{}` must be a `Tensor` with `float` `dtype`.'.format(
        description))
  return fn_result


def _prepare_args(log_likelihood_fn, state,
                  log_likelihood=None, description='log_likelihood'):
  """Processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [tf.convert_to_tensor(s, name='current_state')
                 for s in state_parts]

  log_likelihood = _maybe_call_fn(
      log_likelihood_fn,
      state_parts,
      log_likelihood,
      description)
  return [state_parts, log_likelihood]
