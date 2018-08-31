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
"""Slice sampler transition kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

import tensorflow as tf

from tensorflow_probability.python import distributions
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import slice_sampler_utils as ssu
from tensorflow_probability.python.mcmc import util as mcmc_util

from tensorflow.python.ops.distributions import util as distributions_util


__all__ = [
    'SliceSampler',
]


SliceSamplerKernelResults = collections.namedtuple(
    'SliceSamplerKernelResults',
    [
        'target_log_prob',  # For "next_state".
        'bounds_satisfied',  # Were the slice bounds chosen outside the slice.
        'direction',  # The direction in which the slice was sampled.
        'upper_bounds',  # Upper bound of the slice in the sampling direction.
        'lower_bounds',  # Lower bound of the slice in the sampling direction.
    ])


class SliceSampler(kernel_base.TransitionKernel):
  """Runs one step of the slice sampler using a hit and run approach.

  Slice Sampling is a Markov Chain Monte Carlo (MCMC) algorithm based, as stated
  by [Neal (2003)][1], on the observation that "...one can sample from a
  distribution by sampling uniformly from the region under the plot of its
  density function. A Markov chain that converges to this uniform distribution
  can be constructed by alternately uniform sampling in the vertical direction
  with uniform sampling from the horizontal `slice` defined by the current
  vertical position, or more generally, with some update that leaves the uniform
  distribution over this slice invariant". Mathematical details and derivations
  can be found in [Neal (2003)][1]. The one dimensional slice sampler is
  extended to n-dimensions through use of a hit-and-run approach: choose a
  random direction in n-dimensional space and take a step, as determined by the
  one-dimensional slice sampling algorithm, along that direction
  [Belisle at al. 1993][2].

  The `one_step` function can update multiple chains in parallel. It assumes
  that all leftmost dimensions of `current_state` index independent chain states
  (and are therefore updated independently). The output of
  `target_log_prob_fn(*current_state)` should sum log-probabilities across all
  event dimensions. Slices along the rightmost dimensions may have different
  target distributions; for example, `current_state[0, :]` could have a
  different target distribution from `current_state[1, :]`. These semantics are
  governed by `target_log_prob_fn(*current_state)`. (The number of independent
  chains is `tf.size(target_log_prob_fn(*current_state))`.)

  Note that the sampler only supports states where all components have a common
  dtype.

  ### Examples:

  #### Simple chain with warm-up.

  In this example we sample from a standard univariate normal
  distribution using slice sampling.

  ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    import numpy as np

    tfd = tfp.distributions

    dtype = np.float32

    target = tfd.Normal(loc=dtype(0), scale=dtype(1))

    samples, _ = tfp.mcmc.sample_chain(
        num_results=1000,
        current_state=dtype(1),
        kernel=tfp.mcmc.SliceSampler(
            target.log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=1234),
        num_burnin_steps=500,
        parallel_iterations=1)  # For determinism.

    sample_mean = tf.reduce_mean(samples, axis=0)
    sample_std = tf.sqrt(
      tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                     axis=0))

    with tf.Session() as sess:
      [sample_mean, sample_std] = sess.run([sample_mean, sample_std])

    print "Sample mean: ", sample_mean
    print "Sample Std: ", sample_std
  ```

  #### Sample from a Two Dimensional Normal.

  In the following example we sample from a two dimensional Normal
  distribution using slice sampling.

  ```python
    import tensorflow as tf
    import tensorflow_probability as tfp
    import numpy as np

    tfd = tfp.distributions

    dtype = np.float32
    true_mean = dtype([0, 0])
    true_cov = dtype([[1, 0.5], [0.5, 1]])
    num_results = 500
    num_chains = 50

    # Target distribution is defined through the Cholesky decomposition
    chol = tf.linalg.cholesky(true_cov)
    target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

    # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
    # Then the target log-density is defined as follows:
    def target_log_prob(x, y):
      # Stack the input tensors together
      z = tf.stack([x, y], axis=-1) - true_mean
      return target.log_prob(z)

    # Initial state of the chain
    init_state = [np.ones([num_chains, 1], dtype=dtype),
                  np.ones([num_chains, 1], dtype=dtype)]

    # Run Slice Samper for `num_results` iterations for `num_chains`
    # independent chains:
    [x, y], _ = tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=init_state,
        kernel=tfp.mcmc.SliceSampler(
            target_log_prob_fn=target_log_prob,
            step_size=1.0,
            max_doublings=5,
            seed=47),
        num_burnin_steps=200,
        num_steps_between_results=1,
        parallel_iterations=1)

    states = tf.stack([x, y], axis=-1)
    sample_mean = tf.reduce_mean(states, axis=[0, 1])
    z = states - sample_mean
    sample_cov = tf.reduce_mean(tf.matmul(z, z, transpose_a=True),
                                axis=[0, 1])

    with tf.Session() as sess:
      [sample_mean, sample_cov] = sess.run([
          sample_mean, sample_cov])

    print "sample_mean: ", sample_mean
    print "sample_cov: ", sample_cov
  ```

  ### References

  [1]: Radford M. Neal. Slice Sampling. The Annals of Statistics. 2003, Vol 31,
       No. 3 , 705-767.
       https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461

  [2]: C.J.P. Belisle, H.E. Romeijn, R.L. Smith. Hit-and-run algorithms for
       generating multivariate distributions. Math. Oper. Res., 18(1993),
       225-266.
       https://www.jstor.org/stable/3690278?seq=1#page_scan_tab_contents
  """

  def __init__(self,
               target_log_prob_fn,
               step_size,
               max_doublings,
               seed=None,
               name=None):
    """Initializes this transition kernel.

    Args:
      target_log_prob_fn: Python callable which takes an argument like
        `current_state` (or `*current_state` if it is a list) and returns its
        (possibly unnormalized) log-density under the target distribution.
      step_size: Scalar or `tf.Tensor` with same dtype as and shape compatible
        with `x_initial`. The size of the initial interval.
      max_doublings: Scalar positive int32 `tf.Tensor`. The maximum number of
      doublings to consider.
      seed: Python integer to seed the random number generator.
      name: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., 'slice_sampler_kernel').

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) at each result step. Has same shape as
        `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.
    """
    self._seed_stream = distributions.SeedStream(
        seed, salt='slice_sampler')
    self._parameters = dict(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        max_doublings=max_doublings,
        seed=seed,
        name=name)

  @property
  def target_log_prob_fn(self):
    return self._parameters['target_log_prob_fn']

  @property
  def step_size(self):
    return self._parameters['step_size']

  @property
  def max_doublings(self):
    return self._parameters['max_doublings']

  @property
  def seed(self):
    return self._parameters['seed']

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

  def one_step(self, current_state, previous_kernel_results):
    """Runs one iteration of Slice Sampler.

    Args:
      current_state: `Tensor` or Python `list` of `Tensor`s representing the
        current state(s) of the Markov chain(s). The first `r` dimensions
        index independent chains,
        `r = tf.rank(target_log_prob_fn(*current_state))`.
      previous_kernel_results: `collections.namedtuple` containing `Tensor`s
        representing values from previous calls to this function (or from the
        `bootstrap_results` function.)

    Returns:
      next_state: Tensor or Python list of `Tensor`s representing the state(s)
        of the Markov chain(s) after taking exactly one step. Has same type and
        shape as `current_state`.
      kernel_results: `collections.namedtuple` of internal calculations used to
        advance the chain.

    Raises:
      ValueError: if there isn't one `step_size` or a list with same length as
        `current_state`.
      TypeError: if `not target_log_prob.dtype.is_floating`.
    """
    with tf.name_scope(
        name=mcmc_util.make_name(self.name, 'slice', 'one_step'),
        values=[self.step_size, self.max_doublings, self._seed_stream,
                current_state,
                previous_kernel_results.target_log_prob]):
      with tf.name_scope('initialize'):
        [
            current_state_parts,
            step_sizes,
            current_target_log_prob
        ] = _prepare_args(
            self.target_log_prob_fn,
            current_state,
            self.step_size,
            previous_kernel_results.target_log_prob,
            maybe_expand=True)

        max_doublings = tf.convert_to_tensor(
            self.max_doublings,
            dtype=tf.int32,
            name='max_doublings')

      independent_chain_ndims = distributions_util.prefer_static_rank(
          current_target_log_prob)

      [
          next_state_parts,
          next_target_log_prob,
          bounds_satisfied,
          direction,
          upper_bounds,
          lower_bounds
      ] = _sample_next(
          self.target_log_prob_fn,
          current_state_parts,
          step_sizes,
          max_doublings,
          current_target_log_prob,
          independent_chain_ndims,
          seed=self._seed_stream()
      )

      def maybe_flatten(x):
        return x if mcmc_util.is_list_like(current_state) else x[0]

      return [
          maybe_flatten(next_state_parts),
          SliceSamplerKernelResults(
              target_log_prob=next_target_log_prob,
              bounds_satisfied=bounds_satisfied,
              direction=direction,
              upper_bounds=upper_bounds,
              lower_bounds=lower_bounds
          ),
      ]

  def bootstrap_results(self, init_state):
    with tf.name_scope(
        name=mcmc_util.make_name(self.name, 'slice', 'bootstrap_results'),
        values=[init_state]):
      if not mcmc_util.is_list_like(init_state):
        init_state = [init_state]
      init_state = [tf.convert_to_tensor(x) for x in init_state]
      direction = [tf.zeros_like(x) for x in init_state]
      init_target_log_prob = self.target_log_prob_fn(*init_state)  # pylint:disable=not-callable
      return SliceSamplerKernelResults(
          target_log_prob=init_target_log_prob,
          bounds_satisfied=tf.zeros(shape=tf.shape(init_target_log_prob),
                                    dtype=tf.bool),
          direction=direction,
          upper_bounds=tf.zeros_like(init_target_log_prob),
          lower_bounds=tf.zeros_like(init_target_log_prob)
      )


def _choose_random_direction(current_state_parts, batch_rank, seed=None):
  """Chooses a random direction in the event space."""
  seed_gen = distributions.SeedStream(seed, salt='_choose_random_direction')
  # Chooses the random directions across each of the input components.
  rnd_direction_parts = [tf.random_normal(tf.shape(current_state_part),
                                          dtype=tf.float32, seed=seed_gen())
                         for current_state_part in current_state_parts]

  # Sum squares over all of the input components. Note this takes all
  # components into account.
  sum_squares = sum(
      tf.reduce_sum(rnd_direction**2.,
                    axis=tf.range(batch_rank, tf.rank(rnd_direction)),
                    keepdims=True)
      for rnd_direction
      in rnd_direction_parts)

  # Normalizes the random direction fragments.
  rnd_direction_parts = [rnd_direction / tf.sqrt(sum_squares)
                         for rnd_direction in rnd_direction_parts]

  return rnd_direction_parts


def _sample_next(target_log_prob_fn,
                 current_state_parts,
                 step_sizes,
                 max_doublings,
                 current_target_log_prob,
                 batch_rank,
                 seed=None,
                 name=None):
  """Applies a single iteration of slice sampling update.

  Applies hit and run style slice sampling. Chooses a uniform random direction
  on the unit sphere in the event space. Applies the one dimensional slice
  sampling update along that direction.

  Args:
    target_log_prob_fn: Python callable which takes an argument like
      `*current_state_parts` and returns its (possibly unnormalized) log-density
      under the target distribution.
    current_state_parts: Python `list` of `Tensor`s representing the current
      state(s) of the Markov chain(s). The first `independent_chain_ndims` of
      the `Tensor`(s) index different chains.
    step_sizes: Python `list` of `Tensor`s. Provides a measure of the width
      of the density. Used to find the slice bounds. Must broadcast with the
      shape of `current_state_parts`.
    max_doublings: Integer number of doublings to allow while locating the slice
      boundaries.
    current_target_log_prob: `Tensor` representing the value of
      `target_log_prob_fn(*current_state_parts)`. The only reason to specify
      this argument is to reduce TF graph size.
    batch_rank: Integer. The number of axes in the state that correspond to
      independent batches.
    seed: Python integer to seed random number generators.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., 'find_slice_bounds').

  Returns:
    proposed_state_parts: Tensor or Python list of `Tensor`s representing the
      state(s) of the Markov chain(s) at each result step. Has same shape as
      input `current_state_parts`.
    proposed_target_log_prob: `Tensor` representing the value of
      `target_log_prob_fn` at `next_state`.
    bounds_satisfied: Boolean `Tensor` of the same shape as the log density.
      True indicates whether the an interval containing the slice for that
      batch was found successfully.
    direction: `Tensor` or Python list of `Tensors`s representing the direction
      along which the slice was sampled. Has the same shape and dtype(s) as
      `current_state_parts`.
    upper_bounds: `Tensor` of batch shape and the dtype of the input state. The
      upper bounds of the slices along the sampling direction.
    lower_bounds: `Tensor` of batch shape and the dtype of the input state. The
      lower bounds of the slices along the sampling direction.
  """
  with tf.name_scope(
      name, 'sample_next',
      [current_state_parts, step_sizes, max_doublings, current_target_log_prob,
       batch_rank]):
    # First step: Choose a random direction.
    # Direction is a list of tensors. The i'th tensor should have the same shape
    # as the i'th state part.
    direction = _choose_random_direction(current_state_parts,
                                         batch_rank=batch_rank,
                                         seed=seed)

    # Interpolates the step sizes for the chosen direction.
    # Applies an ellipsoidal interpolation to compute the step direction for
    # the chosen direction. Suppose we are given step sizes for each direction.
    # Label these s_1, s_2, ... s_k. These are the step sizes to use if moving
    # in a direction parallel to one of the axes. Consider an ellipsoid which
    # intercepts the i'th axis at s_i. The step size for a direction specified
    # by the unit vector (n_1, n_2 ...n_k) is then defined as the intersection
    # of the line through this vector with this ellipsoid.
    #
    # One can show that the length of the vector from the origin to the
    # intersection point is given by:
    # 1 / sqrt(n_1^2 / s_1^2  + n_2^2 / s_2^2  + ...).
    #
    # Proof:
    # The equation of the ellipsoid is:
    # Sum_i [x_i^2 / s_i^2 ] = 1. Let n be a unit direction vector. Points
    # along the line given by n may be parameterized as alpha*n where alpha is
    # the distance along the vector. Plugging this into the equation for the
    # ellipsoid, we get:
    # alpha^2 ( n_1^2 / s_1^2 + n_2^2 / s_2^2 + ...) = 1
    # so alpha = \sqrt { \frac{1} { ( n_1^2 / s_1^2 + n_2^2 / s_2^2 + ...) } }
    reduce_axes = [tf.range(batch_rank, tf.rank(dirn_part))
                   for dirn_part in direction]

    components = [tf.reduce_sum((dirn_part / step_size) ** 2,
                                axis=reduce_axes[i])
                  for i, (step_size, dirn_part)
                  in enumerate(zip(step_sizes, direction))]
    step_size = tf.rsqrt(tf.add_n(components))
    # Computes the rank of a tensor. Uses the static rank if possible.
    def _get_rank(x):
      return (len(x.shape.as_list()) if x.shape.dims is not None
              else tf.rank(x))
    state_part_ranks = [_get_rank(part) for part in current_state_parts]

    def _step_along_direction(alpha):
      """Converts the scalar alpha into an n-dim vector with full state info.

      Computes x_0 + alpha * direction where x_0 is the current state and
      direction is the direction chosen above.

      Args:
        alpha: A tensor of shape equal to the batch dimensions of
          `current_state_parts`.

      Returns:
        state_parts: Tensor or Python list of `Tensor`s representing the
          state(s) of the Markov chain(s) for a given alpha and a given chosen
          direction. Has the same shape as `current_state_parts`.
      """
      padded_alphas = [_right_pad(alpha, final_rank=part_rank)
                       for part_rank in state_part_ranks]

      state_parts = [state_part + padded_alpha
                     * direction_part for state_part, direction_part,
                     padded_alpha in
                     zip(current_state_parts, direction, padded_alphas)]
      return state_parts

    def projected_target_log_prob_fn(alpha):
      """The target log density projected along the chosen direction.

      Args:
        alpha: A tensor of shape equal to the batch dimensions of
          `current_state_parts`.

      Returns:
        Target log density evaluated at x_0 + alpha * direction where x_0 is the
        current state and direction is the direction chosen above. Has the same
        shape as `alpha`.
      """
      return target_log_prob_fn(*_step_along_direction(alpha))

    alpha_init = tf.zeros_like(current_target_log_prob,
                               dtype=current_state_parts[0].dtype.base_dtype)
    [
        next_alpha,
        next_target_log_prob,
        bounds_satisfied,
        upper_bounds,
        lower_bounds
    ] = ssu.slice_sampler_one_dim(projected_target_log_prob_fn,
                                  x_initial=alpha_init,
                                  max_doublings=max_doublings,
                                  step_size=step_size, seed=seed)
    return [
        _step_along_direction(next_alpha),
        next_target_log_prob,
        bounds_satisfied,
        direction,
        upper_bounds,
        lower_bounds
    ]


def _maybe_call_fn(fn,
                   fn_arg_list,
                   fn_result=None,
                   description='target_log_prob'):
  """Helper which computes `fn_result` if needed."""
  fn_arg_list = (list(fn_arg_list) if mcmc_util.is_list_like(fn_arg_list)
                 else [fn_arg_list])
  if fn_result is None:
    fn_result = fn(*fn_arg_list)
  if not fn_result.dtype.is_floating:
    raise TypeError('`{}` must be a `Tensor` with `float` `dtype`.'.format(
        description))
  return fn_result


def _right_pad(x, final_rank):
  """Pads the shape of x to the right to be of rank final_rank.

  Expands the dims of `x` to the right such that its rank is equal to
  final_rank. For example, if `x` is of shape [1, 5, 7, 2] and `final_rank` is
  7, we return padded_x, which is of shape [1, 5, 7, 2, 1, 1, 1].

  Args:
    x: The tensor whose shape is to be padded.
    final_rank: Scalar int32 `Tensor` or Python `int`. The desired rank of x.

  Returns:
    padded_x: A tensor of rank final_rank.
  """
  padded_shape = tf.concat(
      [
          tf.shape(x),
          tf.ones(final_rank - tf.rank(x), dtype=tf.int32)
      ],
      axis=0)
  static_padded_shape = None
  if x.shape.is_fully_defined() and isinstance(final_rank, int):
    static_padded_shape = x.shape.as_list()
    extra_dims = final_rank - len(static_padded_shape)
    static_padded_shape.extend([1] * extra_dims)

  padded_x = tf.reshape(x, static_padded_shape or padded_shape)
  return padded_x


def _prepare_args(target_log_prob_fn, state, step_size,
                  target_log_prob=None, maybe_expand=False,
                  description='target_log_prob'):
  """Processes input args to meet list-like assumptions."""
  state_parts = list(state) if mcmc_util.is_list_like(state) else [state]
  state_parts = [tf.convert_to_tensor(s, name='current_state')
                 for s in state_parts]

  target_log_prob = _maybe_call_fn(
      target_log_prob_fn,
      state_parts,
      target_log_prob,
      description)
  step_sizes = (list(step_size) if mcmc_util.is_list_like(step_size)
                else [step_size])
  step_sizes = [
      tf.convert_to_tensor(
          s, name='step_size', dtype=target_log_prob.dtype)
      for s in step_sizes]
  if len(step_sizes) == 1:
    step_sizes *= len(state_parts)
  if len(state_parts) != len(step_sizes):
    raise ValueError('There should be exactly one `step_size` or it should '
                     'have same length as `current_state`.')
  def maybe_flatten(x):
    return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
  return [
      maybe_flatten(state_parts),
      maybe_flatten(step_sizes),
      target_log_prob
  ]
