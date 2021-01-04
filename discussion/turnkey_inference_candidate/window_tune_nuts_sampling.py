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
"""MCMC sampling with HMC/NUTS using an expanding epoch tuning scheme."""

from absl import logging

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.transformed_kernel import make_transform_fn
from tensorflow_probability.python.mcmc.transformed_kernel import make_transformed_log_prob

tfb = tfp.bijectors

__all__ = [
    'window_tune_nuts_sampling',
]


def _sample_posterior(target_log_prob_unconstrained,
                      prior_samples_unconstrained,
                      init_state=None,
                      num_samples=500,
                      nchains=4,
                      init_nchains=1,
                      target_accept_prob=.8,
                      max_tree_depth=9,
                      use_scaled_init=True,
                      tuning_window_schedule=(75, 25, 25, 25, 25, 25, 50),
                      use_wide_window_expanding_mode=True,
                      seed=None,
                      parallel_iterations=10,
                      jit_compile=True,
                      use_input_signature=False,
                      experimental_relax_shapes=False):
  """MCMC sampling with HMC/NUTS using an expanding epoch tuning scheme."""

  seed_stream = tfp.util.SeedStream(seed, 'window_tune_nuts_sampling')
  rv_rank = ps.rank(prior_samples_unconstrained)
  assert rv_rank == 2
  total_ndims = ps.shape(prior_samples_unconstrained)[-1]
  dtype = prior_samples_unconstrained.dtype

  # TODO(b/158878248): explore option to for user to control the
  # parameterization of conditioning_bijector.
  # TODO(b/158878248): right now, we use 2 tf.Variable to initialize a scaling
  # bijector, and update the underlying value at the end of each warmup window.
  # It might be faster to rewrite it into a functional style (with a small
  # additional compilation cost).
  loc_conditioner = tf.Variable(
      tf.zeros([total_ndims], dtype=dtype), name='loc_conditioner')
  scale_conditioner = tf.Variable(
      tf.ones([total_ndims], dtype=dtype), name='scale_conditioner')

  # Start with Identity Covariance matrix
  scale = tf.linalg.LinearOperatorDiag(
      diag=scale_conditioner,
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True)
  conditioning_bijector = tfb.Shift(shift=loc_conditioner)(
      tfb.ScaleMatvecLinearOperator(scale))

  if init_state is None:
    # Start at uniform random [-1, 1] around the prior mean in latent space
    init_state_uniform = tf.random.uniform(
        [init_nchains, total_ndims], dtype=dtype, seed=seed_stream()) * 2. - 1.
    if use_scaled_init:
      prior_z_mean = tf.math.reduce_mean(prior_samples_unconstrained, axis=0)
      prior_z_std = tf.math.reduce_std(prior_samples_unconstrained, axis=0)
      init_state = init_state_uniform * prior_z_std + prior_z_mean
    else:
      init_state = init_state_uniform

  # The denominator is the O(N^0.25) scaling from Beskos et al. 2010. The
  # numerator corresponds to the trajectory length. Candidate value includs: 1,
  # 1.57 (pi / 2). We use a conservately small value here (0.25).
  init_step_size = tf.constant(0.25 / (total_ndims**0.25), dtype=dtype)

  hmc_inner = tfp.mcmc.TransformedTransitionKernel(
      tfp.mcmc.NoUTurnSampler(
          target_log_prob_fn=target_log_prob_unconstrained,
          step_size=init_step_size,
          max_tree_depth=max_tree_depth,
          parallel_iterations=parallel_iterations,
      ), conditioning_bijector)

  hmc_step_size_tuning = tfp.mcmc.DualAveragingStepSizeAdaptation(
      inner_kernel=hmc_inner,
      num_adaptation_steps=max(tuning_window_schedule),
      target_accept_prob=target_accept_prob)

  if use_input_signature:
    input_signature = [
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[None, total_ndims], dtype=dtype),
    ]
  else:
    input_signature = None

  # TODO(b/158878248): move the nested function definitions to module top-level.
  @tf.function(
      input_signature=input_signature,
      autograph=False,
      jit_compile=jit_compile,
      experimental_relax_shapes=experimental_relax_shapes)
  def fast_adaptation_interval(num_steps, previous_state):
    """Step size only adaptation interval.

    This corresponds to window 1 and window 3 in the Stan HMC parameter
    tuning scheme.

    Args:
      num_steps: Number of tuning steps the interval will run.
      previous_state: Initial state of the tuning interval.

    Returns:
      last_state: Last state of the tuning interval.
      last_pkr: Kernel result from the TransitionKernel at the end of the
        tuning interval.
    """

    def body_fn(i, state, pkr):
      next_state, next_pkr = hmc_step_size_tuning.one_step(state, pkr)
      return i + 1, next_state, next_pkr

    current_pkr = hmc_step_size_tuning.bootstrap_results(previous_state)
    _, last_state, last_pkr = tf.while_loop(
        lambda i, *_: i < num_steps,
        body_fn,
        loop_vars=(0, previous_state, current_pkr),
        maximum_iterations=num_steps,
        parallel_iterations=parallel_iterations)
    return last_state, last_pkr

  def body_fn_window2(
      i, previous_state, previous_pkr, previous_mean, previous_cov):
    """Take one MCMC step and update the step size and mass matrix."""
    next_state, next_pkr = hmc_step_size_tuning.one_step(
        previous_state, previous_pkr)
    n_next = i + 1
    delta_pre = previous_state - previous_mean
    next_mean = previous_mean + delta_pre / tf.cast(n_next, delta_pre.dtype)
    delta_post = previous_state - next_mean
    delta_cov = tf.expand_dims(delta_post, -1) * tf.expand_dims(delta_pre, -2)
    next_cov = previous_cov + delta_cov

    next_mean.set_shape(previous_mean.shape)
    next_cov.set_shape(previous_cov.shape)
    return n_next, next_state, next_pkr, next_mean, next_cov

  if use_input_signature:
    input_signature = [
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[None, total_ndims], dtype=dtype),
        tf.TensorSpec(shape=[None, total_ndims], dtype=dtype),
        tf.TensorSpec(shape=[None, total_ndims, total_ndims], dtype=dtype),
    ]
  else:
    input_signature = None

  # TODO(b/158878248): move the nested function definitions to module top-level.
  @tf.function(
      input_signature=input_signature,
      autograph=False,
      jit_compile=jit_compile,
      experimental_relax_shapes=experimental_relax_shapes)
  def slow_adaptation_interval(num_steps, previous_n, previous_state,
                               previous_mean, previous_cov):
    """Interval that tunes the mass matrix and step size simultaneously.

    This corresponds to window 2 in the Stan HMC parameter tuning scheme.

    Args:
      num_steps: Number of tuning steps the interval will run.
      previous_n: Previous number of tuning steps we have run.
      previous_state: Initial state of the tuning interval.
      previous_mean: Current estimated posterior mean.
      previous_cov: Current estimated posterior covariance matrix.

    Returns:
      total_n: Total number of tuning steps we have run.
      next_state: Last state of the tuning interval.
      next_pkr: Kernel result from the TransitionKernel at the end of the
        tuning interval.
      next_mean: estimated posterior mean after tuning.
      next_cov: estimated posterior covariance matrix after tuning.
    """
    previous_pkr = hmc_step_size_tuning.bootstrap_results(previous_state)
    total_n, next_state, next_pkr, next_mean, next_cov = tf.while_loop(
        lambda i, *_: i < num_steps + previous_n,
        body_fn_window2,
        loop_vars=(previous_n, previous_state, previous_pkr, previous_mean,
                   previous_cov),
        maximum_iterations=num_steps,
        parallel_iterations=parallel_iterations)
    float_n = tf.cast(total_n, next_cov.dtype)
    cov = next_cov / (float_n - 1.)

    # Regularization
    scaled_cov = (float_n / (float_n + 5.)) * cov
    shrinkage = 1e-3 * (5. / (float_n + 5.))
    next_cov = scaled_cov + shrinkage

    return total_n, next_state, next_pkr, next_mean, next_cov

  def trace_fn(_, pkr):
    return (
        pkr.inner_results.target_log_prob,
        pkr.inner_results.leapfrogs_taken,
        pkr.inner_results.has_divergence,
        pkr.inner_results.energy,
        pkr.inner_results.log_accept_ratio,
        pkr.inner_results.reach_max_depth,
        pkr.inner_results.step_size,
    )

  @tf.function(autograph=False, jit_compile=jit_compile)
  def run_chain(num_results, current_state, previous_kernel_results):
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0,
        current_state=current_state,
        previous_kernel_results=previous_kernel_results,
        kernel=hmc_inner,
        trace_fn=trace_fn,
        parallel_iterations=parallel_iterations,
        seed=seed_stream())

  # Main sampling with tuning routine.
  num_steps_tuning_window_schedule0 = tuning_window_schedule[0]

  # Window 1 to tune step size
  logging.info('Tuning Window 1...')
  next_state, _ = fast_adaptation_interval(num_steps_tuning_window_schedule0,
                                           init_state)

  next_mean = tf.zeros_like(init_state)
  next_cov = tf.zeros(
      ps.concat([ps.shape(init_state), ps.shape(init_state)[-1:]], axis=-1),
      dtype=dtype)

  mean_updater = tf.zeros([total_ndims], dtype=dtype)
  diag_updater = tf.ones([total_ndims], dtype=dtype)

  # Window 2 to tune mass matrix.
  total_n = 0
  for i, num_steps in enumerate(tuning_window_schedule[1:-1]):
    logging.info('Tuning Window 2 - %s...', i)
    if not use_wide_window_expanding_mode:
      num_steps = num_steps * 2**i
    with tf.control_dependencies([
        loc_conditioner.assign(mean_updater, read_value=False),
        scale_conditioner.assign(diag_updater, read_value=False)
    ]):
      (total_n, next_state_, _, next_mean_,
       next_cov_) = slow_adaptation_interval(num_steps, total_n, next_state,
                                             next_mean, next_cov)
      diag_part = tf.linalg.diag_part(next_cov_)
      if ps.rank(next_state) > 1:
        mean_updater = tf.reduce_mean(next_mean_, axis=0)
        diag_updater = tf.math.sqrt(tf.reduce_mean(diag_part, axis=0))
      else:
        mean_updater = next_mean_
        diag_updater = tf.math.sqrt(diag_part)

      if use_wide_window_expanding_mode:
        next_mean = tf.concat([next_mean_, next_mean_], axis=0)
        next_cov = tf.concat([next_cov_, next_cov_], axis=0)
        next_state = tf.concat([next_state_, next_state_], axis=0)
      else:
        next_mean, next_cov, next_state = next_mean_, next_cov_, next_state_

  num_steps_tuning_window_schedule3 = tuning_window_schedule[-1]
  num_batches = ps.size0(next_state)
  if nchains > num_batches:
    final_init_state = tf.repeat(
        next_state, (nchains + 1) // num_batches, axis=0)[:nchains]
  else:
    final_init_state = next_state[:nchains]

  with tf.control_dependencies([
      loc_conditioner.assign(mean_updater, read_value=False),
      scale_conditioner.assign(diag_updater, read_value=False)
  ]):
    # Window 3 step size tuning
    logging.info('Tuning Window 3...')
    final_tuned_state, final_pkr = fast_adaptation_interval(
        num_steps_tuning_window_schedule3, final_init_state)

    # Final samples
    logging.info('Sampling...')
    nuts_samples, diagnostic = run_chain(num_samples, final_tuned_state,
                                         final_pkr.inner_results)

  return nuts_samples, diagnostic, conditioning_bijector


def window_tune_nuts_sampling(target_log_prob,
                              prior_samples,
                              constraining_bijectors=None,
                              init_state=None,
                              num_samples=500,
                              nchains=4,
                              init_nchains=1,
                              target_accept_prob=.8,
                              max_tree_depth=9,
                              use_scaled_init=True,
                              tuning_window_schedule=(
                                  75, 25, 25, 25, 25, 25, 50),
                              use_wide_window_expanding_mode=True,
                              seed=None,
                              parallel_iterations=10,
                              jit_compile=True,
                              use_input_signature=True,
                              experimental_relax_shapes=False):
  """Sample from a density with NUTS and an expanding window tuning scheme.

  This function implements a turnkey MCMC sampling routine using NUTS and an
  expanding window tuning strategy similar to Stan [1]. It learns a pre-
  conditioner that scales and rotates the target distribution using a series of
  expanding windows - either in number of samples (same as in Stan,
  use_wide_window_expanding_mode=False) or in number of batches/chains
  (use_wide_window_expanding_mode=True).

  Currently, the function uses `prior_samples` to initialize MCMC chains
  uniformly at random between -1 and 1 scaled by the prior standard deviation
  (i.e., [-prior_std, prior_std]). The scaling is ignored if `use_scaled_init`
  is set to False. Alternatively, user can input the `init_state` directly.

  Currently, the tuning and sampling routine is run in Python, with each block
  of the tuning epoch (window 1, 2, and 3 in Stan [1]) run with two @tf.function
  compiled functions. The user can control the compilation options using the
  kwargs `jit_compile`, `use_input_signature`, and
  `experimental_relax_shapes`.  Setting all to True would compile to XLA and
  potentially avoid the small overhead of function recompilation (note that it
  is not yet the case in XLA right now). It is not yet clear whether doing it
  this way is better than just wrapping the full inference routine in
  tf.function with XLA.

  Internally, this calls `_sample_posterior`, which assumes a real-valued target
  density function and takes a Tensor with shape=(batch * dimension) as input.
  The tuning routine is a memory-less (i.e., no warm-up samples are saved) MCMC
  sampling with number of samples specified by a list-like
  `tuning_window_schedule`.

  Args:
    target_log_prob: Python callable which takes an argument like
      `current_state` (or `*current_state` if it's a list) and returns its
      (possibly unnormalized) log-density under the target distribution.
    prior_samples: Nested structure of `Tensor`s, each of shape `[batches,
      latent_part_event_shape]` and should be sample from the prior. They are
      used to generate an inital chain position if `init_state` is not supplied.
    constraining_bijectors: `tfp.distributions.Bijector` or list of
      `tfp.distributions.Bijector`s. These bijectors use `forward` to map the
      state on the real space to the constrained state expected by
      `target_log_prob`.
    init_state: (Optional) `Tensor` or Python `list` of `Tensor`s representing
      the inital state(s) of the Markov chain(s).
    num_samples: Integer number of the Markov chain draws after tuning.
    nchains: Integer number of the Markov chains after tuning.
    init_nchains: Integer number of the Markov chains in the first phase of
      tuning.
    target_accept_prob: Floating point scalar `Tensor`. Target acceptance
      probability for step size adaptation.
    max_tree_depth: Maximum depth of the tree implicitly built by NUTS. See
      `tfp.mcmc.NoUTurnSampler` for more details
    use_scaled_init: Boolean. If `True`, generate inital state within [-1, 1]
      scaled by prior sample standard deviation in the unconstrained real space.
      This kwarg is ignored if `init_state` is not None
    tuning_window_schedule: List-like sequence of integers that specify the
      tuning schedule. Each integer number specifies the number of MCMC samples
      within a single warm-up window. The first and the last window tunes the
      step size (a scalar) only, while the intermediate windows tune both step
      size and the pre-conditioner. Moreover, the intermediate windows double
      the number of samples taken: for example, the default schedule (75,
        25, 25, 25, 25, 25, 50) actually means it will take (75, 25 * 2**0, 25 *
        2**1, 25 * 2**2, 25 * 2**3, 25 * 2**4, 50) samples.
    use_wide_window_expanding_mode: Boolean. Default to `True` that we double
      the number of chains from the previous stage for the intermediate windows.
      See `tuning_window_schedule` kwarg for more details.
    seed: Python integer to seed the random number generator.
    parallel_iterations: The number of iterations allowed to run in parallel.
      It must be a positive integer. See `tf.while_loop` for more details.
      Note that if you set the seed to have deterministic output you should
      also set `parallel_iterations` to 1.
    jit_compile: kwarg pass to tf.function decorator. If True, the
      function is always compiled by XLA.
    use_input_signature: If True, generate an input_signature kwarg to pass to
      tf.function decorator.
    experimental_relax_shapes: kwarg pass to tf.function decorator. When True,
      tf.function may generate fewer, graphs that are less specialized on input
      shapes.

  Returns:
    posterior_samples: A `Tensor` or Python list of `Tensor`s representing the
      posterior MCMC samples after tuning. It has the same structure as
      `prior_samples` but with the leading shape being (num_samples * nchains)
    diagnostic: A list of `Tensor` representing the diagnostics from NUTS:
      `target_log_prob`, `leapfrogs_taken`, `has_divergence`, `energy`,
      `log_accept_ratio`, `reach_max_depth`, `step_size.
    conditioning_bijector: A tfp bijector that scales and rotates the target
      density function in latent unconstrained space as determined by
      adaptation.

  ### Examples

  Sampling from a multivariate Student-T distribution.

  ```python
  DTYPE = np.float32

  nd = 50
  concentration = 1.

  prior_dist = tfd.Sample(tfd.Normal(tf.constant(0., DTYPE), 100.), nd)

  mu = tf.cast(np.linspace(-100, 100, nd), dtype=DTYPE)
  sigma = tf.cast(np.exp(np.linspace(-1, 1.5, nd)), dtype=DTYPE)
  corr_tril = tfd.CholeskyLKJ(
      dimension=nd, concentration=concentration).sample()
  scale_tril = tf.linalg.matmul(tf.linalg.diag(sigma), corr_tril)
  target_dist = tfd.MultivariateStudentTLinearOperator(
      df=5., loc=mu, scale=tf.linalg.LinearOperatorLowerTriangular(scale_tril))

  target_log_prob = lambda *x: (
      prior_dist.log_prob(*x) + target_dist.log_prob(*x))

  (
      [mcmc_samples], diagnostic, conditioning_bijector
  ) = window_tune_nuts_sampling(target_log_prob, [prior_dist.sample(2000)])

  loc_conditioner, scale_conditioner = conditioning_bijector.trainable_variables

  _, ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].plot(mu, loc_conditioner.numpy(), 'o', label='conditioner mean')
  ax[0].plot(mu, tf.reduce_mean(
      mcmc_samples, axis=[0, 1]), 'o', label='estimated mean')
  ax[0].legend()

  sigma_sim = target_dist._stddev()
  ax[1].plot(sigma_sim, scale_conditioner.numpy(), 'o', label='conditioner std')
  ax[1].plot(sigma_sim, tf.math.reduce_std(
      mcmc_samples, axis=[0, 1]), 'o', label='estimated std');
  ax[1].legend()

  ax[0].plot([min(mu), max(mu)], [min(mu), max(mu)])
  ax[1].plot([min(sigma_sim), max(sigma_sim)], [min(sigma_sim), max(sigma_sim)])
  ```

  #### References

  [1]: Stan Reference Manual.
  https://mc-stan.org/docs/2_23/reference-manual/hmc-algorithm-parameters.html#automatic-parameter-tuning
  """

  log_prob_val = target_log_prob(*prior_samples)
  log_prob_rank = ps.rank(log_prob_val)
  assert log_prob_rank == 1

  if constraining_bijectors is not None:
    target_log_prob_unconstrained = make_transformed_log_prob(
        target_log_prob,
        constraining_bijectors,
        direction='forward',
        enable_bijector_caching=False)
    # constrain to unconstrain
    inverse_transform = make_transform_fn(constraining_bijectors, 'inverse')
    # unconstrain to constrain
    forward_transform = make_transform_fn(constraining_bijectors, 'forward')
  else:
    target_log_prob_unconstrained = target_log_prob
    inverse_transform = lambda x: x
    forward_transform = lambda y: y

  prior_samples_unconstrained = inverse_transform(prior_samples)
  init_state_unconstrained = None

  # If the input to target_log_prob_fn is a nested structure of Tensors, we
  # flatten and concatenate them into a 1D vector so that it is easier to work
  # with in mass matrix adaptation.
  if tf.nest.is_nested(prior_samples_unconstrained):
    free_rv_event_shape = [x.shape[log_prob_rank:] for x in prior_samples]
    flat_event_splits = [s.num_elements() for s in free_rv_event_shape]

    # TODO(b/158878248): replace the two function below with `tfb.Split`.
    def split_and_reshape(x):
      assertions = []
      message = 'Input must have at least one dimension.'
      if tensorshape_util.rank(x.shape) is not None:
        if tensorshape_util.rank(x.shape) == 0:
          raise ValueError(message)
      else:
        assertions.append(
            assert_util.assert_rank_at_least(x, 1, message=message))
      with tf.control_dependencies(assertions):
        x = tf.nest.pack_sequence_as(free_rv_event_shape,
                                     tf.split(x, flat_event_splits, axis=-1))

        def _reshape_map_part(part, event_shape):
          static_rank = tf.get_static_value(ps.rank_from_shape(event_shape))
          if static_rank == 1:
            return part
          new_shape = ps.concat([ps.shape(part)[:-1], event_shape], axis=-1)
          return tf.reshape(part, ps.cast(new_shape, tf.int32))

        x = tf.nest.map_structure(_reshape_map_part, x, free_rv_event_shape)
      return x

    def concat_list_event(x):

      def handle_part(x, shape):
        if len(shape) == 0:  # pylint: disable=g-explicit-length-test
          return x[..., tf.newaxis]
        return tf.reshape(x, list(x.shape)[:-len(shape)] + [-1])

      flat_parts = [handle_part(v, s) for v, s in zip(x, free_rv_event_shape)]
      return tf.concat(flat_parts, axis=-1)

    def target_log_prob_unconstrained_concated(x):
      x = split_and_reshape(x)
      return target_log_prob_unconstrained(*x)

    prior_samples_unconstrained_concated = concat_list_event(
        prior_samples_unconstrained)
    if init_state is not None:
      init_state_unconstrained = concat_list_event(
          inverse_transform(init_state))
  else:
    target_log_prob_unconstrained_concated = target_log_prob_unconstrained
    prior_samples_unconstrained_concated = prior_samples_unconstrained
    split_and_reshape = lambda x: x
    if init_state is not None:
      init_state_unconstrained = inverse_transform(init_state)

  nuts_samples, diagnostic, conditioning_bijector = _sample_posterior(
      target_log_prob_unconstrained_concated,
      prior_samples_unconstrained_concated,
      init_state=init_state_unconstrained,
      num_samples=num_samples,
      nchains=nchains,
      init_nchains=init_nchains,
      target_accept_prob=target_accept_prob,
      max_tree_depth=max_tree_depth,
      use_scaled_init=use_scaled_init,
      tuning_window_schedule=tuning_window_schedule,
      use_wide_window_expanding_mode=use_wide_window_expanding_mode,
      seed=seed,
      parallel_iterations=parallel_iterations,
      jit_compile=jit_compile,
      use_input_signature=use_input_signature,
      experimental_relax_shapes=experimental_relax_shapes)
  return forward_transform(
      split_and_reshape(nuts_samples)), diagnostic, conditioning_bijector
