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
"""Methods for fitting StructuralTimeSeries models to data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python import mcmc
from tensorflow_probability.python import util as tfp_util
from tensorflow_probability.python import vi
from tensorflow_probability.python.bijectors import softplus as softplus_lib
from tensorflow_probability.python.distributions import independent as independent_lib
from tensorflow_probability.python.distributions import joint_distribution_named as joint_distribution_named_lib
from tensorflow_probability.python.distributions import normal as normal_lib
from tensorflow_probability.python.distributions import transformed_distribution as transformed_distribution_lib
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.sts.internal import util as sts_util


def sample_uniform_initial_state(parameter,
                                 return_constrained=True,
                                 init_sample_shape=(),
                                 seed=None):
  """Initialize from a uniform [-2, 2] distribution in unconstrained space.

  Args:
    parameter: `sts.Parameter` named tuple instance.
    return_constrained: if `True`, re-applies the constraining bijector
      to return initializations in the original domain. Otherwise, returns
      initializations in the unconstrained space.
      Default value: `True`.
    init_sample_shape: `sample_shape` of the sampled initializations.
      Default value: `[]`.
    seed: Python integer to seed the random number generator.

  Returns:
    uniform_initializer: `Tensor` of shape `concat([init_sample_shape,
    parameter.prior.batch_shape, transformed_event_shape])`, where
    `transformed_event_shape` is `parameter.prior.event_shape`, if
    `return_constrained=True`, and otherwise it is
    `parameter.bijector.inverse_event_shape(parameteter.prior.event_shape)`.
  """
  # Get the shape and dtype of the unconstrained parameter value we'll
  # use in inference, which should match the shape of a prior sample. If
  # possible, elide the actual sampling by using `tf.function` to
  # read off the shape statically.
  unconstrained_prior_sample_fn = tf.function(
      lambda: parameter.bijector.inverse(  # pylint: disable=g-long-lambda
          parameter.prior.sample(init_sample_shape)))
  param_shape = (
      unconstrained_prior_sample_fn.get_concrete_function().output_shapes)
  if not tensorshape_util.is_fully_defined(param_shape):
    param_shape = tf.shape(unconstrained_prior_sample_fn())

  uniform_initializer = 4 * samplers.uniform(
      shape=param_shape,
      dtype=unconstrained_prior_sample_fn.get_concrete_function().output_dtypes,
      seed=seed) - 2
  if return_constrained:
    return parameter.bijector.forward(uniform_initializer)
  else:
    return uniform_initializer


def _build_posterior_for_one_parameter(param, batch_shape, seed):
  """Built a transformed-normal variational dist over a parameter's support."""

  # Build a trainable Normal distribution.
  initial_loc = sample_uniform_initial_state(
      param, init_sample_shape=batch_shape,
      return_constrained=False, seed=seed)
  loc = tf.Variable(initial_value=initial_loc,
                    name=param.name + '_loc')
  scale = tfp_util.TransformedVariable(
      tf.fill(tf.shape(initial_loc),
              value=tf.constant(0.02, initial_loc.dtype),
              name=param.name + '_scale'),
      softplus_lib.Softplus())
  posterior_dist = normal_lib.Normal(loc=loc, scale=scale)

  # Ensure the `event_shape` of the variational distribution matches the
  # parameter.
  if (param.prior.event_shape.ndims is None
      or param.prior.event_shape.ndims > 0):
    posterior_dist = independent_lib.Independent(
        posterior_dist, reinterpreted_batch_ndims=param.prior.event_shape.ndims)

  # Transform to constrained parameter space.
  posterior_dist = transformed_distribution_lib.TransformedDistribution(
      posterior_dist, param.bijector, name='{}_posterior'.format(param.name))
  return posterior_dist


def build_factored_surrogate_posterior(
    model,
    batch_shape=(),
    seed=None,
    name=None):
  """Build a variational posterior that factors over model parameters.

  The surrogate posterior consists of independent Normal distributions for
  each parameter with trainable `loc` and `scale`, transformed using the
  parameter's `bijector` to the appropriate support space for that parameter.

  Args:
    model: An instance of `StructuralTimeSeries` representing a
        time-series model. This represents a joint distribution over
        time-series and their parameters with batch shape `[b1, ..., bN]`.
    batch_shape: Batch shape (Python `tuple`, `list`, or `int`) of initial
      states to optimize in parallel.
      Default value: `()`. (i.e., just run a single optimization).
    seed: Python integer to seed the random number generator.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'build_factored_surrogate_posterior').
  Returns:
    variational_posterior: `tfd.JointDistributionNamed` defining a trainable
        surrogate posterior over model parameters. Samples from this
        distribution are Python `dict`s with Python `str` parameter names as
        keys.

  ### Examples

  Assume we've built a structural time-series model:

  ```python
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week')
    local_linear_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_linear_trend')
    model = tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                        observed_time_series=observed_time_series)
  ```

  To fit the model to data, we define a surrogate posterior and fit it
  by optimizing a variational bound:

  ```python
    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
      model=model)
    loss_curve = tfp.vi.fit_surrogate_posterior(
      target_log_prob_fn=model.joint_log_prob(observed_time_series),
      surrogate_posterior=surrogate_posterior,
      optimizer=tf.optimizers.Adam(learning_rate=0.1),
      num_steps=200)
    posterior_samples = surrogate_posterior.sample(50)

    # In graph mode, we would need to write:
    # with tf.control_dependencies([loss_curve]):
    #   posterior_samples = surrogate_posterior.sample(50)
  ```

  For more control, we can also build and optimize a variational loss
  manually:

  ```python
    @tf.function(autograph=False)  # Ensure the loss is computed efficiently
    def loss_fn():
      return tfp.vi.monte_carlo_variational_loss(
        model.joint_log_prob(observed_time_series),
        surrogate_posterior,
        sample_size=10)

    optimizer = tf.optimizers.Adam(learning_rate=0.1)
    for step in range(200):
      with tf.GradientTape() as tape:
        loss = loss_fn()
      grads = tape.gradient(loss, surrogate_posterior.trainable_variables)
      optimizer.apply_gradients(
        zip(grads, surrogate_posterior.trainable_variables))
      if step % 20 == 0:
        print('step {} loss {}'.format(step, loss))

    posterior_samples = surrogate_posterior.sample(50)
  ```

  """
  with tf.name_scope(name or 'build_factored_surrogate_posterior'):
    seed = tfp_util.SeedStream(
        seed, salt='StructuralTimeSeries_build_factored_surrogate_posterior')
    variational_posterior = collections.OrderedDict()
    for param in model.parameters:
      variational_posterior[param.name] = _build_posterior_for_one_parameter(
          param, batch_shape=batch_shape, seed=seed())
    return joint_distribution_named_lib.JointDistributionNamed(
        variational_posterior)


def fit_with_hmc(model,
                 observed_time_series,
                 num_results=100,
                 num_warmup_steps=50,
                 num_leapfrog_steps=15,
                 initial_state=None,
                 initial_step_size=None,
                 chain_batch_shape=(),
                 num_variational_steps=150,
                 variational_optimizer=None,
                 variational_sample_size=5,
                 seed=None,
                 name=None):
  """Draw posterior samples using Hamiltonian Monte Carlo (HMC).

  Markov chain Monte Carlo (MCMC) methods are considered the gold standard of
  Bayesian inference; under suitable conditions and in the limit of infinitely
  many draws they generate samples from the true posterior distribution. HMC [1]
  uses gradients of the model's log-density function to propose samples,
  allowing it to exploit posterior geometry. However, it is computationally more
  expensive than variational inference and relatively sensitive to tuning.

  This method attempts to provide a sensible default approach for fitting
  StructuralTimeSeries models using HMC. It first runs variational inference as
  a fast posterior approximation, and initializes the HMC sampler from the
  variational posterior, using the posterior standard deviations to set
  per-variable step sizes (equivalently, a diagonal mass matrix). During the
  warmup phase, it adapts the step size to target an acceptance rate of 0.75,
  which is thought to be in the desirable range for optimal mixing [2].


  Args:
    model: An instance of `StructuralTimeSeries` representing a
      time-series model. This represents a joint distribution over
      time-series and their parameters with batch shape `[b1, ..., bN]`.
    observed_time_series: `float` `Tensor` of shape
      `concat([sample_shape, model.batch_shape, [num_timesteps, 1]])` where
      `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
      dimension may (optionally) be omitted if `num_timesteps > 1`. May
      optionally be an instance of `tfp.sts.MaskedTimeSeries`, which includes
      a mask `Tensor` to specify timesteps with missing observations.
    num_results: Integer number of Markov chain draws.
      Default value: `100`.
    num_warmup_steps: Integer number of steps to take before starting to
      collect results. The warmup steps are also used to adapt the step size
      towards a target acceptance rate of 0.75.
      Default value: `50`.
    num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
      for. Total progress per HMC step is roughly proportional to
      `step_size * num_leapfrog_steps`.
      Default value: `15`.
    initial_state: Optional Python `list` of `Tensor`s, one for each model
      parameter, representing the initial state(s) of the Markov chain(s). These
      should have shape `concat([chain_batch_shape, param.prior.batch_shape,
      param.prior.event_shape])`. If `None`, the initial state is set
      automatically using a sample from a variational posterior.
      Default value: `None`.
    initial_step_size: Python `list` of `Tensor`s, one for each model parameter,
      representing the step size for the leapfrog integrator. Must
      broadcast with the shape of `initial_state`. Larger step sizes lead to
      faster progress, but too-large step sizes make rejection exponentially
      more likely. If `None`, the step size is set automatically using the
      standard deviation of a variational posterior.
      Default value: `None`.
    chain_batch_shape: Batch shape (Python `tuple`, `list`, or `int`) of chains
      to run in parallel.
      Default value: `[]` (i.e., a single chain).
    num_variational_steps: Python `int` number of steps to run the variational
      optimization to determine the initial state and step sizes.
      Default value: `150`.
    variational_optimizer: Optional `tf.train.Optimizer` instance to use in
      the variational optimization. If `None`, defaults to
      `tf.train.AdamOptimizer(0.1)`.
      Default value: `None`.
    variational_sample_size: Python `int` number of Monte Carlo samples to use
      in estimating the variational divergence. Larger values may stabilize
      the optimization, but at higher cost per step in time and memory.
      Default value: `1`.
    seed: Python integer to seed the random number generator.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'fit_with_hmc').

  Returns:
    samples: Python `list` of `Tensors` representing posterior samples of model
      parameters, with shapes `[concat([[num_results], chain_batch_shape,
      param.prior.batch_shape, param.prior.event_shape]) for param in
      model.parameters]`.
    kernel_results: A (possibly nested) `tuple`, `namedtuple` or `list` of
      `Tensor`s representing internal calculations made within the HMC sampler.

  #### Examples

  Assume we've built a structural time-series model:

  ```python
    day_of_week = tfp.sts.Seasonal(
        num_seasons=7,
        observed_time_series=observed_time_series,
        name='day_of_week')
    local_linear_trend = tfp.sts.LocalLinearTrend(
        observed_time_series=observed_time_series,
        name='local_linear_trend')
    model = tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                        observed_time_series=observed_time_series)
  ```

  To draw posterior samples using HMC under default settings:

  ```python
  samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
  print("acceptance rate: {}".format(
    np.mean(kernel_results.inner_results.inner_results.is_accepted, axis=0)))
  print("posterior means: {}".format(
    {param.name: np.mean(param_draws, axis=0)
     for (param, param_draws) in zip(model.parameters, samples)}))
  ```

  We can also run multiple chains. This may help diagnose convergence issues
  and allows us to exploit vectorization to draw samples more quickly, although
  warmup still requires the same number of sequential steps.

  ```python
  from matplotlib import pylab as plt

  samples, kernel_results = tfp.sts.fit_with_hmc(
    model, observed_time_series, chain_batch_shape=[10])
  print("acceptance rate: {}".format(
    np.mean(kernel_results.inner_results.inner_results.is_accepted, axis=0)))

  # Plot the sampled traces for each parameter. If the chains have mixed, their
  # traces should all cover the same region of state space, frequently crossing
  # over each other.
  for (param, param_draws) in zip(model.parameters, samples):
    if param.prior.event_shape.ndims > 0:
      print("Only plotting traces for scalar parameters, skipping {}".format(
        param.name))
      continue
    plt.figure(figsize=[10, 4])
    plt.title(param.name)
    plt.plot(param_draws.numpy())
    plt.ylabel(param.name)
    plt.xlabel("HMC step")

  # Combining the samples from multiple chains into a single dimension allows
  # us to easily pass sampled parameters to downstream forecasting methods.
  combined_samples = [np.reshape(param_draws,
                                 [-1] + list(param_draws.shape[2:]))
                      for param_draws in samples]
  ```

  For greater flexibility, you may prefer to implement your own sampler using
  the TensorFlow Probability primitives in `tfp.mcmc`. The following recipe
  constructs a basic HMC sampler, using a `TransformedTransitionKernel` to
  incorporate constraints on the parameter space.

  ```python
  transformed_hmc_kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=tfp.mcmc.DualAveragingStepSizeAdaptation(
          inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=model.joint_log_prob(observed_time_series),
              step_size=step_size,
              num_leapfrog_steps=num_leapfrog_steps,
              state_gradients_are_stopped=True,
              seed=seed),
          num_adaptation_steps = int(0.8 * num_warmup_steps)),
      bijector=[param.bijector for param in model.parameters])

  # Initialize from a Uniform[-2, 2] distribution in unconstrained space.
  initial_state = [tfp.sts.sample_uniform_initial_state(
    param, return_constrained=True) for param in model.parameters]

  samples, kernel_results = tfp.mcmc.sample_chain(
    kernel=transformed_hmc_kernel,
    num_results=num_results,
    current_state=initial_state,
    num_burnin_steps=num_warmup_steps)
  ```

  #### References

  [1]: Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain
       Monte Carlo_, 2011. https://arxiv.org/abs/1206.1901
  [2]  M.J. Betancourt, Simon Byrne, and Mark Girolami. Optimizing The
       Integrator Step Size for Hamiltonian Monte Carlo.
       https://arxiv.org/abs/1411.6669

  """
  with tf.name_scope(name or 'fit_with_hmc') as name:
    seed = tfp_util.SeedStream(seed, salt='StructuralTimeSeries_fit_with_hmc')

    observed_time_series = sts_util.pad_batch_dimension_for_multiple_chains(
        observed_time_series, model, chain_batch_shape=chain_batch_shape)
    target_log_prob_fn = model.joint_log_prob(observed_time_series)

    # Initialize state and step sizes from a variational posterior if not
    # specified.
    if initial_step_size is None or initial_state is None:
      variational_posterior = build_factored_surrogate_posterior(
          model, batch_shape=chain_batch_shape, seed=seed())

      if variational_optimizer is None:
        variational_optimizer = tf1.train.AdamOptimizer(
            learning_rate=0.1)  # TODO(b/137299119) Replace with TF2 optimizer.
      loss_curve = vi.fit_surrogate_posterior(
          target_log_prob_fn,
          variational_posterior,
          sample_size=variational_sample_size,
          num_steps=num_variational_steps,
          optimizer=variational_optimizer,
          seed=seed())

      with tf.control_dependencies([loss_curve]):
        if initial_state is None:
          posterior_sample = variational_posterior.sample()
          initial_state = [posterior_sample[p.name] for p in model.parameters]

        # Set step sizes using the unconstrained variational distribution.
        if initial_step_size is None:
          q_dists_by_name, _ = variational_posterior.sample_distributions()
          initial_step_size = [
              q_dists_by_name[p.name].distribution.stddev()
              for p in model.parameters]

    # Run HMC to sample from the posterior on parameters.
    @tf.function(autograph=False)
    def run_hmc():
      return mcmc.sample_chain(
          num_results=num_results,
          current_state=initial_state,
          num_burnin_steps=num_warmup_steps,
          kernel=mcmc.DualAveragingStepSizeAdaptation(
              inner_kernel=mcmc.TransformedTransitionKernel(
                  inner_kernel=mcmc.HamiltonianMonteCarlo(
                      target_log_prob_fn=target_log_prob_fn,
                      step_size=initial_step_size,
                      num_leapfrog_steps=num_leapfrog_steps,
                      state_gradients_are_stopped=True),
                  bijector=[param.bijector for param in model.parameters]),
              num_adaptation_steps=int(num_warmup_steps * 0.8)),
          seed=seed())
    samples, kernel_results = run_hmc()

    return samples, kernel_results
