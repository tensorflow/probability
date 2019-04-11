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

# Dependency imports
import tensorflow as tf

from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python import mcmc
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
  unconstrained_prior_sample = parameter.bijector.inverse(
      parameter.prior.sample(init_sample_shape, seed=seed))
  uniform_initializer = 4 * tf.random.uniform(
      tf.shape(input=unconstrained_prior_sample),
      dtype=unconstrained_prior_sample.dtype,
      seed=seed) - 2
  if return_constrained:
    return parameter.bijector.forward(uniform_initializer)
  else:
    return uniform_initializer


def _build_trainable_posterior(param, initial_loc_fn):
  """Built a transformed-normal variational dist over a parameter's support."""
  loc = tf.compat.v1.get_variable(
      param.name + '_loc',
      initializer=lambda: initial_loc_fn(param),
      dtype=param.prior.dtype,
      use_resource=True)
  scale = tf.nn.softplus(
      tf.compat.v1.get_variable(
          param.name + '_scale',
          initializer=lambda: -4 * tf.ones_like(initial_loc_fn(param)),
          dtype=param.prior.dtype,
          use_resource=True))

  q = tfd.Normal(loc=loc, scale=scale)

  # Ensure the `event_shape` of the variational distribution matches the
  # parameter.
  if (param.prior.event_shape.ndims is None
      or param.prior.event_shape.ndims > 0):
    q = tfd.Independent(
        q, reinterpreted_batch_ndims=param.prior.event_shape.ndims)

  # Transform to constrained parameter space.
  return tfd.TransformedDistribution(q, param.bijector)


def build_factored_variational_loss(model,
                                    observed_time_series,
                                    init_batch_shape=(),
                                    seed=None,
                                    name=None):
  """Build a loss function for variational inference in STS models.

  Variational inference searches for the distribution within some family of
  approximate posteriors that minimizes a divergence between the approximate
  posterior `q(z)` and true posterior `p(z|observed_time_series)`. By converting
  inference to optimization, it's generally much faster than sampling-based
  inference algorithms such as HMC. The tradeoff is that the approximating
  family rarely contains the true posterior, so it may miss important aspects of
  posterior structure (in particular, dependence between variables) and should
  not be blindly trusted. Results may vary; it's generally wise to compare to
  HMC to evaluate whether inference quality is sufficient for your task at hand.

  This method constructs a loss function for variational inference using the
  Kullback-Liebler divergence `KL[q(z) || p(z|observed_time_series)]`, with an
  approximating family given by independent Normal distributions transformed to
  the appropriate parameter space for each parameter. Minimizing this loss (the
  negative ELBO) maximizes a lower bound on the log model evidence `-log
  p(observed_time_series)`. This is equivalent to the 'mean-field' method
  implemented in [1]. and is a standard approach. The resulting posterior
  approximations are unimodal; they will tend to underestimate posterior
  uncertainty when the true posterior contains multiple modes (the `KL[q||p]`
  divergence encourages choosing a single mode) or dependence between variables.

  Args:
    model: An instance of `StructuralTimeSeries` representing a
      time-series model. This represents a joint distribution over
      time-series and their parameters with batch shape `[b1, ..., bN]`.
    observed_time_series: `float` `Tensor` of shape
      `concat([sample_shape, model.batch_shape, [num_timesteps, 1]]) where
      `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
      dimension may (optionally) be omitted if `num_timesteps > 1`. May
      optionally be an instance of `tfp.sts.MaskedTimeSeries`, which includes
      a mask `Tensor` to specify timesteps with missing observations.
    init_batch_shape: Batch shape (Python `tuple`, `list`, or `int`) of initial
      states to optimize in parallel.
      Default value: `()`. (i.e., just run a single optimization).
    seed: Python integer to seed the random number generator.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'build_factored_variational_loss').

  Returns:
    variational_loss: `float` `Tensor` of shape
      `concat([init_batch_shape, model.batch_shape])`, encoding a stochastic
      estimate of an upper bound on the negative model evidence `-log p(y)`.
      Minimizing this loss performs variational inference; the gap between the
      variational bound and the true (generally unknown) model evidence
      corresponds to the divergence `KL[q||p]` between the approximate and true
      posterior.
    variational_distributions: `collections.OrderedDict` giving
      the approximate posterior for each model parameter. The keys are
      Python `str` parameter names in order, corresponding to
      `[param.name for param in model.parameters]`. The values are
      `tfd.Distribution` instances with batch shape
      `concat([init_batch_shape, model.batch_shape])`; these will typically be
      of the form `tfd.TransformedDistribution(tfd.Normal(...),
      bijector=param.bijector)`.

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

  To run variational inference, we simply construct the loss and optimize
  it:

  ```python
    (variational_loss,
     variational_distributions) = tfp.sts.build_factored_variational_loss(
       model=model, observed_time_series=observed_time_series)

    train_op = tf.train.AdamOptimizer(0.1).minimize(variational_loss)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for step in range(200):
        _, loss_ = sess.run((train_op, variational_loss))

        if step % 20 == 0:
          print("step {} loss {}".format(step, loss_))

      posterior_samples_ = sess.run({
        param_name: q.sample(50)
        for param_name, q in variational_distributions.items()})
  ```

  As a more complex example, we might try to avoid local optima by optimizing
  from multiple initializations in parallel, and selecting the result with the
  lowest loss:

  ```python
    (variational_loss,
     variational_distributions) = tfp.sts.build_factored_variational_loss(
       model=model, observed_time_series=observed_time_series,
       init_batch_shape=[10])

    train_op = tf.train.AdamOptimizer(0.1).minimize(variational_loss)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for step in range(200):
        _, loss_ = sess.run((train_op, variational_loss))

        if step % 20 == 0:
          print("step {} losses {}".format(step, loss_))

      # Draw multiple samples to reduce Monte Carlo error in the optimized
      # variational bounds.
      avg_loss = np.mean(
        [sess.run(variational_loss) for _ in range(25)], axis=0)
      best_posterior_idx = np.argmin(avg_loss, axis=0).astype(np.int32)
  ```

  #### References

  [1]: Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and
       David M. Blei. Automatic Differentiation Variational Inference. In
       _Journal of Machine Learning Research_, 2017.
       https://arxiv.org/abs/1603.00788

  """

  with tf.compat.v1.name_scope(
      name, 'build_factored_variational_loss',
      values=[observed_time_series]) as name:
    seed = tfd.SeedStream(
        seed, salt='StructuralTimeSeries_build_factored_variational_loss')

    variational_distributions = collections.OrderedDict()
    variational_samples = []
    for param in model.parameters:
      def initial_loc_fn(param):
        return sample_uniform_initial_state(
            param, return_constrained=True,
            init_sample_shape=init_batch_shape,
            seed=seed())
      q = _build_trainable_posterior(param, initial_loc_fn=initial_loc_fn)
      variational_distributions[param.name] = q
      variational_samples.append(q.sample(seed=seed()))

    # Multiple initializations (similar to HMC chains) manifest as an extra
    # param batch dimension, so we need to add corresponding batch dimension(s)
    # to `observed_time_series`.
    observed_time_series = sts_util.pad_batch_dimension_for_multiple_chains(
        observed_time_series, model, chain_batch_shape=init_batch_shape)

    # Construct the variational bound.
    log_prob_fn = model.joint_log_prob(observed_time_series)
    expected_log_joint = log_prob_fn(*variational_samples)
    entropy = tf.reduce_sum(
        input_tensor=[
            -q.log_prob(sample) for (q, sample) in zip(
                variational_distributions.values(), variational_samples)
        ],
        axis=0)
    variational_loss = -(expected_log_joint + entropy)  # -ELBO

  return variational_loss, variational_distributions


def _minimize_in_graph(build_loss_fn, num_steps=200, optimizer=None):
  """Run an optimizer within the graph to minimize a loss function."""
  optimizer = tf.compat.v1.train.AdamOptimizer(
      0.1) if optimizer is None else optimizer

  def train_loop_body(step):
    train_op = optimizer.minimize(
        build_loss_fn if tf.executing_eagerly() else build_loss_fn())
    return tf.tuple(tensors=[tf.add(step, 1)], control_inputs=[train_op])

  minimize_op = tf.compat.v1.while_loop(
      cond=lambda step: step < num_steps,
      body=train_loop_body,
      loop_vars=[tf.constant(0)],
      return_same_structure=True)[0]  # Always return a single op.
  return minimize_op


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
      `concat([sample_shape, model.batch_shape, [num_timesteps, 1]]) where
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

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    samples_, kernel_results_ = sess.run((samples, kernel_results))

  print("acceptance rate: {}".format(
    np.mean(kernel_results_.inner_results.is_accepted, axis=0)))
  print("posterior means: {}".format(
    {param.name: np.mean(param_draws, axis=0)
     for (param, param_draws) in zip(model.parameters, samples_)}))
  ```

  We can also run multiple chains. This may help diagnose convergence issues
  and allows us to exploit vectorization to draw samples more quickly, although
  warmup still requires the same number of sequential steps.

  ```python
  from matplotlib import pylab as plt

  samples, kernel_results = tfp.sts.fit_with_hmc(
    model, observed_time_series, chain_batch_shape=[10])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    samples_, kernel_results_ = sess.run((samples, kernel_results))

  print("acceptance rate: {}".format(
    np.mean(kernel_results_.inner_results.inner_results.is_accepted, axis=0)))

  # Plot the sampled traces for each parameter. If the chains have mixed, their
  # traces should all cover the same region of state space, frequently crossing
  # over each other.
  for (param, param_draws) in zip(model.parameters, samples_):
    if param.prior.event_shape.ndims > 0:
      print("Only plotting traces for scalar parameters, skipping {}".format(
        param.name))
      continue
    plt.figure(figsize=[10, 4])
    plt.title(param.name)
    plt.plot(param_draws)
    plt.ylabel(param.name)
    plt.xlabel("HMC step")

  # Combining the samples from multiple chains into a single dimension allows
  # us to easily pass sampled parameters to downstream forecasting methods.
  combined_samples_ = [np.reshape(param_draws,
                                  [-1] + list(param_draws.shape[2:]))
                       for param_draws in samples_]
  ```

  For greater flexibility, you may prefer to implement your own sampler using
  the TensorFlow Probability primitives in `tfp.mcmc`. The following recipe
  constructs a basic HMC sampler, using a `TransformedTransitionKernel` to
  incorporate constraints on the parameter space.

  ```python
  transformed_hmc_kernel = mcmc.TransformedTransitionKernel(
      inner_kernel=mcmc.SimpleStepSizeAdaptation(
          inner_kernel=mcmc.HamiltonianMonteCarlo(
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
  with tf.compat.v1.name_scope(
      name, 'fit_with_hmc', values=[observed_time_series]) as name:
    seed = tfd.SeedStream(seed, salt='StructuralTimeSeries_fit_with_hmc')

    # Initialize state and step sizes from a variational posterior if not
    # specified.
    if initial_step_size is None or initial_state is None:

      # To avoid threading variational distributions through the training
      # while loop, we build our own copy here. `make_template` ensures
      # that our variational distributions share the optimized parameters.
      def make_variational():
        return build_factored_variational_loss(
            model, observed_time_series,
            init_batch_shape=chain_batch_shape, seed=seed())

      make_variational = tf.compat.v1.make_template('make_variational',
                                                    make_variational)
      _, variational_distributions = make_variational()
      minimize_op = _minimize_in_graph(
          build_loss_fn=lambda: make_variational()[0],  # return just the loss.
          num_steps=num_variational_steps,
          optimizer=variational_optimizer)

      with tf.control_dependencies([minimize_op]):
        if initial_state is None:
          initial_state = [tf.stop_gradient(d.sample())
                           for d in variational_distributions.values()]

        # Set step sizes using the unconstrained variational distribution.
        if initial_step_size is None:
          initial_step_size = [
              transformed_q.distribution.stddev()
              for transformed_q in variational_distributions.values()]

    # Multiple chains manifest as an extra param batch dimension, so we need to
    # add a corresponding batch dimension to `observed_time_series`.
    observed_time_series = sts_util.pad_batch_dimension_for_multiple_chains(
        observed_time_series, model, chain_batch_shape=chain_batch_shape)

    # Run HMC to sample from the posterior on parameters.
    samples, kernel_results = mcmc.sample_chain(
        num_results=num_results,
        current_state=initial_state,
        num_burnin_steps=num_warmup_steps,
        kernel=mcmc.SimpleStepSizeAdaptation(
            inner_kernel=mcmc.TransformedTransitionKernel(
                inner_kernel=mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=model.joint_log_prob(
                        observed_time_series),
                    step_size=initial_step_size,
                    num_leapfrog_steps=num_leapfrog_steps,
                    state_gradients_are_stopped=True,
                    seed=seed()),
                bijector=[param.bijector for param in model.parameters]),
            num_adaptation_steps=int(num_warmup_steps * 0.8),
            adaptation_rate=tf.convert_to_tensor(
                value=0.1, dtype=initial_state[0].dtype)),
        parallel_iterations=1 if seed is not None else 10)

    return samples, kernel_results
