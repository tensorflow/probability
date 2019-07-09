<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.fit_with_hmc" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.fit_with_hmc

Draw posterior samples using Hamiltonian Monte Carlo (HMC).

``` python
tfp.sts.fit_with_hmc(
    model,
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
    name=None
)
```



Defined in [`python/sts/fitting.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/fitting.py).

<!-- Placeholder for "Used in" -->

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


#### Args:


* <b>`model`</b>: An instance of `StructuralTimeSeries` representing a
  time-series model. This represents a joint distribution over
  time-series and their parameters with batch shape `[b1, ..., bN]`.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape
  `concat([sample_shape, model.batch_shape, [num_timesteps, 1]]) where
  `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
  dimension may (optionally) be omitted if `num_timesteps > 1`. May
  optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>, which includes
  a mask `Tensor` to specify timesteps with missing observations.
* <b>`num_results`</b>: Integer number of Markov chain draws.
  Default value: `100`.
* <b>`num_warmup_steps`</b>: Integer number of steps to take before starting to
  collect results. The warmup steps are also used to adapt the step size
  towards a target acceptance rate of 0.75.
  Default value: `50`.
* <b>`num_leapfrog_steps`</b>: Integer number of steps to run the leapfrog integrator
  for. Total progress per HMC step is roughly proportional to
  `step_size * num_leapfrog_steps`.
  Default value: `15`.
* <b>`initial_state`</b>: Optional Python `list` of `Tensor`s, one for each model
  parameter, representing the initial state(s) of the Markov chain(s). These
  should have shape `concat([chain_batch_shape, param.prior.batch_shape,
  param.prior.event_shape])`. If `None`, the initial state is set
  automatically using a sample from a variational posterior.
  Default value: `None`.
* <b>`initial_step_size`</b>: Python `list` of `Tensor`s, one for each model parameter,
  representing the step size for the leapfrog integrator. Must
  broadcast with the shape of `initial_state`. Larger step sizes lead to
  faster progress, but too-large step sizes make rejection exponentially
  more likely. If `None`, the step size is set automatically using the
  standard deviation of a variational posterior.
  Default value: `None`.
* <b>`chain_batch_shape`</b>: Batch shape (Python `tuple`, `list`, or `int`) of chains
  to run in parallel.
  Default value: `[]` (i.e., a single chain).
* <b>`num_variational_steps`</b>: Python `int` number of steps to run the variational
  optimization to determine the initial state and step sizes.
  Default value: `150`.
* <b>`variational_optimizer`</b>: Optional `tf.train.Optimizer` instance to use in
  the variational optimization. If `None`, defaults to
  `tf.train.AdamOptimizer(0.1)`.
  Default value: `None`.
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` (i.e., 'fit_with_hmc').


#### Returns:


* <b>`samples`</b>: Python `list` of `Tensors` representing posterior samples of model
  parameters, with shapes `[concat([[num_results], chain_batch_shape,
  param.prior.batch_shape, param.prior.event_shape]) for param in
  model.parameters]`.
* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
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
the TensorFlow Probability primitives in <a href="../../tfp/mcmc.md"><code>tfp.mcmc</code></a>. The following recipe
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