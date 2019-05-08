<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.sample_chain" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.sample_chain

Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.

``` python
tfp.mcmc.sample_chain(
    num_results,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    num_burnin_steps=0,
    num_steps_between_results=0,
    trace_fn=(lambda current_state, kernel_results: kernel_results),
    return_final_kernel_results=False,
    parallel_iterations=10,
    name=None
)
```



Defined in [`python/mcmc/sample.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/sample.py).

<!-- Placeholder for "Used in" -->

This function samples from an Markov chain at `current_state` and whose
stationary distribution is governed by the supplied `TransitionKernel`
instance (`kernel`).

This function can sample from multiple chains, in parallel. (Whether or not
there are multiple chains is dictated by the `kernel`.)

The `current_state` can be represented as a single `Tensor` or a `list` of
`Tensors` which collectively represent the current state.

Since MCMC states are correlated, it is sometimes desirable to produce
additional intermediate states, and then discard them, ending up with a set of
states with decreased autocorrelation.  See [Owen (2017)][1]. Such "thinning"
is made possible by setting `num_steps_between_results > 0`. The chain then
takes `num_steps_between_results` extra steps between the steps that make it
into the results. The extra steps are never materialized (in calls to
`sess.run`), and thus do not increase memory requirements.

Warning: when setting a `seed` in the `kernel`, ensure that `sample_chain`'s
`parallel_iterations=1`, otherwise results will not be reproducible.

In addition to returning the chain state, this function supports tracing of
auxiliary variables used by the kernel. The traced values are selected by
specifying `trace_fn`. By default, all kernel results are traced but in the
future the default will be changed to no results being traced, so plan
accordingly. See below for some examples of this feature.

#### Args:

* <b>`num_results`</b>: Integer number of Markov chain draws.
* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s).
* <b>`previous_kernel_results`</b>: A `Tensor` or a nested collection of `Tensor`s
  representing internal calculations made within the previous call to this
  function (or as returned by `bootstrap_results`).
* <b>`kernel`</b>: An instance of <a href="../../tfp/mcmc/TransitionKernel.md"><code>tfp.mcmc.TransitionKernel</code></a> which implements one step
  of the Markov chain.
* <b>`num_burnin_steps`</b>: Integer number of chain steps to take before starting to
  collect results.
  Default value: 0 (i.e., no burn-in).
* <b>`num_steps_between_results`</b>: Integer number of chain steps between collecting
  a result. Only one out of every `num_steps_between_samples + 1` steps is
  included in the returned results.  The number of returned chain states is
  still equal to `num_results`.  Default value: 0 (i.e., no thinning).
* <b>`trace_fn`</b>: A callable that takes in the current chain state and the previous
  kernel results and return a `Tensor` or a nested collection of `Tensor`s
  that is then traced along with the chain state.
* <b>`return_final_kernel_results`</b>: If `True`, then the final kernel results are
  returned alongside the chain state and the trace specified by the
  `trace_fn`.
* <b>`parallel_iterations`</b>: The number of iterations allowed to run in parallel. It
  must be a positive integer. See `tf.while_loop` for more details.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., "mcmc_sample_chain").


#### Returns:

  checkpointable_states_and_trace: if `return_final_kernel_results` is
    `True`. The return value is an instance of
    `CheckpointableStatesAndTrace`.
  all_states: if `return_final_kernel_results` is `False` and `trace_fn` is
    `None`. The return value is a `Tensor` or Python list of `Tensor`s
    representing the state(s) of the Markov chain(s) at each result step. Has
    same shape as input `current_state` but with a prepended
    `num_results`-size dimension.
  states_and_trace: if `return_final_kernel_results` is `False` and
    `trace_fn` is not `None`. The return value is an instance of
    `StatesAndTrace`.

#### Examples

##### Sample from a diagonal-variance Gaussian.

I.e.,

```none
for i=1..n:
  x[i] ~ MultivariateNormal(loc=0, scale=diag(true_stddev))  # likelihood
```

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dims = 10
true_stddev = np.sqrt(np.linspace(1., 3., dims))
likelihood = tfd.MultivariateNormalDiag(loc=0., scale_diag=true_stddev)

states = tfp.mcmc.sample_chain(
    num_results=1000,
    num_burnin_steps=500,
    current_state=tf.zeros(dims),
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=likelihood.log_prob,
      step_size=0.5,
      num_leapfrog_steps=2),
    trace_fn=None)

sample_mean = tf.reduce_mean(states, axis=0)
# ==> approx all zeros

sample_stddev = tf.sqrt(tf.reduce_mean(
    tf.squared_difference(states, sample_mean),
    axis=0))
# ==> approx equal true_stddev
```

##### Sampling from factor-analysis posteriors with known factors.

I.e.,

```none
# prior
w ~ MultivariateNormal(loc=0, scale=eye(d))
for i=1..n:
  # likelihood
  x[i] ~ Normal(loc=w^T F[i], scale=1)
```

where `F` denotes factors.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# Specify model.
def make_prior(dims):
  return tfd.MultivariateNormalDiag(
      loc=tf.zeros(dims))

def make_likelihood(weights, factors):
  return tfd.MultivariateNormalDiag(
      loc=tf.matmul(weights, factors, adjoint_b=True))

def joint_log_prob(num_weights, factors, x, w):
  return (make_prior(num_weights).log_prob(w) +
          make_likelihood(w, factors).log_prob(x))

def unnormalized_log_posterior(w):
  # Posterior is proportional to: `p(W, X=x | factors)`.
  return joint_log_prob(num_weights, factors, x, w)

# Setup data.
num_weights = 10 # == d
num_factors = 40 # == n
num_chains = 100

weights = make_prior(num_weights).sample(1)
factors = tf.random_normal([num_factors, num_weights])
x = make_likelihood(weights, factors).sample()

# Sample from Hamiltonian Monte Carlo Markov Chain.

# Get `num_results` samples from `num_chains` independent chains.
chains_states, kernels_results = tfp.mcmc.sample_chain(
    num_results=1000,
    num_burnin_steps=500,
    current_state=tf.zeros([num_chains, num_weights], name='init_weights'),
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=unnormalized_log_posterior,
      step_size=0.1,
      num_leapfrog_steps=2))

# Compute sample stats.
sample_mean = tf.reduce_mean(chains_states, axis=[0, 1])
# ==> approx equal to weights

sample_var = tf.reduce_mean(
    tf.squared_difference(chains_states, sample_mean),
    axis=[0, 1])
# ==> less than 1
```

##### Custom tracing functions.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

likelihood = tfd.Normal(loc=0., scale=1.)

def sample_chain(trace_fn):
  return tfp.mcmc.sample_chain(
    num_results=1000,
    num_burnin_steps=500,
    current_state=0.,
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=likelihood.log_prob,
      step_size=0.5,
      num_leapfrog_steps=2),
    trace_fn=trace_fn)

def trace_log_accept_ratio(states, previous_kernel_results):
  return previous_kernel_results.log_accept_ratio

def trace_everything(states, previous_kernel_results):
  return previous_kernel_results

_, log_accept_ratio = sample_chain(trace_fn=trace_log_accept_ratio)
_, kernel_results = sample_chain(trace_fn=trace_everything)

acceptance_prob = tf.exp(tf.minimum(log_accept_ratio_, 0.))
# Equivalent to, but more efficient than:
acceptance_prob = tf.exp(tf.minimum(kernel_results.log_accept_ratio_, 0.))
```

#### References

[1]: Art B. Owen. Statistically efficient thinning of a Markov chain sampler.
     _Technical Report_, 2017.
     http://statweb.stanford.edu/~owen/reports/bestthinning.pdf