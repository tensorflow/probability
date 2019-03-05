<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.sample_chain" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.sample_chain

``` python
tfp.mcmc.sample_chain(
    num_results,
    current_state,
    previous_kernel_results=None,
    kernel=None,
    num_burnin_steps=0,
    num_steps_between_results=0,
    parallel_iterations=10,
    name=None
)
```

Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.

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

#### Args:

* <b>`num_results`</b>: Integer number of Markov chain draws.
* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
    current state(s) of the Markov chain(s).
* <b>`previous_kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or
    `list` of `Tensor`s representing internal calculations made within the
    previous call to this function (or as returned by `bootstrap_results`).
* <b>`kernel`</b>: An instance of <a href="../../tfp/mcmc/TransitionKernel.md"><code>tfp.mcmc.TransitionKernel</code></a> which implements one step
    of the Markov chain.
* <b>`num_burnin_steps`</b>: Integer number of chain steps to take before starting to
    collect results.
    Default value: 0 (i.e., no burn-in).
* <b>`num_steps_between_results`</b>: Integer number of chain steps between collecting
    a result. Only one out of every `num_steps_between_samples + 1` steps is
    included in the returned results.  The number of returned chain states is
    still equal to `num_results`.  Default value: 0 (i.e., no thinning).
* <b>`parallel_iterations`</b>: The number of iterations allowed to run in parallel.
      It must be a positive integer. See `tf.while_loop` for more details.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., "mcmc_sample_chain").


#### Returns:

* <b>`next_states`</b>: Tensor or Python list of `Tensor`s representing the
    state(s) of the Markov chain(s) at each result step. Has same shape as
    input `current_state` but with a prepended `num_results`-size dimension.
* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
    `Tensor`s representing internal calculations made within this function.

#### Examples

##### Sample from a diagonal-variance Gaussian.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def make_likelihood(true_variances):
  return tfd.MultivariateNormalDiag(
      scale_diag=tf.sqrt(true_variances))

dims = 10
dtype = np.float32
true_variances = tf.linspace(dtype(1), dtype(3), dims)
likelihood = make_likelihood(true_variances)

states, kernel_results = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=tf.zeros(dims),
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=likelihood.log_prob,
      step_size=0.5,
      num_leapfrog_steps=2),
    num_burnin_steps=500)

# Compute sample stats.
sample_mean = tf.reduce_mean(states, axis=0)
sample_var = tf.reduce_mean(
    tf.squared_difference(states, sample_mean),
    axis=0)
```

##### Sampling from factor-analysis posteriors with known factors.

I.e.,

```none
for i=1..n:
  w[i] ~ Normal(0, eye(d))            # prior
  x[i] ~ Normal(loc=matmul(w[i], F))  # likelihood
```

where `F` denotes factors.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def make_prior(dims, dtype):
  return tfd.MultivariateNormalDiag(
      loc=tf.zeros(dims, dtype))

def make_likelihood(weights, factors):
  return tfd.MultivariateNormalDiag(
      loc=tf.tensordot(weights, factors, axes=[[0], [-1]]))

# Setup data.
num_weights = 10
num_factors = 4
num_chains = 100
dtype = np.float32

prior = make_prior(num_weights, dtype)
weights = prior.sample(num_chains)
factors = np.random.randn(num_factors, num_weights).astype(dtype)
x = make_likelihood(weights, factors).sample(num_chains)

def target_log_prob(w):
  # Target joint is: `f(w) = p(w, x | factors)`.
  return prior.log_prob(w) + make_likelihood(w, factors).log_prob(x)

# Get `num_results` samples from `num_chains` independent chains.
chains_states, kernels_results = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=tf.zeros([num_chains, dims], dtype),
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob,
      step_size=0.1,
      num_leapfrog_steps=2),
    num_burnin_steps=500)

# Compute sample stats.
sample_mean = tf.reduce_mean(chains_states, axis=[0, 1])
sample_var = tf.reduce_mean(
    tf.squared_difference(chains_states, sample_mean),
    axis=[0, 1])
```

#### References

[1]: Art B. Owen. Statistically efficient thinning of a Markov chain sampler.
     _Technical Report_, 2017.
     http://statweb.stanford.edu/~owen/reports/bestthinning.pdf