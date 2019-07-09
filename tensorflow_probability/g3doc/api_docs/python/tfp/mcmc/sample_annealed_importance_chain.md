<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.sample_annealed_importance_chain" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.sample_annealed_importance_chain

Runs annealed importance sampling (AIS) to estimate normalizing constants.

``` python
tfp.mcmc.sample_annealed_importance_chain(
    num_steps,
    proposal_log_prob_fn,
    target_log_prob_fn,
    current_state,
    make_kernel_fn,
    parallel_iterations=10,
    name=None
)
```



Defined in [`python/mcmc/sample_annealed_importance.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/sample_annealed_importance.py).

<!-- Placeholder for "Used in" -->

This function uses an MCMC transition operator (e.g., Hamiltonian Monte Carlo)
to sample from a series of distributions that slowly interpolates between
an initial "proposal" distribution:

`exp(proposal_log_prob_fn(x) - proposal_log_normalizer)`

and the target distribution:

`exp(target_log_prob_fn(x) - target_log_normalizer)`,

accumulating importance weights along the way. The product of these
importance weights gives an unbiased estimate of the ratio of the
normalizing constants of the initial distribution and the target
distribution:

`E[exp(ais_weights)] = exp(target_log_normalizer - proposal_log_normalizer)`.

Note: When running in graph mode, `proposal_log_prob_fn` and
`target_log_prob_fn` are called exactly three times (although this may be
reduced to two times in the future).

#### Args:


* <b>`num_steps`</b>: Integer number of Markov chain updates to run. More
  iterations means more expense, but smoother annealing between q
  and p, which in turn means exponentially lower variance for the
  normalizing constant estimator.
* <b>`proposal_log_prob_fn`</b>: Python callable that returns the log density of the
  initial distribution.
* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
  `current_state` (or `*current_state` if it's a list) and returns its
  (possibly unnormalized) log-density under the target distribution.
* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s). The first `r` dimensions index
  independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
* <b>`make_kernel_fn`</b>: Python `callable` which returns a `TransitionKernel`-like
  object. Must take one argument representing the `TransitionKernel`'s
  `target_log_prob_fn`. The `target_log_prob_fn` argument represents the
  `TransitionKernel`'s target log distribution.  Note:
  `sample_annealed_importance_chain` creates a new `target_log_prob_fn`
  which is an interpolation between the supplied `target_log_prob_fn` and
  `proposal_log_prob_fn`; it is this interpolated function which is used as
  an argument to `make_kernel_fn`.
* <b>`parallel_iterations`</b>: The number of iterations allowed to run in parallel.
    It must be a positive integer. See `tf.while_loop` for more details.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., "sample_annealed_importance_chain").


#### Returns:


* <b>`next_state`</b>: `Tensor` or Python list of `Tensor`s representing the
  state(s) of the Markov chain(s) at the final iteration. Has same shape as
  input `current_state`.
* <b>`ais_weights`</b>: Tensor with the estimated weight(s). Has shape matching
  `target_log_prob_fn(current_state)`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.

#### Examples

##### Estimate the normalizing constant of a log-gamma distribution.

```python
tfd = tfp.distributions

# Run 100 AIS chains in parallel
num_chains = 100
dims = 20
dtype = np.float32

proposal = tfd.MultivariateNormalDiag(
   loc=tf.zeros([dims], dtype=dtype))

target = tfd.TransformedDistribution(
  distribution=tfd.Gamma(concentration=dtype(2),
                         rate=dtype(3)),
  bijector=tfp.bijectors.Invert(tfp.bijectors.Exp()),
  event_shape=[dims])

chains_state, ais_weights, kernels_results = (
    tfp.mcmc.sample_annealed_importance_chain(
        num_steps=1000,
        proposal_log_prob_fn=proposal.log_prob,
        target_log_prob_fn=target.log_prob,
        current_state=proposal.sample(num_chains),
        make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=tlp_fn,
          step_size=0.2,
          num_leapfrog_steps=2)))

log_estimated_normalizer = (tf.reduce_logsumexp(ais_weights)
                            - np.log(num_chains))
log_true_normalizer = tf.lgamma(2.) - 2. * tf.log(3.)
```

##### Estimate marginal likelihood of a Bayesian regression model.

```python
tfd = tfp.distributions

def make_prior(dims, dtype):
  return tfd.MultivariateNormalDiag(
      loc=tf.zeros(dims, dtype))

def make_likelihood(weights, x):
  return tfd.MultivariateNormalDiag(
      loc=tf.tensordot(weights, x, axes=[[0], [-1]]))

# Run 100 AIS chains in parallel
num_chains = 100
dims = 10
dtype = np.float32

# Make training data.
x = np.random.randn(num_chains, dims).astype(dtype)
true_weights = np.random.randn(dims).astype(dtype)
y = np.dot(x, true_weights) + np.random.randn(num_chains)

# Setup model.
prior = make_prior(dims, dtype)
def target_log_prob_fn(weights):
  return prior.log_prob(weights) + make_likelihood(weights, x).log_prob(y)

proposal = tfd.MultivariateNormalDiag(
    loc=tf.zeros(dims, dtype))

weight_samples, ais_weights, kernel_results = (
    tfp.mcmc.sample_annealed_importance_chain(
      num_steps=1000,
      proposal_log_prob_fn=proposal.log_prob,
      target_log_prob_fn=target_log_prob_fn
      current_state=tf.zeros([num_chains, dims], dtype),
      make_kernel_fn=lambda tlp_fn: tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=tlp_fn,
        step_size=0.1,
        num_leapfrog_steps=2)))
log_normalizer_estimate = (tf.reduce_logsumexp(ais_weights)
                           - np.log(num_chains))
```