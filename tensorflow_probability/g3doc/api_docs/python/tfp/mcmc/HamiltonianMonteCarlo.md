<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.HamiltonianMonteCarlo" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_leapfrog_steps"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="state_gradients_are_stopped"/>
<meta itemprop="property" content="step_size"/>
<meta itemprop="property" content="step_size_update_fn"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.HamiltonianMonteCarlo

## Class `HamiltonianMonteCarlo`

Runs one step of Hamiltonian Monte Carlo.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/hmc.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/hmc.py).

<!-- Placeholder for "Used in" -->

Hamiltonian Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm
that takes a series of gradient-informed steps to produce a Metropolis
proposal. This class implements one random HMC step from a given
`current_state`. Mathematical details and derivations can be found in
[Neal (2011)][1].

The `one_step` function can update multiple chains in parallel. It assumes
that all leftmost dimensions of `current_state` index independent chain states
(and are therefore updated independently). The output of
`target_log_prob_fn(*current_state)` should sum log-probabilities across all
event dimensions. Slices along the rightmost dimensions may have different
target distributions; for example, `current_state[0, :]` could have a
different target distribution from `current_state[1, :]`. These semantics are
governed by `target_log_prob_fn(*current_state)`. (The number of independent
chains is `tf.size(target_log_prob_fn(*current_state))`.)

#### Examples:

##### Simple chain with warm-up.

In this example we sample from a standard univariate normal
distribution using HMC with adaptive step size.

```python
import tensorflow as tf
import tensorflow_probability as tfp

tf.enable_eager_execution()

# Target distribution is proportional to: `exp(-x (1 + x))`.
def unnormalized_log_prob(x):
  return -x - x**2.

# Initialize the HMC transition kernel.
num_results = int(10e3)
num_burnin_steps = int(1e3)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_log_prob,
        num_leapfrog_steps=3,
        step_size=1.),
    num_adaptation_steps=int(num_burnin_steps * 0.8))

# Run the chain (with burn-in).
@tf.function
def run_chain():
  # Run the chain (with burn-in).
  samples, is_accepted = tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=1.,
      kernel=adaptive_hmc,
      trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

  sample_mean = tf.reduce_mean(samples)
  sample_stddev = tf.math.reduce_std(samples)
  is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
  return sample_mean, sample_stddev, is_accepted

sample_mean, sample_stddev, is_accepted = run_chain()

print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
    sample_mean.numpy(), sample_stddev.numpy(), is_accepted.numpy()))
```

##### Estimate parameters of a more complicated posterior.

In this example, we'll use Monte-Carlo EM to find best-fit parameters. See
[_Convergence of a stochastic approximation version of the EM algorithm_][2]
for more details.

More precisely, we use HMC to form a chain conditioned on parameter `sigma`
and training data `{ (x[i], y[i]) : i=1...n }`. Then we use one gradient step
of maximum-likelihood to improve the `sigma` estimate. Then repeat the process
until convergence. (This procedure is a [Robbins--Monro algorithm](
https://en.wikipedia.org/wiki/Stochastic_approximation).)

The generative assumptions are:

```none
  W ~ MVN(loc=0, scale=sigma * eye(dims))
  for i=1...num_samples:
      X[i] ~ MVN(loc=0, scale=eye(dims))
    eps[i] ~ Normal(loc=0, scale=1)
      Y[i] = X[i].T * W + eps[i]
```

We now implement a stochastic approximation of Expectation Maximization (SAEM)
using `tensorflow_probability` intrinsics. [Bernard (1999)][2]

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tf.enable_eager_execution()

tfd = tfp.distributions

def make_training_data(num_samples, dims, sigma):
  dt = np.asarray(sigma).dtype
  x = np.random.randn(dims, num_samples).astype(dt)
  w = sigma * np.random.randn(1, dims).astype(dt)
  noise = np.random.randn(num_samples).astype(dt)
  y = w.dot(x) + noise
  return y[0], x, w[0]

def make_weights_prior(dims, log_sigma):
  return tfd.MultivariateNormalDiag(
      loc=tf.zeros([dims], dtype=log_sigma.dtype),
      scale_identity_multiplier=tf.exp(log_sigma))

def make_response_likelihood(w, x):
  if w.shape.ndims == 1:
    y_bar = tf.matmul(w[tf.newaxis], x)[0]
  else:
    y_bar = tf.matmul(w, x)
  return tfd.Normal(loc=y_bar, scale=tf.ones_like(y_bar))  # [n]

# Setup assumptions.
dtype = np.float32
num_samples = 500
dims = 10
tf.compat.v1.random.set_random_seed(10014)
np.random.seed(10014)

weights_prior_true_scale = np.array(0.3, dtype)
y, x, _ = make_training_data(
    num_samples, dims, weights_prior_true_scale)

log_sigma = tf.compat.v2.Variable(
    name='log_sigma', initial_value=np.array(0, dtype))

optimizer = tf.compat.v2.optimizers.SGD(learning_rate=0.01)

@tf.function
def mcem_iter(weights_chain_start, step_size):
  with tf.GradientTape() as tape:
    tape.watch(log_sigma)
    prior = make_weights_prior(dims, log_sigma)

    def unnormalized_posterior_log_prob(w):
      likelihood = make_response_likelihood(w, x)
      return (
          prior.log_prob(w) +
          tf.reduce_sum(
              input_tensor=likelihood.log_prob(y), axis=-1))  # [m]

    def trace_fn(_, pkr):
      return (
          pkr.inner_results.log_accept_ratio,
          pkr.inner_results.accepted_results.target_log_prob,
          pkr.inner_results.accepted_results.step_size)

    num_results = 2
    weights, (
        log_accept_ratio, target_log_prob, step_size) = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=0,
        current_state=weights_chain_start,
        kernel=tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=unnormalized_posterior_log_prob,
                num_leapfrog_steps=2,
                step_size=step_size,
                state_gradients_are_stopped=True,
            ),
            # Adapt for the entirety of the trajectory.
            num_adaptation_steps=2),
        trace_fn=trace_fn,
        parallel_iterations=1)

    # We do an optimization step to propagate `log_sigma` after two HMC
    # steps to propagate `weights`.
    loss = -tf.reduce_mean(input_tensor=target_log_prob)

  avg_acceptance_ratio = tf.reduce_mean(
      input_tensor=tf.exp(tf.minimum(log_accept_ratio, 0.)))

  optimizer.apply_gradients(
      [[tape.gradient(loss, log_sigma), log_sigma]])

  weights_prior_estimated_scale = tf.exp(log_sigma)
  return (weights_prior_estimated_scale, weights[-1], loss,
          step_size[-1], avg_acceptance_ratio)

num_iters = int(40)

weights_prior_estimated_scale_ = np.zeros(num_iters, dtype)
weights_ = np.zeros([num_iters + 1, dims], dtype)
loss_ = np.zeros([num_iters], dtype)
weights_[0] = np.random.randn(dims).astype(dtype)
step_size_ = 0.03

for iter_ in range(num_iters):
  [
      weights_prior_estimated_scale_[iter_],
      weights_[iter_ + 1],
      loss_[iter_],
      step_size_,
      avg_acceptance_ratio_,
  ] = mcem_iter(weights_[iter_], step_size_)
  tf.compat.v1.logging.vlog(
      1, ('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
          'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
              iter_, loss_[iter_], weights_prior_estimated_scale_[iter_],
              step_size_, avg_acceptance_ratio_))

# Should converge to ~0.22.
import matplotlib.pyplot as plt
plt.plot(weights_prior_estimated_scale_)
plt.ylabel('weights_prior_estimated_scale')
plt.xlabel('iteration')
```

#### References

[1]: Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain
     Monte Carlo_, 2011. https://arxiv.org/abs/1206.1901

[2]: Bernard Delyon, Marc Lavielle, Eric, Moulines. _Convergence of a
     stochastic approximation version of the EM algorithm_, Ann. Statist. 27
     (1999), no. 1, 94--128. https://projecteuclid.org/euclid.aos/1018031103

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Initializes this transition kernel. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(step_size_update_fn)`. They will be removed after 2019-05-22.
Instructions for updating:
The `step_size_update_fn` argument is deprecated. Use <a href="../../tfp/mcmc/SimpleStepSizeAdaptation.md"><code>tfp.mcmc.SimpleStepSizeAdaptation</code></a> instead.

#### Args:


* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
  `current_state` (or `*current_state` if it's a list) and returns its
  (possibly unnormalized) log-density under the target distribution.
* <b>`step_size`</b>: `Tensor` or Python `list` of `Tensor`s representing the step
  size for the leapfrog integrator. Must broadcast with the shape of
  `current_state`. Larger step sizes lead to faster progress, but
  too-large step sizes make rejection exponentially more likely. When
  possible, it's often helpful to match per-variable step sizes to the
  standard deviations of the target distribution in each variable.
* <b>`num_leapfrog_steps`</b>: Integer number of steps to run the leapfrog integrator
  for. Total progress per HMC step is roughly proportional to
  `step_size * num_leapfrog_steps`.
* <b>`state_gradients_are_stopped`</b>: Python `bool` indicating that the proposed
  new state be run through `tf.stop_gradient`. This is particularly useful
  when combining optimization over samples from the HMC chain.
  Default value: `False` (i.e., do not apply `stop_gradient`).
* <b>`step_size_update_fn`</b>: Python `callable` taking current `step_size`
  (typically a `tf.Variable`) and `kernel_results` (typically
  `collections.namedtuple`) and returns updated step_size (`Tensor`s).
  Default value: `None` (i.e., do not update `step_size` automatically).
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`store_parameters_in_results`</b>: If `True`, then `step_size` and
  `num_leapfrog_steps` are written to and read from eponymous fields in
  the kernel results objects returned from `one_step` and
  `bootstrap_results`. This allows wrapper kernels to adjust those
  parameters on the fly. This is incompatible with `step_size_update_fn`,
  which must be set to `None`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'hmc_kernel').



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="name"><code>name</code></h3>




<h3 id="num_leapfrog_steps"><code>num_leapfrog_steps</code></h3>

Returns the num_leapfrog_steps parameter.

If `store_parameters_in_results` argument to the initializer was set to
`True`, this only returns the value of the `num_leapfrog_steps` placed in
the kernel results by the `bootstrap_results` method. The actual
`num_leapfrog_steps` in that situation is governed by the
`previous_kernel_results` argument to `one_step` method.

#### Returns:


* <b>`num_leapfrog_steps`</b>: An integer `Tensor`.

<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.


<h3 id="seed"><code>seed</code></h3>




<h3 id="state_gradients_are_stopped"><code>state_gradients_are_stopped</code></h3>




<h3 id="step_size"><code>step_size</code></h3>

Returns the step_size parameter.

If `store_parameters_in_results` argument to the initializer was set to
`True`, this only returns the value of the `step_size` placed in the kernel
results by the `bootstrap_results` method. The actual step size in that
situation is governed by the `previous_kernel_results` argument to
`one_step` method.

#### Returns:


* <b>`step_size`</b>: A floating point `Tensor` or a list of such `Tensors`.

<h3 id="step_size_update_fn"><code>step_size_update_fn</code></h3>




<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>






## Methods

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(init_state)
```

Creates initial `previous_kernel_results` using a supplied `state`.


<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Runs one iteration of Hamiltonian Monte Carlo.


#### Args:


* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s). The first `r` dimensions index
  independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
* <b>`previous_kernel_results`</b>: `collections.namedtuple` containing `Tensor`s
  representing values from previous calls to this function (or from the
  `bootstrap_results` function.)


#### Returns:


* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
  of the Markov chain(s) after taking exactly one step. Has same type and
  shape as `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.


#### Raises:


* <b>`ValueError`</b>: if there isn't one `step_size` or a list with same length as
  `current_state`.



