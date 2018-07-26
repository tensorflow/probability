Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.HamiltonianMonteCarlo" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_leapfrog_steps"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="step_size"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.HamiltonianMonteCarlo

## Class `HamiltonianMonteCarlo`

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)

Runs one step of Hamiltonian Monte Carlo.

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
import numpy as np

tfd = tf.contrib.distributions

# Tuning acceptance rates:
dtype = np.float32
num_warmup_iter = 500
num_chain_iter = 500
# Set the target average acceptance ratio for the HMC as suggested by
# Beskos et al. (2013):
# https://projecteuclid.org/download/pdfview_1/euclid.bj/1383661192

target_accept_rate = 0.651

x = tf.get_variable(name='x', initializer=dtype(1))
step_size = tf.get_variable(name='step_size', initializer=dtype(1))

# Target distribution is standard univariate Normal.
target = tfd.Normal(loc=dtype(0), scale=dtype(1))

# Initialize the HMC sampler. In order to retain `tfe` compatibility,
# `target_log_prob_fn` is passed as `lambda x: target.log_prob(x)`.
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=lambda x: target.log_prob(x),
    step_size=step_size,
    num_leapfrog_steps=3)

# One iteration of the HMC
next_x, other_results = hmc.one_step(
    current_state=x,
    previous_kernel_results=hmc.bootstrap_results(x))

x_update = x.assign(next_x)

# Adapt the step size using standard adaptive MCMC procedure. See Section 4.2
# of Andrieu and Thoms (2008):
# http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf

step_size_update = step_size.assign_add(
    step_size * tf.where(
        tf.exp(tf.minimum(other_results.log_accept_ratio, 0.)) >
            target_accept_rate,
        0.01, -0.01))

# Note, the adaptations are performed during warmup only.
warmup = tf.group([x_update, step_size_update])

init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  # Warm up the sampler and adapt the step size
  for _ in xrange(num_warmup_iter):
    sess.run(warmup)
  # Collect samples without adapting step size
  samples = np.zeros([num_chain_iter])
  for i in xrange(num_chain_iter):
    _, x_,= sess.run([x_update, x])
    samples[i] = x_

print(samples.mean(), samples.std())
```

##### Estimate parameters of a more complicated posterior.

In this example, we'll use Monte-Carlo EM to find best-fit parameters. See
["Implementations of the Monte Carlo EM algorithm " by Levine and Casella](
https://ecommons.cornell.edu/bitstream/handle/1813/32030/BU-1431-M.pdf?sequence=1).

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

We now implement MCEM using `tensorflow_probability` intrinsics.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tf.contrib.distributions

def make_training_data(num_samples, dims, sigma):
  dt = np.asarray(sigma).dtype
  zeros = tf.zeros(dims, dtype=dt)
  x = tfd.MultivariateNormalDiag(
      loc=zeros).sample(num_samples, seed=1)
  w = tfd.MultivariateNormalDiag(
      loc=zeros,
      scale_identity_multiplier=sigma).sample(seed=2)
  noise = tfd.Normal(
      loc=dt.type(0),
      scale=dt.type(1)).sample(num_samples, seed=3)
  y = tf.tensordot(x, w, axes=[[1], [0]]) + noise
  return y, x, w

def make_prior(sigma, dims):
  # p(w | sigma)
  return tfd.MultivariateNormalDiag(
      loc=tf.zeros([dims], dtype=sigma.dtype),
      scale_identity_multiplier=sigma)

def make_likelihood(x, w):
  # p(y | x, w)
  return tfd.MultivariateNormalDiag(
      loc=tf.tensordot(x, w, axes=[[1], [0]]))

# Setup assumptions.
dtype = np.float32
num_samples = 150
dims = 10
num_iters = int(5e3)

true_sigma = dtype(0.3)
y, x, true_weights = make_training_data(num_samples, dims, true_sigma)

# Estimate of `log(true_sigma)`.
log_sigma = tf.get_variable(name='log_sigma', initializer=dtype(0))
sigma = tf.exp(log_sigma)

# State of the Markov chain.
# We set `trainable=False` so it is unaffected by the M-step.
weights = tf.get_variable(
    name='weights',
    initializer=np.random.randn(dims).astype(dtype),
    trainable=False)

prior = make_prior(sigma, dims)

def joint_log_prob(w):
  # f(w) = log p(w, y | x)
  return prior.log_prob(w) + make_likelihood(x, w).log_prob(y)

# Initialize the HMC sampler.
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=joint_log_prob,
    step_size=0.1,
    num_leapfrog_steps=5)

weights_update = weights.assign(
    hmc.one_step(weights, hmc.bootstrap_results(weights))[0])

# We do an optimization step to propagate `log_sigma` after one HMC step to
# propagate `weights`. The loss function for the optimization algorithm is
# exactly the prior distribution since the likelihood does not depend on
# `log_sigma`.
with tf.control_dependencies([weights_update]):
  loss = -prior.log_prob(weights)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
log_sigma_update = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sigma_history = np.zeros(num_iters, dtype)
weights_history = np.zeros([num_iters, dims], dtype)

with tf.Session() as sess:
  sess.run(init)
  for i in xrange(num_iters):
    _, sigma_, weights_ = sess.run([log_sigma_update, sigma, weights])
    weights_history[i, :] = weights_
    sigma_history[i] = sigma_
  true_weights_ = sess.run(true_weights)

# Should oscillate around true_sigma.
import matplotlib.pyplot as plt
plt.plot(sigma_history)
plt.ylabel('sigma')
plt.xlabel('iteration')

# Mean error should be close to zero
print('mean error:', np.abs(np.mean(sigma_history) - true_sigma))
```

#### References

[1]: Radford Neal. MCMC Using Hamiltonian Dynamics. _Handbook of Markov Chain
     Monte Carlo_, 2011. https://arxiv.org/abs/1206.1901

## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="num_leapfrog_steps"><code>num_leapfrog_steps</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



<h3 id="step_size"><code>step_size</code></h3>



<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>





## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    num_leapfrog_steps,
    seed=None,
    name=None
)
```

Initializes this transition kernel.

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
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., 'hmc_kernel').


#### Returns:

* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
    of the Markov chain(s) at each result step. Has same shape as
    `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
    advance the chain.


#### Raises:

* <b>`ValueError`</b>: if there isn't one `step_size` or a list with same length as
    `current_state`.

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



