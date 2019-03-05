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

# Target distribution is proportional to: `exp(-x (1 + x))`.
def unnormalized_log_prob(x):
  return -x - x**2.

# Create state to hold updated `step_size`.
step_size = tf.get_variable(
    name='step_size',
    initializer=1.,
    use_resource=True,  # For TFE compatibility.
    trainable=False)

# Initialize the HMC transition kernel.
num_results = int(10e3)
num_burnin_steps = int(1e3)
hmc = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_log_prob,
    num_leapfrog_steps=3,
    step_size=step_size,
    step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
      num_adaptation_steps=int(num_burnin_steps * 0.8)))

# Run the chain (with burn-in).
samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=1.,
    kernel=hmc)

# Initialize all constructed variables.
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  init_op.run()
  samples_, kernel_results_ = sess.run([samples, kernel_results])

print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
    samples_.mean(), samples_.std(), kernel_results_.is_accepted.mean()))
# mean:-0.5003  stddev:0.7711  acceptance:0.6240
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

tfd = tfp.distributions

def make_training_data(num_samples, dims, sigma):
  dt = np.asarray(sigma).dtype
  zeros = tf.zeros(dims, dtype=dt)
  x = tf.transpose(tfd.MultivariateNormalDiag(loc=zeros).sample(
      num_samples, seed=1))  # [d, n]
  w = tfd.MultivariateNormalDiag(
      loc=zeros,
      scale_identity_multiplier=sigma).sample([1], seed=2)  # [1, d]
  noise = tfd.Normal(loc=np.array(0, dt), scale=np.array(1, dt)).sample(
      num_samples, seed=3)  # [n]
  y = tf.matmul(w, x) + noise  # [1, n]
  return y[0], x, w[0]

def make_weights_prior(dims, dtype):
  return tfd.MultivariateNormalDiag(
      loc=tf.zeros([dims], dtype=dtype),
      scale_identity_multiplier=tf.exp(tf.get_variable(
          name='log_sigma',
          initializer=np.array(0, dtype),
          use_resource=True)))

def make_response_likelihood(w, x):
  w_shape = tf.pad(
      tf.shape(w),
      paddings=[[tf.where(tf.rank(w) > 1, 0, 1), 0]],
      constant_values=1)
  y_shape = tf.concat([tf.shape(w)[:-1], [tf.shape(x)[-1]]], axis=0)
  w_expand = tf.reshape(w, w_shape)
  return tfd.Normal(
      loc=tf.reshape(tf.matmul(w_expand, x), y_shape),
      scale=np.array(1, w.dtype.as_numpy_dtype))  # [n]

# Setup assumptions.
dtype = np.float32
num_samples = 500
dims = 10

weights_prior_true_scale = np.array(0.3, dtype)
with tf.Session() as sess:
  y, x, true_weights = sess.run(
      make_training_data(num_samples, dims, weights_prior_true_scale))

prior = make_weights_prior(dims, dtype)
def unnormalized_posterior_log_prob(w):
  likelihood = make_response_likelihood(w, x)
  return (prior.log_prob(w)
          + tf.reduce_sum(likelihood.log_prob(y), axis=-1))  # [m]

weights_chain_start = tf.placeholder(dtype, shape=[dims])

step_size = tf.get_variable(
    name='step_size',
    initializer=np.array(0.05, dtype),
    use_resource=True,
    trainable=False)

num_results = 2
weights, kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=0,
    current_state=weights_chain_start,
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        num_leapfrog_steps=2,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
          num_adaptation_steps=None),
        state_gradients_are_stopped=True))

avg_acceptance_ratio = tf.reduce_mean(
    tf.exp(tf.minimum(kernel_results.log_accept_ratio, 0.)))

# We do an optimization step to propagate `log_sigma` after two HMC steps to
# propagate `weights`.
loss = -tf.reduce_mean(kernel_results.accepted_results.target_log_prob)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

with tf.variable_scope(tf.get_variable_scope(), reuse=True):
  weights_prior_estimated_scale = tf.exp(
      tf.get_variable(name='log_sigma', dtype=dtype))

init_op = tf.global_variables_initializer()

num_iters = int(40)

weights_prior_estimated_scale_ = np.zeros(num_iters, dtype)
weights_ = np.zeros([num_iters + 1, dims], dtype)
weights_[0] = np.random.randn(dims).astype(dtype)

with tf.Session() as sess:
  init_op.run()
  for iter_ in range(num_iters):
    [
        _,
        weights_prior_estimated_scale_[iter_],
        weights_[iter_ + 1],
        loss_,
        step_size_,
        avg_acceptance_ratio_,
    ] = sess.run([
        train_op,
        weights_prior_estimated_scale,
        weights[-1],
        loss,
        step_size,
        avg_acceptance_ratio,
    ], feed_dict={weights_chain_start: weights_[iter_]})
    print('iter:{:>2}  loss:{: 9.3f}  scale:{:.3f}  '
          'step_size:{:.4f}  avg_acceptance_ratio:{:.4f}').format(
              iter_, loss_, weights_prior_estimated_scale_[iter_],
              step_size_, avg_acceptance_ratio_))

# Should converge to ~0.24.
import matplotlib.pyplot as plt
plot.plot(weights_prior_estimated_scale_)
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
    target_log_prob_fn,
    step_size,
    num_leapfrog_steps,
    state_gradients_are_stopped=False,
    step_size_update_fn=None,
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
* <b>`state_gradients_are_stopped`</b>: Python `bool` indicating that the proposed
    new state be run through `tf.stop_gradient`. This is particularly useful
    when combining optimization over samples from the HMC chain.
    Default value: `False` (i.e., do not apply `stop_gradient`).
* <b>`step_size_update_fn`</b>: Python `callable` taking current `step_size`
    (typically a `tf.Variable`) and `kernel_results` (typically
    `collections.namedtuple`) and returns updated step_size (`Tensor`s).
    Default value: `None` (i.e., do not update `step_size` automatically).
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., 'hmc_kernel').



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="num_leapfrog_steps"><code>num_leapfrog_steps</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



<h3 id="state_gradients_are_stopped"><code>state_gradients_are_stopped</code></h3>



<h3 id="step_size"><code>step_size</code></h3>



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



