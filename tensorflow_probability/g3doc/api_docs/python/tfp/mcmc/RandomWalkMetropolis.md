<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.RandomWalkMetropolis" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="new_state_fn"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.RandomWalkMetropolis

## Class `RandomWalkMetropolis`

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)

Runs one step of the RWM algorithm with symmetric proposal.

Random Walk Metropolis is a gradient-free Markov chain Monte Carlo
(MCMC) algorithm. The algorithm involves a proposal generating step
`proposal_state = current_state + perturb` by a random
perturbation, followed by Metropolis-Hastings accept/reject step. For more
details see [Section 2.1 of Roberts and Rosenthal (2004)](
http://emis.ams.org/journals/PS/images/getdoc510c.pdf?id=35&article=15&mode=pdf).

Current class implements RWM for normal and uniform proposals. Alternatively,
the user can supply any custom proposal generating function.

The function `one_step` can update multiple chains in parallel. It assumes
that all leftmost dimensions of `current_state` index independent chain states
(and are therefore updated independently). The output of
`target_log_prob_fn(*current_state)` should sum log-probabilities across all
event dimensions. Slices along the rightmost dimensions may have different
target distributions; for example, `current_state[0, :]` could have a
different target distribution from `current_state[1, :]`. These semantics
are governed by `target_log_prob_fn(*current_state)`. (The number of
independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

#### Examples:

##### Sampling from the Standard Normal Distribution.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dtype = np.float32

target = tfd.Normal(loc=dtype(0), scale=dtype(1))

samples, _ = tfp.mcmc.sample_chain(
  num_results=1000,
  current_state=dtype(1),
  kernel=tfp.mcmc.RandomWalkMetropolis(
     target.log_prob,
     seed=42),
  num_burnin_steps=500,
  parallel_iterations=1)  # For determinism.

sample_mean = tf.math.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.math.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))
with tf.Session() as sess:
  [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

print('Estimated mean: {}'.format(sample_mean_))
print('Estimated standard deviation: {}'.format(sample_std_))
```

##### Sampling from a 2-D Normal Distribution.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

dtype = np.float32
true_mean = dtype([0, 0])
true_cov = dtype([[1, 0.5],
                 [0.5, 1]])
num_results = 500
num_chains = 100

# Target distribution is defined through the Cholesky decomposition `L`:
L = tf.linalg.cholesky(true_cov)
target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=L)

# Assume that the state is passed as a list of 1-d tensors `x` and `y`.
# Then the target log-density is defined as follows:
def target_log_prob(x, y):
  # Stack the input tensors together
  z = tf.stack([x, y], axis=-1)
  return target.log_prob(tf.squeeze(z))

# Initial state of the chain
init_state = [np.ones([num_chains, 1], dtype=dtype),
              np.ones([num_chains, 1], dtype=dtype)]

# Run Random Walk Metropolis with normal proposal for `num_results`
# iterations for `num_chains` independent chains:
samples, _ = tfp.mcmc.sample_chain(
    num_results=num_results,
    current_state=init_state,
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=target_log_prob,
        seed=54),
    num_burnin_steps=200,
    num_steps_between_results=1,  # Thinning.
    parallel_iterations=1)
samples = tf.stack(samples, axis=-1)

sample_mean = tf.math.reduce_mean(samples, axis=0)
x = tf.squeeze(samples - sample_mean)
sample_cov = tf.matmul(tf.transpose(x, [1, 2, 0]),
                       tf.transpose(x, [1, 0, 2])) / num_results

mean_sample_mean = tf.math.reduce_mean(sample_mean)
mean_sample_cov = tf.math.reduce_mean(sample_cov, axis=0)
x = tf.reshape(sample_cov - mean_sample_cov, [num_chains, 2 * 2])
cov_sample_cov = tf.reshape(tf.matmul(x, x, transpose_a=True) / num_chains,
                            shape=[2 * 2, 2 * 2])

with tf.Session() as sess:
  [
    mean_sample_mean_,
    mean_sample_cov_,
    cov_sample_cov_,
  ] = sess.run([
    mean_sample_mean,
    mean_sample_cov,
    cov_sample_cov,
  ])

print('Estimated mean: {}'.format(mean_sample_mean_))
print('Estimated avg covariance: {}'.format(mean_sample_cov_))
print('Estimated covariance of covariance: {}'.format(cov_sample_cov_))
```

##### Sampling from the Standard Normal Distribution using Cauchy proposal.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

dtype = np.float32
num_burnin_steps = 500
num_chain_results = 1000

def cauchy_new_state_fn(scale, dtype):
  cauchy = tfd.Cauchy(loc=dtype(0), scale=dtype(scale))
  def _fn(state_parts, seed):
    next_state_parts = []
    seed_stream  = tfd.SeedStream(seed, salt='RandomCauchy')
    for sp in state_parts:
      next_state_parts.append(sp + cauchy.sample(
        sample_shape=sp.shape, seed=seed_stream()))
    return next_state_parts
  return _fn

target = tfd.Normal(loc=dtype(0), scale=dtype(1))

samples, _ = tfp.mcmc.sample_chain(
    num_results=num_chain_results,
    num_burnin_steps=num_burnin_steps,
    current_state=dtype(1),
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target.log_prob,
        new_state_fn=cauchy_new_state_fn(scale=0.5, dtype=dtype),
        seed=42),
    parallel_iterations=1)  # For determinism.

sample_mean = tf.math.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.math.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))
with tf.Session() as sess:
  [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

print('Estimated mean: {}'.format(sample_mean_))
print('Estimated standard deviation: {}'.format(sample_std_))
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    target_log_prob_fn,
    new_state_fn=None,
    seed=None,
    name=None
)
```

Initializes this transition kernel.

#### Args:

* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
    `current_state` (or `*current_state` if it's a list) and returns its
    (possibly unnormalized) log-density under the target distribution.
* <b>`new_state_fn`</b>: Python callable which takes a list of state parts and a
    seed; returns a same-type `list` of `Tensor`s, each being a perturbation
    of the input state parts. The perturbation distribution is assumed to be
    a symmetric distribution centered at the input state part.
    Default value: `None` which is mapped to
      `tfp.mcmc.random_walk_normal_fn()`.
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., 'rwm_kernel').


#### Returns:

* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
    of the Markov chain(s) at each result step. Has same shape as
    `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
    advance the chain.


#### Raises:

* <b>`ValueError`</b>: if there isn't one `scale` or a list with same length as
    `current_state`.



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="new_state_fn"><code>new_state_fn</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



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

Runs one iteration of Random Walk Metropolis with normal proposal.

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

* <b>`ValueError`</b>: if there isn't one `scale` or a list with same length as
    `current_state`.



