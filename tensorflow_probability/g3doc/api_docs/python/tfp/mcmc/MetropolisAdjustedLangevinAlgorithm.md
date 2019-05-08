<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.MetropolisAdjustedLangevinAlgorithm" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parallel_iterations"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="step_size"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="volatility_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.MetropolisAdjustedLangevinAlgorithm

## Class `MetropolisAdjustedLangevinAlgorithm`

Runs one step of Metropolis-adjusted Langevin algorithm.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/langevin.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/langevin.py).

<!-- Placeholder for "Used in" -->

Metropolis-adjusted Langevin algorithm (MALA) is a Markov chain Monte Carlo
(MCMC) algorithm that takes a step of a discretised Langevin diffusion as a
proposal. This class implements one step of MALA using Euler-Maruyama method
for a given `current_state` and diagonal preconditioning `volatility` matrix.
Mathematical details and derivations can be found in
[Roberts and Rosenthal (1998)][1] and [Xifara et al. (2013)][2].

See `UncalibratedLangevin` class description below for details on the proposal
generating step of the algorithm.

The `one_step` function can update multiple chains in parallel. It assumes
that all leftmost dimensions of `current_state` index independent chain states
(and are therefore updated independently). The output of
`target_log_prob_fn(*current_state)` should reduce log-probabilities across
all event dimensions. Slices along the rightmost dimensions may have different
target distributions; for example, `current_state[0, :]` could have a
different target distribution from `current_state[1, :]`. These semantics are
governed by `target_log_prob_fn(*current_state)`. (The number of independent
chains is `tf.size(target_log_prob_fn(*current_state))`.)

#### Examples:

##### Simple chain with warm-up.

In this example we sample from a standard univariate normal
distribution using MALA with `step_size` equal to 0.75.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

dtype = np.float32

with tf.Session(graph=tf.Graph()) as sess:
  # Target distribution is Standard Univariate Normal
  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  # Define MALA sampler with `step_size` equal to 0.75
  samples, _ = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=dtype(1),
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=target.log_prob,
          step_size=0.75,
          seed=42),
      num_burnin_steps=500,
      parallel_iterations=1)  # For determinism.

  sample_mean = tf.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
      tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                     axis=0))

  sess.graph.finalize()  # No more graph building.

  [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

print('sample mean', sample_mean_)
print('sample standard deviation', sample_std_)
```

##### Same example but in eager mode.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# Support for eager execution
tf.enable_eager_execution()

tfd = tfp.distributions
dtype = np.float32

# Target distribution is Standard Univariate Normal
target = tfd.Normal(loc=dtype(0), scale=dtype(1))

def target_log_prob(x):
  return target.log_prob(x)

# Define MALA sampler with `step_size` equal to 0.75
samples, _ = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=dtype(1),
    kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
        target_log_prob_fn=target_log_prob,
        step_size=0.75,
        seed=42),
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.

sample_mean = tf.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))

print('sample mean', sample_mean)
print('sample standard deviation', sample_std)

plt.title('Traceplot')
plt.plot(samples.numpy(), 'b')
plt.xlabel('Iteration')
plt.ylabel('Position')
plt.show()
```

##### Sample from a 3-D Multivariate Normal distribution.

In this example we also consider a non-constant volatility function.

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions

dtype = np.float32
true_mean = dtype([0, 0, 0])
true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
num_results = 500
num_chains = 500

with tf.Session(graph=tf.Graph()) as sess:
  # Target distribution is defined through the Cholesky decomposition
  chol = tf.linalg.cholesky(true_cov)
  target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

  # Assume that the state is passed as a list of tensors `x` and `y`.
  # Then the target log-density is defined as follows:
  def target_log_prob(x, y):
    # Stack the input tensors together
    z = tf.concat([x, y], axis=-1) - true_mean
    return target.log_prob(z)

  # Here we define the volatility function to be non-constant
  def volatility_fn(x, y):
    # Stack the input tensors together
    return [1. / (0.5 + 0.1 * tf.sqrt(x * x)),
            1. / (0.5 + 0.1 *tf.sqrt(y * y))]

  # Initial state of the chain
  init_state = [np.ones([num_chains, 2], dtype=dtype),
                np.ones([num_chains, 1], dtype=dtype)]

  # Run MALA with normal proposal for `num_results` iterations for
  # `num_chains` independent chains:
  states, _ = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
          target_log_prob_fn=target_log_prob,
          step_size=.1,
          volatility_fn=volatility_fn,
          seed=42),
      num_burnin_steps=200,
      num_steps_between_results=1,
      parallel_iterations=1)

  states = tf.concat(states, axis=-1)
  sample_mean = tf.reduce_mean(states, axis=[0, 1])
  x = tf.expand_dims(states - sample_mean, -1)
  sample_cov = tf.reduce_mean(
      tf.matmul(x, tf.transpose(x, [0, 1, 3, 2])), [0, 1])

  [sample_mean_, sample_cov_] = sess.run([
      sample_mean, sample_cov])

print('sample mean', sample_mean_)
print('sample covariance matrix', sample_cov_)
```

#### References

[1]: Gareth Roberts and Jeffrey Rosenthal. Optimal Scaling of Discrete
     Approximations to Langevin Diffusions. _Journal of the Royal Statistical
     Society: Series B (Statistical Methodology)_, 60: 255-268, 1998.
     https://doi.org/10.1111/1467-9868.00123

[2]: T. Xifara et al. Langevin diffusions and the Metropolis-adjusted
     Langevin algorithm. _arXiv preprint arXiv:1309.2983_, 2013.
     https://arxiv.org/abs/1309.2983

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    volatility_fn=None,
    seed=None,
    parallel_iterations=10,
    name=None
)
```

Initializes MALA transition kernel.

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
* <b>`volatility_fn`</b>: Python callable which takes an argument like
  `current_state` (or `*current_state` if it's a list) and returns
  volatility value at `current_state`. Should return a `Tensor` or Python
  `list` of `Tensor`s that must broadcast with the shape of
  `current_state` Defaults to the identity function.
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`parallel_iterations`</b>: the number of coordinates for which the gradients of
  the volatility matrix `volatility_fn` can be computed in parallel.
  Default value: `None` (i.e., no seed).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'mala_kernel').


#### Returns:

* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
  of the Markov chain(s) at each result step. Has same shape as
  `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.


#### Raises:

* <b>`ValueError`</b>: if there isn't one `step_size` or a list with same length as
  `current_state`.
* <b>`TypeError`</b>: if `volatility_fn` is not callable.



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="name"><code>name</code></h3>



<h3 id="parallel_iterations"><code>parallel_iterations</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



<h3 id="step_size"><code>step_size</code></h3>



<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>



<h3 id="volatility_fn"><code>volatility_fn</code></h3>





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

Runs one iteration of MALA.

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
  `current_state` or `diffusion_drift`.



