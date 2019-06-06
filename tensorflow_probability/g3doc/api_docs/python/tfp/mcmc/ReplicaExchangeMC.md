<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.ReplicaExchangeMC" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="exchange_proposed_fn"/>
<meta itemprop="property" content="inverse_temperatures"/>
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_replica"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.ReplicaExchangeMC

## Class `ReplicaExchangeMC`

Runs one step of the Replica Exchange Monte Carlo.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/replica_exchange_mc.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/replica_exchange_mc.py).

<!-- Placeholder for "Used in" -->

[Replica Exchange Monte Carlo](
https://en.wikipedia.org/wiki/Parallel_tempering) is a Markov chain
Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering. This
algorithm performs multiple sampling with different temperatures in parallel,
and exchanges those samplings according to the Metropolis-Hastings criterion.

The `K` replicas are parameterized in terms of `inverse_temperature`'s,
`(beta[0], beta[1], ..., beta[K-1])`.  If the target distribution has
probability density `p(x)`, the `kth` replica has density `p(x)**beta_k`.

Typically `beta[0] = 1.0`, and `1.0 > beta[1] > beta[2] > ... > 0.0`.

* `beta[0] == 1` ==> First replicas samples from the target density, `p`.
* `beta[k] < 1`, for `k = 1, ..., K-1` ==> Other replicas sample from
  "flattened" versions of `p` (peak is less high, valley less low).  These
  distributions are somewhat closer to a uniform on the support of `p`.

Samples from adjacent replicas `i`, `i + 1` are used as proposals for each
other in a Metropolis step.  This allows the lower `beta` samples, which
explore less dense areas of `p`, to occasionally be used to help the
`beta == 1` chain explore new regions of the support.

Samples from replica 0 are returned, and the others are discarded.

#### Examples

##### Sampling from the Standard Normal Distribution.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

dtype = np.float32

target = tfd.Normal(loc=dtype(0), scale=dtype(1))

def make_kernel_fn(target_log_prob_fn, seed):
  return tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      seed=seed, step_size=1.0, num_leapfrog_steps=3)

remc = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=target.log_prob,
    inverse_temperatures=[1., 0.3, 0.1, 0.03],
    make_kernel_fn=make_kernel_fn,
    seed=42)

samples, _ = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=dtype(1),
    kernel=remc,
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.

sample_mean = tf.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))
with tf.Session() as sess:
  [sample_mean_, sample_std_] = sess.run([sample_mean, sample_std])

print('Estimated mean: {}'.format(sample_mean_))
print('Estimated standard deviation: {}'.format(sample_std_))
```

##### Sampling from a 2-D Mixture Normal Distribution.

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions

dtype = np.float32

target = tfd.MixtureSameFamily(
    mixture_distribution=tfd.Categorical(probs=[0.5, 0.5]),
    components_distribution=tfd.MultivariateNormalDiag(
        loc=[[-1., -1], [1., 1.]],
        scale_identity_multiplier=[0.1, 0.1]))

def make_kernel_fn(target_log_prob_fn, seed):
  return tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target_log_prob_fn,
      seed=seed, step_size=0.3, num_leapfrog_steps=3)

remc = tfp.mcmc.ReplicaExchangeMC(
    target_log_prob_fn=target.log_prob,
    inverse_temperatures=[1., 0.3, 0.1, 0.03, 0.01],
    make_kernel_fn=make_kernel_fn,
    seed=42)

samples, _ = tfp.mcmc.sample_chain(
    num_results=1000,
    # Start near the [1, 1] mode.  Standard HMC would get stuck there.
    current_state=np.ones(2, dtype=dtype),
    kernel=remc,
    num_burnin_steps=500,
    parallel_iterations=1)  # For determinism.

with tf.Session() as sess:
  samples_ = sess.run(samples)

plt.figure(figsize=(8, 8))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.plot(samples_[:, 0], samples_[:, 1], '.')
plt.show()
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    target_log_prob_fn,
    inverse_temperatures,
    make_kernel_fn,
    exchange_proposed_fn=default_exchange_proposed_fn(1.0),
    seed=None,
    name=None
)
```

Instantiates this object.


#### Args:


* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
  `current_state` (or `*current_state` if it's a list) and returns its
  (possibly unnormalized) log-density under the target distribution.
* <b>`inverse_temperatures`</b>: `1D` `Tensor of inverse temperatures to perform
  samplings with each replica. Must have statically known `shape`.
  `inverse_temperatures[0]` produces the states returned by samplers,
  and is typically == 1.
* <b>`make_kernel_fn`</b>: Python callable which takes target_log_prob_fn and seed
  args and returns a TransitionKernel instance.
* <b>`exchange_proposed_fn`</b>: Python callable which take a number of replicas, and
  return combinations of replicas for exchange.
* <b>`seed`</b>: Python integer to seed the random number generator.
  Default value: `None` (i.e., no seed).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., "remc_kernel").


#### Raises:


* <b>`ValueError`</b>: `inverse_temperatures` doesn't have statically known 1D shape.



## Properties

<h3 id="exchange_proposed_fn"><code>exchange_proposed_fn</code></h3>




<h3 id="inverse_temperatures"><code>inverse_temperatures</code></h3>




<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="name"><code>name</code></h3>




<h3 id="num_replica"><code>num_replica</code></h3>




<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.


<h3 id="seed"><code>seed</code></h3>




<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>






## Methods

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(init_state)
```

Returns an object with the same type as returned by `one_step`.


#### Args:


* <b>`init_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  initial state(s) of the Markov chain(s).


#### Returns:


* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
  `Tensor`s representing internal calculations made within this function.
  This inculdes replica states.

<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Takes one step of the TransitionKernel.


#### Args:


* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s).
* <b>`previous_kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or
  `list` of `Tensor`s representing internal calculations made within the
  previous call to this function (or as returned by `bootstrap_results`).


#### Returns:


* <b>`next_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  next state(s) of the Markov chain(s).
* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
  `Tensor`s representing internal calculations made within this function.
  This inculdes replica states.



