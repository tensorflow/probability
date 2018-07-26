Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.ReplicaExchangeMC" />
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

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)

Runs one step of the Replica Exchange Monte Carlo.

[Replica Exchange Monte Carlo](
https://en.wikipedia.org/wiki/Parallel_tempering) is a Markov chain
Monte Carlo (MCMC) algorithm that is also known as Parallel Tempering. This
algorithm performs multiple sampling with different temperatures in parallel,
and exchange those samplings according to the Metropolis-Hastings criterion.
By using the sampling result of high temperature, sampling with less influence
of the local solution becomes possible.

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
    inverse_temperatures=10.**tf.linspace(0., -2., 5),
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
    inverse_temperatures=10.**tf.linspace(0., -2., 5),
    make_kernel_fn=make_kernel_fn,
    seed=42)

samples, _ = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=np.zeros(2, dtype=dtype),
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

## Properties

<h3 id="exchange_proposed_fn"><code>exchange_proposed_fn</code></h3>



<h3 id="inverse_temperatures"><code>inverse_temperatures</code></h3>



<h3 id="is_calibrated"><code>is_calibrated</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="num_replica"><code>num_replica</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>





## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    target_log_prob_fn,
    inverse_temperatures,
    make_kernel_fn,
    exchange_proposed_fn=default_exchange_proposed_fn(1.0),
    seed=None,
    name=None,
    **kwargs
)
```

Instantiates this object.

#### Args:

* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
    `current_state` (or `*current_state` if it's a list) and returns its
    (possibly unnormalized) log-density under the target distribution.
* <b>`inverse_temperatures`</b>: sequence of inverse temperatures to perform
    samplings with each replica. Must have statically known `rank` and
    statically known leading shape, i.e.,
    `inverse_temperatures.shape[0].value is not None`
* <b>`make_kernel_fn`</b>: Python callable which takes target_log_prob_fn and seed
    args and returns a TransitionKernel instance.
* <b>`exchange_proposed_fn`</b>: Python callable which take a number of replicas, and
    return combinations of replicas for exchange and a number of
    combinations.
* <b>`seed`</b>: Python integer to seed the random number generator.
    Default value: `None` (i.e., no seed).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., "remc_kernel").
* <b>`**kwargs`</b>: Arguments for `make_kernel_fn`.


#### Raises:

* <b>`ValueError`</b>: if `inverse_temperatures` doesn't have statically known rank
    and statically known leading shape

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(init_state)
```

Returns an object with the same type as returned by `one_step`.

#### Args:

* <b>`init_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
    a state(s) of the Markov chain(s).


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



