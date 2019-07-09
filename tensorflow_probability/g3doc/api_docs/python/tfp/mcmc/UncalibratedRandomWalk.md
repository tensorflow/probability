<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.UncalibratedRandomWalk" />
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

# tfp.mcmc.UncalibratedRandomWalk

## Class `UncalibratedRandomWalk`

Generate proposal for the Random Walk Metropolis algorithm.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/random_walk_metropolis.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/random_walk_metropolis.py).

<!-- Placeholder for "Used in" -->

Warning: this kernel will not result in a chain which converges to the
`target_log_prob`. To get a convergent MCMC, use
`tfp.mcmc.RandomWalkMetropolisNormal(...)` or
`tfp.mcmc.MetropolisHastings(tfp.mcmc.UncalibratedRandomWalk(...))`.

For more details on `UncalibratedRandomWalk`, see
`RandomWalkMetropolis`.

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

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

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



