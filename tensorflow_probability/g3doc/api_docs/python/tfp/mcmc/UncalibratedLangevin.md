<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.UncalibratedLangevin" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="compute_acceptance"/>
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

# tfp.mcmc.UncalibratedLangevin

## Class `UncalibratedLangevin`

Runs one step of Uncalibrated Langevin discretized diffusion.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/langevin.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/langevin.py).

<!-- Placeholder for "Used in" -->

The class generates a Langevin proposal using `_euler_method` function and
also computes helper `UncalibratedLangevinKernelResults` for the next
iteration.

Warning: this kernel will not result in a chain which converges to the
`target_log_prob`. To get a convergent MCMC, use
`MetropolisAdjustedLangevinAlgorithm(...)` or
`MetropolisHastings(UncalibratedLangevin(...))`.

For more details on `UncalibratedLangevin`, see
`MetropolisAdjustedLangevinAlgorithm`.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    volatility_fn=None,
    parallel_iterations=10,
    compute_acceptance=True,
    seed=None,
    name=None
)
```

Initializes Langevin diffusion transition kernel.


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
* <b>`parallel_iterations`</b>: the number of coordinates for which the gradients of
  the volatility matrix `volatility_fn` can be computed in parallel.
* <b>`compute_acceptance`</b>: Python 'bool' indicating whether to compute the
  Metropolis log-acceptance ratio used to construct
  `MetropolisAdjustedLangevinAlgorithm` kernel.
* <b>`seed`</b>: Python integer to seed the random number generator.
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

<h3 id="compute_acceptance"><code>compute_acceptance</code></h3>




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



