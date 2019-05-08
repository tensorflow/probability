<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.MetropolisHastings" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="inner_kernel"/>
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.MetropolisHastings

## Class `MetropolisHastings`

Runs one step of the Metropolis-Hastings algorithm.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/metropolis_hastings.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/metropolis_hastings.py).

<!-- Placeholder for "Used in" -->

The [Metropolis-Hastings algorithm](
https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) is a
Markov chain Monte Carlo (MCMC) technique which uses a proposal distribution
to eventually sample from a target distribution.

Note: `inner_kernel.one_step` must return `kernel_results` as a
`collections.namedtuple` which must:

- have a `target_log_prob` field,
- optionally have a `log_acceptance_correction` field, and,
- have only fields which are `Tensor`-valued.

The Metropolis-Hastings log acceptance-probability is computed as:

```python
log_accept_ratio = (current_kernel_results.target_log_prob
                    - previous_kernel_results.target_log_prob
                    + current_kernel_results.log_acceptance_correction)
```

If `current_kernel_results.log_acceptance_correction` does not exist, it is
presumed `0.` (i.e., that the proposal distribution is symmetric).

The most common use-case for `log_acceptance_correction` is in the
Metropolis-Hastings algorithm, i.e.,

```none
accept_prob(x' | x) = p(x') / p(x) (g(x|x') / g(x'|x))

where,
  p  represents the target distribution,
  g  represents the proposal (conditional) distribution,
  x' is the proposed state, and,
  x  is current state
```

The log of the parenthetical term is the `log_acceptance_correction`.

The `log_acceptance_correction` may not necessarily correspond to the ratio of
proposal distributions, e.g, `log_acceptance_correction` has a different
interpretation in Hamiltonian Monte Carlo.

#### Examples

```python
import tensorflow_probability as tfp
hmc = tfp.mcmc.MetropolisHastings(
    tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
        target_log_prob_fn=lambda x: -x - x**2,
        step_size=0.1,
        num_leapfrog_steps=3))
# ==> functionally equivalent to:
# hmc = tfp.mcmc.HamiltonianMonteCarlo(
#     target_log_prob_fn=lambda x: -x - x**2,
#     step_size=0.1,
#     num_leapfrog_steps=3)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    inner_kernel,
    seed=None,
    name=None
)
```

Instantiates this object.

#### Args:

* <b>`inner_kernel`</b>: `TransitionKernel`-like object which has
  `collections.namedtuple` `kernel_results` and which contains a
  `target_log_prob` member and optionally a `log_acceptance_correction`
  member.
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., "mh_kernel").


#### Returns:

* <b>`metropolis_hastings_kernel`</b>: Instance of `TransitionKernel` which wraps the
  input transition kernel with the Metropolis-Hastings algorithm.



## Properties

<h3 id="inner_kernel"><code>inner_kernel</code></h3>



<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="name"><code>name</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>





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


#### Raises:

* <b>`ValueError`</b>: if `inner_kernel` results doesn't contain the member
  "target_log_prob".

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


#### Raises:

* <b>`ValueError`</b>: if `inner_kernel` results doesn't contain the member
  "target_log_prob".



