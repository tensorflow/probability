<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.TransitionKernel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.TransitionKernel

## Class `TransitionKernel`

Base class for all MCMC `TransitionKernel`s.





Defined in [`python/mcmc/kernel.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/kernel.py).

<!-- Placeholder for "Used in" -->

This class defines the minimal requirements to efficiently implement a Markov
chain Monte Carlo (MCMC) transition kernel. A transition kernel returns a new
state given some old state. It also takes (and returns) "side information"
which may be used for debugging or optimization purposes (i.e, to "recycle"
previously computed results).

## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.



## Methods

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(init_state)
```

Returns an object with the same type as returned by `one_step(...)[1]`.


#### Args:


* <b>`init_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  initial state(s) of the Markov chain(s).


#### Returns:


* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
  `Tensor`s representing internal calculations made within this function.

<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Takes one step of the TransitionKernel.

Must be overridden by subclasses.

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



