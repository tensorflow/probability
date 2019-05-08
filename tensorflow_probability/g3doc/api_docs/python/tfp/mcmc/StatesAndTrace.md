<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.StatesAndTrace" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="all_states"/>
<meta itemprop="property" content="trace"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.mcmc.StatesAndTrace

## Class `StatesAndTrace`

States and auxiliary trace of an MCMC chain.





Defined in [`python/mcmc/sample.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/sample.py).

<!-- Placeholder for "Used in" -->

The first dimension of all the `Tensor`s in this structure is the same and
represents the chain length.

#### Attributes:

* <b>`all_states`</b>: A `Tensor` or a nested collection of `Tensor`s representing the
  MCMC chain state.
* <b>`trace`</b>: A `Tensor` or a nested collection of `Tensor`s representing the
  auxiliary values traced alongside the chain.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    all_states,
    trace
)
```

Create new instance of StatesAndTrace(all_states, trace)



## Properties

<h3 id="all_states"><code>all_states</code></h3>



<h3 id="trace"><code>trace</code></h3>





