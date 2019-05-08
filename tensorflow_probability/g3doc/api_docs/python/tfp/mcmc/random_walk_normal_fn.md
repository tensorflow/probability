<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.random_walk_normal_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.random_walk_normal_fn

Returns a callable that adds a random normal perturbation to the input.

``` python
tfp.mcmc.random_walk_normal_fn(
    scale=1.0,
    name=None
)
```



Defined in [`python/mcmc/random_walk_metropolis.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/random_walk_metropolis.py).

<!-- Placeholder for "Used in" -->

This function returns a callable that accepts a Python `list` of `Tensor`s of
any shapes and `dtypes`  representing the state parts of the `current_state`
and a random seed. The supplied argument `scale` must be a `Tensor` or Python
`list` of `Tensor`s representing the scale of the generated
proposal. `scale` must broadcast with the state parts of `current_state`.
The callable adds a sample from a zero-mean normal distribution with the
supplied scales to each state part and returns a same-type `list` of `Tensor`s
as the state parts of `current_state`.

#### Args:

* <b>`scale`</b>: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
  controlling the scale of the normal proposal distribution.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: 'random_walk_normal_fn'.


#### Returns:

* <b>`random_walk_normal_fn`</b>: A callable accepting a Python `list` of `Tensor`s
  representing the state parts of the `current_state` and an `int`
  representing the random seed to be used to generate the proposal. The
  callable returns the same-type `list` of `Tensor`s as the input and
  represents the proposal for the RWM algorithm.