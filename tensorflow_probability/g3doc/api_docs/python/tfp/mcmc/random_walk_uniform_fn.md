<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.random_walk_uniform_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.random_walk_uniform_fn

Returns a callable that adds a random uniform perturbation to the input.

``` python
tfp.mcmc.random_walk_uniform_fn(
    scale=1.0,
    name=None
)
```



Defined in [`python/mcmc/random_walk_metropolis.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/random_walk_metropolis.py).

<!-- Placeholder for "Used in" -->

For more details on `random_walk_uniform_fn`, see
`random_walk_normal_fn`. `scale` might
be a `Tensor` or a list of `Tensor`s that should broadcast with state parts
of the `current_state`. The generated uniform perturbation is sampled as a
uniform point on the rectangle `[-scale, scale]`.

#### Args:

* <b>`scale`</b>: a `Tensor` or Python `list` of `Tensor`s of any shapes and `dtypes`
  controlling the upper and lower bound of the uniform proposal
  distribution.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: 'random_walk_uniform_fn'.


#### Returns:

* <b>`random_walk_uniform_fn`</b>: A callable accepting a Python `list` of `Tensor`s
  representing the state parts of the `current_state` and an `int`
  representing the random seed used to generate the proposal. The callable
  returns the same-type `list` of `Tensor`s as the input and represents the
  proposal for the RWM algorithm.