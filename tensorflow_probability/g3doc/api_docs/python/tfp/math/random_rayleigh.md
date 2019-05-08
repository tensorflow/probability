<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.random_rayleigh" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.random_rayleigh

Generates `Tensor` of positive reals drawn from a Rayleigh distributions.

``` python
tfp.math.random_rayleigh(
    shape,
    scale=None,
    dtype=tf.float32,
    seed=None,
    name=None
)
```



Defined in [`python/math/random_ops.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/random_ops.py).

<!-- Placeholder for "Used in" -->

The probability density function of a Rayleigh distribution with `scale`
parameter is given by:

```none
f(x) = x scale**-2 exp(-x**2 0.5 scale**-2)
```

For more details, see [Rayleigh distribution](
https://en.wikipedia.org/wiki/Rayleigh_distribution)

#### Args:

* <b>`shape`</b>: Vector-shaped, `int` `Tensor` representing shape of output.
* <b>`scale`</b>: (Optional) Positive `float` `Tensor` representing `Rayleigh` scale.
  Default value: `None` (i.e., `scale = 1.`).
* <b>`dtype`</b>: (Optional) TF `dtype` representing `dtype` of output.
  Default value: `tf.float32`.
* <b>`seed`</b>: (Optional) Python integer to seed the random number generator.
  Default value: `None` (i.e., no seed).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'random_rayleigh').


#### Returns:

* <b>`rayleigh`</b>: `Tensor` with specified `shape` and `dtype` consisting of positive
  real values drawn from a Rayleigh distribution with specified `scale`.