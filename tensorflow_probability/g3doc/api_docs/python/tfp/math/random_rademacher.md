<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.random_rademacher" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.random_rademacher

Generates `Tensor` consisting of `-1` or `+1`, chosen uniformly at random.

``` python
tfp.math.random_rademacher(
    shape,
    dtype=tf.float32,
    seed=None,
    name=None
)
```



Defined in [`python/math/random_ops.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/random_ops.py).

<!-- Placeholder for "Used in" -->

For more details, see [Rademacher distribution](
https://en.wikipedia.org/wiki/Rademacher_distribution).

#### Args:


* <b>`shape`</b>: Vector-shaped, `int` `Tensor` representing shape of output.
* <b>`dtype`</b>: (Optional) TF `dtype` representing `dtype` of output.
* <b>`seed`</b>: (Optional) Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'random_rademacher').


#### Returns:


* <b>`rademacher`</b>: `Tensor` with specified `shape` and `dtype` consisting of `-1`
  or `+1` chosen uniformly-at-random.