<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.random_rademacher" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.random_rademacher


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/random_ops.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Generates `Tensor` consisting of `-1` or `+1`, chosen uniformly at random.

``` python
tfp.math.random_rademacher(
    shape,
    dtype=tf.float32,
    seed=None,
    name=None
)
```



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