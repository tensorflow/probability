<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.math.generic.smootherstep" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.jax.math.generic.smootherstep


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes a sigmoid-like interpolation function on the unit-interval.

``` python
tfp.experimental.substrates.jax.math.generic.smootherstep(
    x,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Equivalent to:



```python
x = tf.clip_by_value(x, clip_value_min=0., clip_value_max=1.)
y = x**3. * (6. * x**2. - 15. * x + 10.)
```

For more details see [Wikipedia][1].

#### Args:


* <b>`x`</b>: `float` `Tensor`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `'smootherstep'`).


#### Returns:


* <b>`smootherstep`</b>: `float` `Tensor` with the same shape and dtype as `x`,
  representing the value of the smootherstep function.

#### References

[1]: "Smoothstep." Wikipedia.
     https://en.wikipedia.org/wiki/Smoothstep#Variations