<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.math.log1psquare" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.jax.math.log1psquare


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/math/numeric.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

### Aliases:

* `tfp.experimental.substrates.jax.math.numeric.log1psquare`


``` python
tfp.experimental.substrates.jax.math.log1psquare(
    x,
    name=None
)
```



<!-- Placeholder for "Used in" -->

For sufficiently large `x` we use the following observation:

```none
log(1 + x**2) =   2 log(|x|) + log(1 + 1 / x**2)
              --> 2 log(|x|)  as x --> inf
```

Numerically, `log(1 + 1 / x**2)` is `0` when `1 / x**2` is small relative to
machine epsilon.

#### Args:


* <b>`x`</b>: Float `Tensor` input.
* <b>`name`</b>: Python string indicating the name of the TensorFlow operation.
  Default value: `'log1psquare'`.


#### Returns:


* <b>`log1psq`</b>: Float `Tensor` representing `log(1. + x**2.)`.