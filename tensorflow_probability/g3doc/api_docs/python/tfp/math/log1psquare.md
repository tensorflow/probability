<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.log1psquare" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.log1psquare

Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

``` python
tfp.math.log1psquare(
    x,
    name=None
)
```



Defined in [`python/math/numeric.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/numeric.py).

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