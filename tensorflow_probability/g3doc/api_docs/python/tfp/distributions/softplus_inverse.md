<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.softplus_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.softplus_inverse

Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

``` python
tfp.distributions.softplus_inverse(
    x,
    name=None
)
```



Defined in [`python/internal/distribution_util.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/distribution_util.py).

<!-- Placeholder for "Used in" -->

Mathematically this op is equivalent to:

```none
softplus_inverse = log(exp(x) - 1.)
```

#### Args:


* <b>`x`</b>: `Tensor`. Non-negative (not enforced), floating-point.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

`Tensor`. Has the same type/shape as input `x`.
