<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.softplus_and_shift" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.trainable_distributions.softplus_and_shift

Converts (batch of) scalars to (batch of) positive valued scalars.

``` python
tfp.trainable_distributions.softplus_and_shift(
    x,
    shift=1e-05,
    name=None
)
```



Defined in [`python/trainable_distributions/trainable_distributions_lib.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/trainable_distributions/trainable_distributions_lib.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`x`</b>: (Batch of) `float`-like `Tensor` representing scalars which will be
  transformed into positive elements.
* <b>`shift`</b>: `Tensor` added to `softplus` transformation of elements.
  Default value: `1e-5`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
  Default value: `None` (i.e., "positive_tril_with_shift").


#### Returns:


* <b>`scale`</b>: (Batch of) scalars`with `x.dtype` and `x.shape`.