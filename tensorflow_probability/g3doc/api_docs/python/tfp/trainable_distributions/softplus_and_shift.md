<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.softplus_and_shift" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.trainable_distributions.softplus_and_shift

Converts (batch of) scalars to (batch of) positive valued scalars. (deprecated)

``` python
tfp.trainable_distributions.softplus_and_shift(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-09-01.
Instructions for updating:
`softplus_and_shift` is deprecated; use `tfp.bijectors.{Chain, AffineScalar, Softplus}`.

#### Args:


* <b>`x`</b>: (Batch of) `float`-like `Tensor` representing scalars which will be
  transformed into positive elements.
* <b>`shift`</b>: `Tensor` added to `softplus` transformation of elements.
  Default value: `1e-5`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
  Default value: `None` (i.e., "positive_tril_with_shift").


#### Returns:


* <b>`scale`</b>: (Batch of) scalars`with `x.dtype` and `x.shape`.