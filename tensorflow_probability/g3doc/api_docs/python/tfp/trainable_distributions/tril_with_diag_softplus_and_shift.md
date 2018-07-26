Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions.tril_with_diag_softplus_and_shift" />
</div>

# tfp.trainable_distributions.tril_with_diag_softplus_and_shift

``` python
tfp.trainable_distributions.tril_with_diag_softplus_and_shift(
    x,
    diag_shift=1e-05,
    name=None
)
```

Converts (batch of) vectors to (batch of) lower-triangular scale matrices.

#### Args:

* <b>`x`</b>: (Batch of) `float`-like `Tensor` representing vectors which will be
    transformed into lower-triangular scale matrices with positive diagonal
    elements. Rightmost shape `n` must be such that
    `n = dims * (dims + 1) / 2` for some positive, integer `dims`.
* <b>`diag_shift`</b>: `Tensor` added to `softplus` transformation of diagonal
    elements.
    Default value: `1e-5`.
* <b>`name`</b>: A `name_scope` name for operations created by this function.
    Default value: `None` (i.e., "tril_with_diag_softplus_and_shift").


#### Returns:

* <b>`scale_tril`</b>: (Batch of) lower-triangular `Tensor` with `x.dtype` and
    rightmost shape `[dims, dims]` where `n = dims * (dims + 1) / 2` where
    `n = x.shape[-1]`.