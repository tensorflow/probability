<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.is_inferring" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.is_inferring

Returns whether type inference is running.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.is_inferring`
* `tfp.experimental.auto_batching.type_inference.is_inferring`

``` python
tfp.experimental.auto_batching.type_inference.is_inferring()
```



Defined in [`python/internal/auto_batching/type_inference.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/type_inference.py).

<!-- Placeholder for "Used in" -->

This can be useful for writing special primitives that change their behavior
depending on whether they are being inferred, staged (see
`virtual_machine.is_staging`), or neither (i.e., dry-run execution, see
`frontend.Context.batch`).

#### Returns:


* <b>`inferring`</b>: Python `bool`, `True` if this is called in the dynamic scope of
  type inference, otherwise `False`.