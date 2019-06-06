<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.type_of_pattern" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.type_of_pattern

Returns the `instructions.Type` of `val`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.type_of_pattern`
* `tfp.experimental.auto_batching.type_inference.type_of_pattern`

``` python
tfp.experimental.auto_batching.type_inference.type_of_pattern(
    val,
    backend,
    preferred_type=None
)
```



Defined in [`python/internal/auto_batching/type_inference.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/type_inference.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`val`</b>: Pattern of backend-specific `Tensor`s or a Python or numpy constant.
* <b>`backend`</b>: Object implementing required backend operations.
* <b>`preferred_type`</b>: `instructions.Type` to prefer, if `t` is a constant.


#### Returns:


* <b>`vm_type`</b>: Pattern of `instructions.TensorType` describing `t`