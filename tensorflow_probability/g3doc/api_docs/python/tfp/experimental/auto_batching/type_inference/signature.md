<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.signature" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.signature

Computes a type signature for the given `inputs`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.signature`
* `tfp.experimental.auto_batching.type_inference.signature`

``` python
tfp.experimental.auto_batching.type_inference.signature(
    program,
    inputs,
    backend
)
```



Defined in [`python/internal/auto_batching/type_inference.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/type_inference.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`program`</b>: `instructions.Program` for whose inputs to compute the signature.
* <b>`inputs`</b>: A `list` of backend-compatible tensors aligned with
  `program.vars_in`.
* <b>`backend`</b>: Backend implementation.


#### Returns:


* <b>`sig`</b>: A `list` of (patterns of) `instructions.TensorType` aligned with
  `program.vars_in`.