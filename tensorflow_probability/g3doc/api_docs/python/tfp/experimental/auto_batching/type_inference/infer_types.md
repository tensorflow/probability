<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.infer_types" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.infer_types

Infers the variable types of a given program.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.infer_types`
* `tfp.experimental.auto_batching.type_inference.infer_types`

``` python
tfp.experimental.auto_batching.type_inference.infer_types(
    program,
    inputs,
    backend
)
```



Defined in [`python/internal/auto_batching/type_inference.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/type_inference.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`program`</b>: `instructions.Program` whose types to infer.
* <b>`inputs`</b>: A `list` of backend-compatible tensors aligned with
  `program.vars_in`.
* <b>`backend`</b>: Backend implementation.


#### Returns:


* <b>`typed`</b>: `instructions.Program` with types inferred.


#### Raises:


* <b>`ValueError`</b>: If some types still remain incomplete after inference.