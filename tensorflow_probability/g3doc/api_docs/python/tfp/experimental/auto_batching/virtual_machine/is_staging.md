<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.virtual_machine.is_staging" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.virtual_machine.is_staging

Returns whether the virtual machine is staging a computation.

### Aliases:

* `tfp.experimental.auto_batching.frontend.vm.is_staging`
* `tfp.experimental.auto_batching.virtual_machine.is_staging`

``` python
tfp.experimental.auto_batching.virtual_machine.is_staging()
```



Defined in [`python/internal/auto_batching/virtual_machine.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/virtual_machine.py).

<!-- Placeholder for "Used in" -->

This can be useful for writing special primitives that change their behavior
depending on whether they are being staged, run stackless, inferred (see
`type_inference.is_inferring`), or none of the above (i.e., dry-run execution,
see `frontend.Context.batch`).

#### Returns:


* <b>`staging`</b>: Python `bool`, `True` if this is called in the dynamic scope of
  VM staging, otherwise `False`.