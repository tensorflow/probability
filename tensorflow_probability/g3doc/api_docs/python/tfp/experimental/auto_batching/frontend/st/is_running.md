<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.frontend.st.is_running" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.frontend.st.is_running

Returns whether the stackless machine is running a computation.

``` python
tfp.experimental.auto_batching.frontend.st.is_running()
```



Defined in [`python/internal/auto_batching/stackless.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/stackless.py).

<!-- Placeholder for "Used in" -->

This can be useful for writing special primitives that change their behavior
depending on whether they are being staged, run stackless, inferred (see
`type_inference.is_inferring`), or none of the above (i.e., dry-run execution,
see `frontend.Context.batch`).

#### Returns:


* <b>`running`</b>: Python `bool`, `True` if this is called in the dynamic scope of
  stackless running, otherwise `False`.