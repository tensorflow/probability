<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.push_op" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.instructions.push_op

Returns an `Op` that pushes values from `vars_in` into `vars_out`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.instructions.push_op`
* `tfp.experimental.auto_batching.frontend.st.inst.push_op`
* `tfp.experimental.auto_batching.instructions.push_op`

``` python
tfp.experimental.auto_batching.instructions.push_op(
    vars_in,
    vars_out
)
```



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`vars_in`</b>: Python pattern of `string`, the variables to read.
* <b>`vars_out`</b>: Python pattern of `string`, matching with `vars_in`; the
  variables to write to.


#### Returns:


* <b>`op`</b>: An `Op` that accomplishes the push.