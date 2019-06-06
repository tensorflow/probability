<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.extract_referenced_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.instructions.extract_referenced_variables

Extracts a set of the variable names referenced by the node in question.

### Aliases:

* `tfp.experimental.auto_batching.frontend.instructions.extract_referenced_variables`
* `tfp.experimental.auto_batching.frontend.st.inst.extract_referenced_variables`
* `tfp.experimental.auto_batching.instructions.extract_referenced_variables`

``` python
tfp.experimental.auto_batching.instructions.extract_referenced_variables(node)
```



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`node`</b>: Most structures from the VM, including ops, sequences of ops, blocks.


#### Returns:


* <b>`varnames`</b>: `set` of variable names referenced by the node in question.