<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.extract_referenced_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.instructions.extract_referenced_variables


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Extracts a set of the variable names referenced by the node in question.

### Aliases:

* `tfp.experimental.auto_batching.frontend.instructions.extract_referenced_variables`
* `tfp.experimental.auto_batching.frontend.st.inst.extract_referenced_variables`
* `tfp.experimental.auto_batching.frontend.stack.inst.extract_referenced_variables`
* `tfp.experimental.auto_batching.stack_optimization.inst.extract_referenced_variables`
* `tfp.experimental.auto_batching.stackless.inst.extract_referenced_variables`


``` python
tfp.experimental.auto_batching.instructions.extract_referenced_variables(node)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`node`</b>: Most structures from the VM, including ops, sequences of ops, blocks.


#### Returns:


* <b>`varnames`</b>: `set` of variable names referenced by the node in question.