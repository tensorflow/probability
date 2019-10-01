<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.push_op" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.instructions.push_op


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Returns an `Op` that pushes values from `vars_in` into `vars_out`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.instructions.push_op`
* `tfp.experimental.auto_batching.frontend.st.inst.push_op`
* `tfp.experimental.auto_batching.frontend.stack.inst.push_op`
* `tfp.experimental.auto_batching.stack_optimization.inst.push_op`
* `tfp.experimental.auto_batching.stackless.inst.push_op`


``` python
tfp.experimental.auto_batching.instructions.push_op(
    vars_in,
    vars_out
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`vars_in`</b>: Python pattern of `string`, the variables to read.
* <b>`vars_out`</b>: Python pattern of `string`, matching with `vars_in`; the
  variables to write to.


#### Returns:


* <b>`op`</b>: An `Op` that accomplishes the push.