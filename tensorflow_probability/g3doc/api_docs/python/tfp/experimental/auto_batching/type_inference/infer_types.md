<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.infer_types" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.infer_types


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/type_inference.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Infers the variable types of a given program.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.infer_types`


``` python
tfp.experimental.auto_batching.type_inference.infer_types(
    program,
    inputs,
    backend
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`program`</b>: <a href="../../../../tfp/experimental/auto_batching/instructions/Program.md"><code>instructions.Program</code></a> whose types to infer.
* <b>`inputs`</b>: A `list` of backend-compatible tensors aligned with
  `program.vars_in`.
* <b>`backend`</b>: Backend implementation.


#### Returns:


* <b>`typed`</b>: <a href="../../../../tfp/experimental/auto_batching/instructions/Program.md"><code>instructions.Program</code></a> with types inferred.


#### Raises:


* <b>`ValueError`</b>: If some types still remain incomplete after inference.