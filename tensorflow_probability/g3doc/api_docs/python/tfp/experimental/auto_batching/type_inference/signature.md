<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.signature" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.signature


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/type_inference.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes a type signature for the given `inputs`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.signature`


``` python
tfp.experimental.auto_batching.type_inference.signature(
    program,
    inputs,
    backend
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`program`</b>: <a href="../../../../tfp/experimental/auto_batching/instructions/Program.md"><code>instructions.Program</code></a> for whose inputs to compute the signature.
* <b>`inputs`</b>: A `list` of backend-compatible tensors aligned with
  `program.vars_in`.
* <b>`backend`</b>: Backend implementation.


#### Returns:


* <b>`sig`</b>: A `list` of (patterns of) <a href="../../../../tfp/experimental/auto_batching/TensorType.md"><code>instructions.TensorType</code></a> aligned with
  `program.vars_in`.