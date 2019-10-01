<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.type_of_pattern" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.type_of_pattern


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/type_inference.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Returns the <a href="../../../../tfp/experimental/auto_batching/Type.md"><code>instructions.Type</code></a> of `val`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.type_of_pattern`


``` python
tfp.experimental.auto_batching.type_inference.type_of_pattern(
    val,
    backend,
    preferred_type=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`val`</b>: Pattern of backend-specific `Tensor`s or a Python or numpy constant.
* <b>`backend`</b>: Object implementing required backend operations.
* <b>`preferred_type`</b>: <a href="../../../../tfp/experimental/auto_batching/Type.md"><code>instructions.Type</code></a> to prefer, if `t` is a constant.


#### Returns:


* <b>`vm_type`</b>: Pattern of <a href="../../../../tfp/experimental/auto_batching/TensorType.md"><code>instructions.TensorType</code></a> describing `t`