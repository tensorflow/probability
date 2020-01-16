<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference.is_inferring" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.type_inference.is_inferring


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/type_inference.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Returns whether type inference is running.

### Aliases:

* `tfp.experimental.auto_batching.frontend.ab_type_inference.is_inferring`


``` python
tfp.experimental.auto_batching.type_inference.is_inferring()
```



<!-- Placeholder for "Used in" -->

This can be useful for writing special primitives that change their behavior
depending on whether they are being inferred, staged (see
<a href="../../../../tfp/experimental/auto_batching/virtual_machine/is_staging.md"><code>virtual_machine.is_staging</code></a>), or neither (i.e., dry-run execution, see
<a href="../../../../tfp/experimental/auto_batching/Context.md#batch"><code>frontend.Context.batch</code></a>).

#### Returns:


* <b>`inferring`</b>: Python `bool`, `True` if this is called in the dynamic scope of
  type inference, otherwise `False`.