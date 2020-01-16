<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.stackless.is_running" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.stackless.is_running


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/stackless.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Returns whether the stackless machine is running a computation.

### Aliases:

* `tfp.experimental.auto_batching.frontend.st.is_running`


``` python
tfp.experimental.auto_batching.stackless.is_running()
```



<!-- Placeholder for "Used in" -->

This can be useful for writing special primitives that change their behavior
depending on whether they are being staged, run stackless, inferred (see
<a href="../../../../tfp/experimental/auto_batching/type_inference/is_inferring.md"><code>type_inference.is_inferring</code></a>), or none of the above (i.e., dry-run execution,
see <a href="../../../../tfp/experimental/auto_batching/Context.md#batch"><code>frontend.Context.batch</code></a>).

#### Returns:


* <b>`running`</b>: Python `bool`, `True` if this is called in the dynamic scope of
  stackless running, otherwise `False`.