<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.numpy_backend" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.numpy_backend


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/numpy_backend.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Numpy backend for auto-batching VM.

<!-- Placeholder for "Used in" -->

It can be faster than TF for tiny examples and prototyping, and moderately
simpler due to immediate as opposed to deferred result computation.

All operations take and ignore name= arguments to allow for useful op names in
the TensorFlow backend.

## Classes

[`class NumpyBackend`](../../../tfp/experimental/auto_batching/NumpyBackend.md): Implements the Numpy backend ops for a PC auto-batching VM.

