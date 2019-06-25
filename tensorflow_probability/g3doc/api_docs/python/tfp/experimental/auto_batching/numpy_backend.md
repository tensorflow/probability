<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.numpy_backend" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.numpy_backend

Numpy backend for auto-batching VM.



Defined in [`python/internal/auto_batching/numpy_backend.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/numpy_backend.py).

<!-- Placeholder for "Used in" -->

It can be faster than TF for tiny examples and prototyping, and moderately
simpler due to immediate as opposed to deferred result computation.

All operations take and ignore name= arguments to allow for useful op names in
the TensorFlow backend.

## Classes

[`class NumpyBackend`](../../../tfp/experimental/auto_batching/NumpyBackend.md): Implements the Numpy backend ops for a PC auto-batching VM.

