<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.sparse_or_dense_matmul" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.sparse_or_dense_matmul

Returns (batched) matmul of a SparseTensor (or Tensor) with a Tensor.

``` python
tfp.math.sparse_or_dense_matmul(
    sparse_or_dense_a,
    dense_b,
    validate_args=False,
    name=None,
    **kwargs
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`sparse_or_dense_a`</b>: `SparseTensor` or `Tensor` representing a (batch of)
  matrices.
* <b>`dense_b`</b>: `Tensor` representing a (batch of) matrices, with the same batch
  shape as `sparse_or_dense_a`. The shape must be compatible with the shape
  of `sparse_or_dense_a` and kwargs.
* <b>`validate_args`</b>: When `True`, additional assertions might be embedded in the
  graph.
  Default value: `False` (i.e., no graph assertions are added).
* <b>`name`</b>: Python `str` prefixed to ops created by this function.
  Default value: 'sparse_or_dense_matmul'.
* <b>`**kwargs`</b>: Keyword arguments to `tf.sparse_tensor_dense_matmul` or
  `tf.matmul`.


#### Returns:


* <b>`product`</b>: A dense (batch of) matrix-shaped Tensor of the same batch shape and
dtype as `sparse_or_dense_a` and `dense_b`. If `sparse_or_dense_a` or
`dense_b` is adjointed through `kwargs` then the shape is adjusted
accordingly.