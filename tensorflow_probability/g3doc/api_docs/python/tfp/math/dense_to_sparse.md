<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.dense_to_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.dense_to_sparse


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/sparse.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Converts dense `Tensor` to `SparseTensor`, dropping `ignore_value` cells.

``` python
tfp.math.dense_to_sparse(
    x,
    ignore_value=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`x`</b>: A `Tensor`.
* <b>`ignore_value`</b>: Entries in `x` equal to this value will be
  absent from the return `SparseTensor`. If `None`, default value of
  `x` dtype will be used (e.g. '' for `str`, 0 for `int`).
* <b>`name`</b>: Python `str` prefix for ops created by this function.


#### Returns:


* <b>`sparse_x`</b>: A `tf.SparseTensor` with the same shape as `x`.


#### Raises:


* <b>`ValueError`</b>: when `x`'s rank is `None`.