<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.matrix_rank" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.matrix_rank


<table class="tfo-notebook-buttons tfo-api" align="left">
</table>



Compute the matrix rank of one or more matrices. (deprecated)

``` python
tfp.math.matrix_rank(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-10-01.
Instructions for updating:
tfp.math.matrix_rank is deprecated. Use tf.linalg.matrix_rank instead

#### Arguments:


* <b>`a`</b>: (Batch of) `float`-like matrix-shaped `Tensor`(s) which are to be
  pseudo-inverted.
* <b>`tol`</b>: Threshold below which the singular value is counted as 'zero'.
  Default value: `None` (i.e., `eps * max(rows, cols) * max(singular_val)`).
* <b>`validate_args`</b>: When `True`, additional assertions might be embedded in the
  graph.
  Default value: `False` (i.e., no graph assertions are added).
* <b>`name`</b>: Python `str` prefixed to ops created by this function.
  Default value: 'matrix_rank'.


#### Returns:


* <b>`matrix_rank`</b>: (Batch of) `int32` scalars representing the number of non-zero
  singular values.