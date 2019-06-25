<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.fill_triangular_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.fill_triangular_inverse

Creates a vector from a (batch of) triangular matrix. (deprecated)

``` python
tfp.distributions.fill_triangular_inverse(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-10-01.
Instructions for updating:
This function has moved to <a href="../../tfp/math.md"><code>tfp.math</code></a>.

The vector is created from the lower-triangular or upper-triangular portion
depending on the value of the parameter `upper`.

If `x.shape` is `[b1, b2, ..., bB, n, n]` then the output shape is
`[b1, b2, ..., bB, d]` where `d = n (n + 1) / 2`.

#### Example:



```python
fill_triangular_inverse(
  [[4, 0, 0],
   [6, 5, 0],
   [3, 2, 1]])

# ==> [1, 2, 3, 4, 5, 6]

fill_triangular_inverse(
  [[1, 2, 3],
   [0, 5, 6],
   [0, 0, 4]], upper=True)

# ==> [1, 2, 3, 4, 5, 6]
```

#### Args:


* <b>`x`</b>: `Tensor` representing lower (or upper) triangular elements.
* <b>`upper`</b>: Python `bool` representing whether output matrix should be upper
  triangular (`True`) or lower triangular (`False`, default).
* <b>`name`</b>: Python `str`. The name to give this op.


#### Returns:


* <b>`flat_tril`</b>: (Batch of) vector-shaped `Tensor` representing vectorized lower
  (or upper) triangular elements from `x`.