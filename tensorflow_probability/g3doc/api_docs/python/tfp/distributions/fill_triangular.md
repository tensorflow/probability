<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.fill_triangular" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.fill_triangular

``` python
tfp.distributions.fill_triangular(
    x,
    upper=False,
    name=None
)
```

Creates a (batch of) triangular matrix from a vector of inputs.

Created matrix can be lower- or upper-triangular. (It is more efficient to
create the matrix as upper or lower, rather than transpose.)

Triangular matrix elements are filled in a clockwise spiral. See example,
below.

If `x.shape` is `[b1, b2, ..., bB, d]` then the output shape is
`[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
`n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

Example:

```python
fill_triangular([1, 2, 3, 4, 5, 6])
# ==> [[4, 0, 0],
#      [6, 5, 0],
#      [3, 2, 1]]

fill_triangular([1, 2, 3, 4, 5, 6], upper=True)
# ==> [[1, 2, 3],
#      [0, 5, 6],
#      [0, 0, 4]]
```

For comparison, a pure numpy version of this function can be found in
`util_test.py`, function `_fill_triangular`.

#### Args:

* <b>`x`</b>: `Tensor` representing lower (or upper) triangular elements.
* <b>`upper`</b>: Python `bool` representing whether output matrix should be upper
    triangular (`True`) or lower triangular (`False`, default).
* <b>`name`</b>: Python `str`. The name to give this op.


#### Returns:

* <b>`tril`</b>: `Tensor` with lower (or upper) triangular elements filled from `x`.


#### Raises:

* <b>`ValueError`</b>: if `x` cannot be mapped to a triangular matrix.