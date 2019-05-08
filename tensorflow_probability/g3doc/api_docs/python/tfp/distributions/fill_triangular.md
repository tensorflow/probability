<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.fill_triangular" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.fill_triangular

Creates a (batch of) triangular matrix from a vector of inputs.

``` python
tfp.distributions.fill_triangular(
    x,
    upper=False,
    name=None
)
```



Defined in [`python/internal/distribution_util.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/distribution_util.py).

<!-- Placeholder for "Used in" -->

Created matrix can be lower- or upper-triangular. (It is more efficient to
create the matrix as upper or lower, rather than transpose.)

Triangular matrix elements are filled in a clockwise spiral. See example,
below.

If `x.shape` is `[b1, b2, ..., bB, d]` then the output shape is
`[b1, b2, ..., bB, n, n]` where `n` is such that `d = n(n+1)/2`, i.e.,
`n = int(np.sqrt(0.25 + 2. * m) - 0.5)`.

#### Example:


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

The key trick is to create an upper triangular matrix by concatenating `x`
and a tail of itself, then reshaping.

Suppose that we are filling the upper triangle of an `n`-by-`n` matrix `M`
from a vector `x`. The matrix `M` contains n**2 entries total. The vector `x`
contains `n * (n+1) / 2` entries. For concreteness, we'll consider `n = 5`
(so `x` has `15` entries and `M` has `25`). We'll concatenate `x` and `x` with
the first (`n = 5`) elements removed and reversed:

```python
x = np.arange(15) + 1
xc = np.concatenate([x, x[5:][::-1]])
# ==> array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15, 14, 13,
#            12, 11, 10, 9, 8, 7, 6])

# (We add one to the arange result to disambiguate the zeros below the
# diagonal of our upper-triangular matrix from the first entry in `x`.)

# Now, when reshapedlay this out as a matrix:
y = np.reshape(xc, [5, 5])
# ==> array([[ 1,  2,  3,  4,  5],
#            [ 6,  7,  8,  9, 10],
#            [11, 12, 13, 14, 15],
#            [15, 14, 13, 12, 11],
#            [10,  9,  8,  7,  6]])

# Finally, zero the elements below the diagonal:
y = np.triu(y, k=0)
# ==> array([[ 1,  2,  3,  4,  5],
#            [ 0,  7,  8,  9, 10],
#            [ 0,  0, 13, 14, 15],
#            [ 0,  0,  0, 12, 11],
#            [ 0,  0,  0,  0,  6]])
```

From this example we see that the resuting matrix is upper-triangular, and
contains all the entries of x, as desired. The rest is details:
- If `n` is even, `x` doesn't exactly fill an even number of rows (it fills
  `n / 2` rows and half of an additional row), but the whole scheme still
  works.
- If we want a lower triangular matrix instead of an upper triangular,
  we remove the first `n` elements from `x` rather than from the reversed
  `x`.

For additional comparisons, a pure numpy version of this function can be found
in `distribution_util_test.py`, function `_fill_triangular`.


#### Args:

* <b>`x`</b>: `Tensor` representing lower (or upper) triangular elements.
* <b>`upper`</b>: Python `bool` representing whether output matrix should be upper
  triangular (`True`) or lower triangular (`False`, default).
* <b>`name`</b>: Python `str`. The name to give this op.


#### Returns:

* <b>`tril`</b>: `Tensor` with lower (or upper) triangular elements filled from `x`.


#### Raises:

* <b>`ValueError`</b>: if `x` cannot be mapped to a triangular matrix.