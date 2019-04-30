<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.quantiles" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.quantiles

``` python
tfp.stats.quantiles(
    x,
    num_quantiles,
    axis=None,
    interpolation=None,
    keep_dims=False,
    validate_args=False,
    name=None
)
```

Compute quantiles of `x` along `axis`.

The quantiles of a distribution are cut points dividing the range into
intervals with equal probabilities.

Given a vector `x` of samples, this function estimates the cut points by
returning `num_quantiles + 1` cut points, `(c0, ..., cn)`, such that, roughly
speaking, equal number of sample points lie in the `num_quantiles` intervals
`[c0, c1), [c1, c2), ..., [c_{n-1}, cn]`.  That is,

* About `1 / n` fraction of the data lies in `[c_{k-1}, c_k)`, `k = 1, ..., n`
* About `k / n` fraction of the data lies below `c_k`.
* `c0` is the sample minimum and `cn` is the maximum.

The exact number of data points in each interval depends on the size of
`x` (e.g. whether the size is divisible by `n`) and the `interpolation` kwarg.


```python
# Get quartiles of x with various interpolation choices.
x = [0.,  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.]

tfp.stats.quantiles(x, num_quantiles=4, interpolation='nearest')
==> [  0.,   2.,   5.,   8.,  10.]

tfp.stats.quantiles(x, num_quantiles=4, interpolation='linear')
==> [  0. ,   2.5,   5. ,   7.5,  10. ]

tfp.stats.quantiles(x, num_quantiles=4, interpolation='lower')
==> [  0.,   2.,   5.,   7.,  10.]

# Get deciles of columns of an R x C data set.
data = load_my_columnar_data(...)
tfp.stats.quantiles(data, num_quantiles=10)
==> Shape [11, C] Tensor
```

#### Args:

* <b>`x`</b>:  Floating point `N-D` `Tensor` with `N > 0`.  If `axis` is not `None`,
    `x` must have statically known number of dimensions.
* <b>`num_quantiles`</b>:  Scalar `integer` `Tensor`.  The number of intervals the
    returned `num_quantiles + 1` cut points divide the range into.
* <b>`axis`</b>:  Optional `0-D` or `1-D` integer `Tensor` with constant values. The
    axis that hold independent samples over which to return the desired
    percentile.  If `None` (the default), treat every dimension as a sample
    dimension, returning a scalar.
* <b>`interpolation `</b>: {'nearest', 'linear', 'lower', 'higher', 'midpoint'}.
    Default value: 'nearest'.  This specifies the interpolation method to
    use when the fractions `k / n` lie between two data points `i < j`:
      * linear: i + (j - i) * fraction, where fraction is the fractional part
        of the index surrounded by i and j.
      * lower: `i`.
      * higher: `j`.
      * nearest: `i` or `j`, whichever is nearest.
      * midpoint: (i + j) / 2. `linear` and `midpoint` interpolation do not
        work with integer dtypes.
* <b>`keep_dims`</b>:  Python `bool`. If `True`, the last dimension is kept with size 1
    If `False`, the last dimension is removed from the output shape.
* <b>`validate_args`</b>:  Whether to add runtime checks of argument validity. If
    False, and arguments are incorrect, correct behavior is not guaranteed.
* <b>`name`</b>:  A Python string name to give this `Op`.  Default is 'percentile'


#### Returns:

* <b>`cut_points`</b>:  A `rank(x) + 1 - len(axis)` dimensional `Tensor` with same
  `dtype` as `x` and shape `[num_quantiles + 1, ...]` where the trailing shape
  is that of `x` without the dimensions in `axis` (unless `keep_dims is True`)


#### Raises:

* <b>`ValueError`</b>:  If argument 'interpolation' is not an allowed type.
* <b>`ValueError`</b>:  If interpolation type not compatible with `dtype`.