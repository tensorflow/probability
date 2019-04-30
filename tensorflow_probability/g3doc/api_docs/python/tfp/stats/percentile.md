<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.percentile" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.percentile

``` python
tfp.stats.percentile(
    x,
    q,
    axis=None,
    interpolation=None,
    keep_dims=False,
    validate_args=False,
    preserve_gradients=True,
    name=None
)
```

Compute the `q`-th percentile(s) of `x`.

Given a vector `x`, the `q`-th percentile of `x` is the value `q / 100` of the
way from the minimum to the maximum in a sorted copy of `x`.

The values and distances of the two nearest neighbors as well as the
`interpolation` parameter will determine the percentile if the normalized
ranking does not match the location of `q` exactly.

This function is the same as the median if `q = 50`, the same as the minimum
if `q = 0` and the same as the maximum if `q = 100`.

Multiple percentiles can be computed at once by using `1-D` vector `q`.
Dimension zero of the returned `Tensor` will index the different percentiles.


```python
# Get 30th percentile with default ('nearest') interpolation.
x = [1., 2., 3., 4.]
tfp.stats.percentile(x, q=30.)
==> 2.0

# Get 30th percentile with 'linear' interpolation.
x = [1., 2., 3., 4.]
tfp.stats.percentile(x, q=30., interpolation='linear')
==> 1.9

# Get 30th and 70th percentiles with 'lower' interpolation
x = [1., 2., 3., 4.]
tfp.stats.percentile(x, q=[30., 70.], interpolation='lower')
==> [1., 3.]

# Get 100th percentile (maximum).  By default, this is computed over every dim
x = [[1., 2.]
     [3., 4.]]
tfp.stats.percentile(x, q=100.)
==> 4.

# Treat the leading dim as indexing samples, and find the 100th quantile (max)
# over all such samples.
x = [[1., 2.]
     [3., 4.]]
tfp.stats.percentile(x, q=100., axis=[0])
==> [3., 4.]
```

Compare to `numpy.percentile`.

#### Args:

* <b>`x`</b>:  Floating point `N-D` `Tensor` with `N > 0`.  If `axis` is not `None`,
    `x` must have statically known number of dimensions.
* <b>`q`</b>:  Scalar or vector `Tensor` with values in `[0, 100]`. The percentile(s).
* <b>`axis`</b>:  Optional `0-D` or `1-D` integer `Tensor` with constant values. The
    axis that hold independent samples over which to return the desired
    percentile.  If `None` (the default), treat every dimension as a sample
    dimension, returning a scalar.
* <b>`interpolation `</b>: {'nearest', 'linear', 'lower', 'higher', 'midpoint'}.
    Default value: 'nearest'.  This specifies the interpolation method to
    use when the desired quantile lies between two data points `i < j`:
      * linear: i + (j - i) * fraction, where fraction is the fractional part
        of the index surrounded by i and j.
      * lower: `i`.
      * higher: `j`.
      * nearest: `i` or `j`, whichever is nearest.
      * midpoint: (i + j) / 2.
    `linear` and `midpoint` interpolation do not work with integer dtypes.
* <b>`keep_dims`</b>:  Python `bool`. If `True`, the last dimension is kept with size 1
    If `False`, the last dimension is removed from the output shape.
* <b>`validate_args`</b>:  Whether to add runtime checks of argument validity. If
    False, and arguments are incorrect, correct behavior is not guaranteed.
* <b>`preserve_gradients`</b>:  Python `bool`.  If `True`, ensure that gradient w.r.t
    the percentile `q` is preserved in the case of linear interpolation.
    If `False`, the gradient will be (incorrectly) zero when `q` corresponds
    to a point in `x`.
* <b>`name`</b>:  A Python string name to give this `Op`.  Default is 'percentile'


#### Returns:

A `(rank(q) + N - len(axis))` dimensional `Tensor` of same dtype as `x`, or,
  if `axis` is `None`, a `rank(q)` `Tensor`.  The first `rank(q)` dimensions
  index quantiles for different values of `q`.


#### Raises:

* <b>`ValueError`</b>:  If argument 'interpolation' is not an allowed type.
* <b>`ValueError`</b>:  If interpolation type not compatible with `dtype`.