<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.find_bins" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.find_bins

Bin values into discrete intervals.

``` python
tfp.stats.find_bins(
    x,
    edges,
    extend_lower_interval=False,
    extend_upper_interval=False,
    dtype=None,
    name=None
)
```



Defined in [`python/stats/quantiles.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/stats/quantiles.py).

<!-- Placeholder for "Used in" -->

Given `edges = [c0, ..., cK]`, defining intervals
`I0 = [c0, c1)`, `I1 = [c1, c2)`, ..., `I_{K-1} = [c_{K-1}, cK]`,
This function returns `bins`, such that:
`edges[bins[i]] <= x[i] < edges[bins[i] + 1]`.

#### Args:


* <b>`x`</b>:  Numeric `N-D` `Tensor` with `N > 0`.
* <b>`edges`</b>:  `Tensor` of same `dtype` as `x`.  The first dimension indexes edges
  of intervals.  Must either be `1-D` or have
  `x.shape[1:] == edges.shape[1:]`.  If `rank(edges) > 1`, `edges[k]`
  designates a shape `edges.shape[1:]` `Tensor` of bin edges for the
  corresponding dimensions of `x`.
* <b>`extend_lower_interval`</b>:  Python `bool`.  If `True`, extend the lowest
  interval `I0` to `(-inf, c1]`.
* <b>`extend_upper_interval`</b>:  Python `bool`.  If `True`, extend the upper
  interval `I_{K-1}` to `[c_{K-1}, +inf)`.
* <b>`dtype`</b>: The output type (`int32` or `int64`). `Default value:` `x.dtype`.
  This effects the output values when `x` is below/above the intervals,
  which will be `-1/K+1` for `int` types and `NaN` for `float`s.
  At indices where `x` is `NaN`, the output values will be `0` for `int`
  types and `NaN` for floats.
* <b>`name`</b>:  A Python string name to prepend to created ops. Default: 'find_bins'


#### Returns:


* <b>`bins`</b>: `Tensor` with same `shape` as `x` and `dtype`.
  Has whole number values.  `bins[i] = k` means the `x[i]` falls into the
  `kth` bin, ie, `edges[bins[i]] <= x[i] < edges[bins[i] + 1]`.


#### Raises:


* <b>`ValueError`</b>:  If `edges.shape[0]` is determined to be less than 2.

#### Examples

Cut a `1-D` array

```python
x = [0., 5., 6., 10., 20.]
edges = [0., 5., 10.]
tfp.stats.find_bins(x, edges)
==> [0., 0., 1., 1., np.nan]
```

Cut `x` into its deciles

```python
x = tf.random_uniform(shape=(100, 200))
decile_edges = tfp.stats.quantiles(x, num_quantiles=10)
bins = tfp.stats.find_bins(x, edges=decile_edges)
bins.shape
==> (100, 200)
tf.reduce_mean(bins == 0.)
==> approximately 0.1
tf.reduce_mean(bins == 1.)
==> approximately 0.1
```