<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.histogram" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.histogram

Count how often `x` falls in intervals defined by `edges`.

``` python
tfp.stats.histogram(
    x,
    edges,
    axis=None,
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
This function counts how often `x` falls into each interval.

Values of `x` outside of the intervals cause errors.  Consider using
`extend_lower_interval`, `extend_upper_interval` to deal with this.

#### Args:


* <b>`x`</b>:  Numeric `N-D` `Tensor` with `N > 0`.  If `axis` is not
  `None`, must have statically known number of dimensions. The
  `axis` kwarg determines which dimensions index iid samples.
  Other dimensions of `x` index "events" for which we will compute different
  histograms.
* <b>`edges`</b>:  `Tensor` of same `dtype` as `x`.  The first dimension indexes edges
  of intervals.  Must either be `1-D` or have `edges.shape[1:]` the same
  as the dimensions of `x` excluding `axis`.
  If `rank(edges) > 1`, `edges[k]` designates a shape `edges.shape[1:]`
  `Tensor` of interval edges for the corresponding dimensions of `x`.
* <b>`axis`</b>:  Optional `0-D` or `1-D` integer `Tensor` with constant
  values. The axis in `x` that index iid samples.
  `Default value:` `None` (treat every dimension as sample dimension).
* <b>`extend_lower_interval`</b>:  Python `bool`.  If `True`, extend the lowest
  interval `I0` to `(-inf, c1]`.
* <b>`extend_upper_interval`</b>:  Python `bool`.  If `True`, extend the upper
  interval `I_{K-1}` to `[c_{K-1}, +inf)`.
* <b>`dtype`</b>: The output type (`int32` or `int64`). `Default value:` `x.dtype`.
* <b>`name`</b>:  A Python string name to prepend to created ops.
  `Default value:` 'histogram'


#### Returns:


* <b>`counts`</b>: `Tensor` of type `dtype` and, with
  `~axis = [i for i in range(arr.ndim) if i not in axis]`,
  `counts.shape = [edges.shape[0]] + x.shape[~axis]`.
  With `I` a multi-index into `~axis`, `counts[k][I]` is the number of times
  event(s) fell into the `kth` interval of `edges`.

#### Examples

```python
# x.shape = [1000, 2]
# x[:, 0] ~ Uniform(0, 1), x[:, 1] ~ Uniform(1, 2).
x = tf.stack([tf.random_uniform([1000]), 1 + tf.random_uniform([1000])],
             axis=-1)

# edges ==> bins [0, 0.5), [0.5, 1.0), [1.0, 1.5), [1.5, 2.0].
edges = [0., 0.5, 1.0, 1.5, 2.0]

tfp.stats.histogram(x, edges)
==> approximately [500, 500, 500, 500]

tfp.stats.histogram(x, edges, axis=0)
==> approximately [[500, 500, 0, 0], [0, 0, 500, 500]]
```