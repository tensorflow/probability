<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.covariance" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.covariance


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/sample_stats.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Sample covariance between observations indexed by `event_axis`.

``` python
tfp.stats.covariance(
    x,
    y=None,
    sample_axis=0,
    event_axis=-1,
    keepdims=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Given `N` samples of scalar random variables `X` and `Y`, covariance may be
estimated as

```none
Cov[X, Y] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(Y_n - Ybar)}
Xbar := N^{-1} sum_{n=1}^N X_n
Ybar := N^{-1} sum_{n=1}^N Y_n
```

For vector-variate random variables `X = (X1, ..., Xd)`, `Y = (Y1, ..., Yd)`,
one is often interested in the covariance matrix, `C_{ij} := Cov[Xi, Yj]`.

```python
x = tf.random_normal(shape=(100, 2, 3))
y = tf.random_normal(shape=(100, 2, 3))

# cov[i, j] is the sample covariance between x[:, i, j] and y[:, i, j].
cov = tfp.stats.covariance(x, y, sample_axis=0, event_axis=None)

# cov_matrix[i, m, n] is the sample covariance of x[:, i, m] and y[:, i, n]
cov_matrix = tfp.stats.covariance(x, y, sample_axis=0, event_axis=-1)
```

Notice we divide by `N` (the numpy default), which does not create `NaN`
when `N = 1`, but is slightly biased.

#### Args:


* <b>`x`</b>:  A numeric `Tensor` holding samples.
* <b>`y`</b>:  Optional `Tensor` with same `dtype` and `shape` as `x`.
  Default value: `None` (`y` is effectively set to `x`).
* <b>`sample_axis`</b>: Scalar or vector `Tensor` designating axis holding samples, or
  `None` (meaning all axis hold samples).
  Default value: `0` (leftmost dimension).
* <b>`event_axis`</b>:  Scalar or vector `Tensor`, or `None` (scalar events).
  Axis indexing random events, whose covariance we are interested in.
  If a vector, entries must form a contiguous block of dims. `sample_axis`
  and `event_axis` should not intersect.
  Default value: `-1` (rightmost axis holds events).
* <b>`keepdims`</b>:  Boolean.  Whether to keep the sample axis as singletons.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'covariance'`).


#### Returns:


* <b>`cov`</b>: A `Tensor` of same `dtype` as the `x`, and rank equal to
  `rank(x) - len(sample_axis) + 2 * len(event_axis)`.


#### Raises:


* <b>`AssertionError`</b>:  If `x` and `y` are found to have different shape.
* <b>`ValueError`</b>:  If `sample_axis` and `event_axis` are found to overlap.
* <b>`ValueError`</b>:  If `event_axis` is found to not be contiguous.