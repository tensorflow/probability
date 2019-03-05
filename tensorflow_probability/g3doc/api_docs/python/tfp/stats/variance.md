<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.variance" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.variance

``` python
tfp.stats.variance(
    x,
    sample_axis=0,
    keepdims=False,
    name=None
)
```

Estimate variance using samples.

Given `N` samples of scalar valued random variable `X`, variance may
be estimated as

```none
Var[X] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(X_n - Xbar)}
Xbar := N^{-1} sum_{n=1}^N X_n
```

```python
x = tf.random_normal(shape=(100, 2, 3))

# var[i, j] is the sample variance of the (i, j) batch member of x.
var = tfp.stats.variance(x, sample_axis=0)
```

Notice we divide by `N` (the numpy default), which does not create `NaN`
when `N = 1`, but is slightly biased.

#### Args:

* <b>`x`</b>:  A numeric `Tensor` holding samples.
* <b>`sample_axis`</b>: Scalar or vector `Tensor` designating axis holding samples, or
    `None` (meaning all axis hold samples).
    Default value: `0` (leftmost dimension).
* <b>`keepdims`</b>:  Boolean.  Whether to keep the sample axis as singletons.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
        Default value: `None` (i.e., `'variance'`).


#### Returns:

* <b>`var`</b>: A `Tensor` of same `dtype` as the `x`, and rank equal to
    `rank(x) - len(sample_axis)`