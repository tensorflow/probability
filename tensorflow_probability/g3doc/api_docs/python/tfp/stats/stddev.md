<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.stddev" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.stddev

``` python
tfp.stats.stddev(
    x,
    sample_axis=0,
    keepdims=False,
    name=None
)
```

Estimate standard deviation using samples.

Given `N` samples of scalar valued random variable `X`, standard deviation may
be estimated as

```none
Stddev[X] := Sqrt[Var[X]],
Var[X] := N^{-1} sum_{n=1}^N (X_n - Xbar) Conj{(X_n - Xbar)},
Xbar := N^{-1} sum_{n=1}^N X_n
```

```python
x = tf.random_normal(shape=(100, 2, 3))

# stddev[i, j] is the sample standard deviation of the (i, j) batch member.
stddev = tfp.stats.stddev(x, sample_axis=0)
```

Scaling a unit normal by a standard deviation produces normal samples
with that standard deviation.

```python
observed_data = read_data_samples(...)
stddev = tfp.stats.stddev(observed_data)

# Make fake_data with the same standard deviation as observed_data.
fake_data = stddev * tf.random_normal(shape=(100,))
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
        Default value: `None` (i.e., `'stddev'`).


#### Returns:

* <b>`stddev`</b>: A `Tensor` of same `dtype` as the `x`, and rank equal to
    `rank(x) - len(sample_axis)`