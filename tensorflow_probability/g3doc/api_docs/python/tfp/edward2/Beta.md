<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Beta" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Beta

Create a random variable for Beta.

``` python
tfp.edward2.Beta(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Beta for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a batch of Beta distributions.


#### Args:

* <b>`concentration1`</b>: Positive floating-point `Tensor` indicating mean
  number of successes; aka "alpha". Implies `self.dtype` and
  `self.batch_shape`, i.e.,
  `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
* <b>`concentration0`</b>: Positive floating-point `Tensor` indicating mean
  number of failures; aka "beta". Otherwise has same semantics as
  `concentration1`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.