<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.DirichletMultinomial" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.DirichletMultinomial

Create a random variable for DirichletMultinomial.

``` python
tfp.edward2.DirichletMultinomial(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See DirichletMultinomial for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize a batch of DirichletMultinomial distributions.

#### Args:


* <b>`total_count`</b>:  Non-negative floating point tensor, whose dtype is the same
  as `concentration`. The shape is broadcastable to `[N1,..., Nm]` with
  `m >= 0`. Defines this as a batch of `N1 x ... x Nm` different
  Dirichlet multinomial distributions. Its components should be equal to
  integer values.
* <b>`concentration`</b>: Positive floating point tensor, whose dtype is the
  same as `n` with shape broadcastable to `[N1,..., Nm, K]` `m >= 0`.
  Defines this as a batch of `N1 x ... x Nm` different `K` class Dirichlet
  multinomial distributions.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.