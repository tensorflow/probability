<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.DirichletMultinomial" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.DirichletMultinomial


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for DirichletMultinomial.

### Aliases:

* `tfp.experimental.edward2.DirichletMultinomial`


``` python
tfp.edward2.DirichletMultinomial(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See DirichletMultinomial for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize a batch of DirichletMultinomial distributions.

#### Args:

total_count: Non-negative integer-valued tensor, whose dtype is the same
  as `concentration`. The shape is broadcastable to `[N1,..., Nm]` with
  `m >= 0`. Defines this as a batch of `N1 x ... x Nm` different
  Dirichlet multinomial distributions. Its components should be equal to
  integer values.
concentration: Positive floating point tensor with shape broadcastable to
  `[N1,..., Nm, K]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
  different `K` class Dirichlet multinomial distributions.
validate_args: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.

* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
   (e.g., mean, variance) use the value "`NaN`" to indicate the result is
   undefined. When `False`, an exception is raised if one or more of the
   statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.