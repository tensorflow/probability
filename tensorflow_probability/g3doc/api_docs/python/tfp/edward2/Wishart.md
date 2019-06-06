<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Wishart" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Wishart

Create a random variable for Wishart.

``` python
tfp.edward2.Wishart(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Wishart for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Wishart distributions.

#### Args:


* <b>`df`</b>: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
  or equal to dimension of the scale matrix.
* <b>`scale`</b>: `float` or `double` `Tensor`. The symmetric positive definite
  scale matrix of the distribution. Exactly one of `scale` and
  'scale_tril` must be passed.
* <b>`scale_tril`</b>: `float` or `double` `Tensor`. The Cholesky factorization
  of the symmetric positive definite scale matrix of the distribution.
  Exactly one of `scale` and 'scale_tril` must be passed.
* <b>`input_output_cholesky`</b>: Python `bool`. If `True`, functions whose input or
  output have the semantics of samples assume inputs are in Cholesky form
  and return outputs in Cholesky form. In particular, if this flag is
  `True`, input to `log_prob` is presumed of Cholesky form and output from
  `sample`, `mean`, and `mode` are of Cholesky form.  Setting this
  argument to `True` is purely a computational optimization and does not
  change the underlying distribution; for instance, `mean` returns the
  Cholesky of the mean, not the mean of Cholesky factors. The `variance`
  and `stddev` methods are unaffected by this flag.
  Default value: `False` (i.e., input/output does not have Cholesky
  semantics).
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.

#### Raises:


* <b>`ValueError`</b>: if zero or both of 'scale' and 'scale_tril' are passed in.