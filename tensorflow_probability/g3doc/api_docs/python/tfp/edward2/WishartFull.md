Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.WishartFull" />
</div>

# tfp.edward2.WishartFull

``` python
tfp.edward2.WishartFull(
    *args,
    **kwargs
)
```

Create a random variable for WishartFull.

See WishartFull for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Wishart distributions.


#### Args:

* <b>`df`</b>: `float` or `double` `Tensor`. Degrees of freedom, must be greater than
    or equal to dimension of the scale matrix.
* <b>`scale`</b>: `float` or `double` `Tensor`. The symmetric positive definite
    scale matrix of the distribution.
* <b>`cholesky_input_output_matrices`</b>: Python `bool`. Any function which whose
    input or output is a matrix assumes the input is Cholesky and returns a
    Cholesky factored matrix. Example `log_prob` input takes a Cholesky and
    `sample_n` returns a Cholesky when
    `cholesky_input_output_matrices=True`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.