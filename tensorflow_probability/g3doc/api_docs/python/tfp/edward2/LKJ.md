<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.LKJ" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.LKJ


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for LKJ.

### Aliases:

* `tfp.experimental.edward2.LKJ`


``` python
tfp.edward2.LKJ(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See LKJ for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct LKJ distributions.

#### Args:


* <b>`dimension`</b>: Python `int`. The dimension of the correlation matrices
  to sample.
* <b>`concentration`</b>: `float` or `double` `Tensor`. The positive concentration
  parameter of the LKJ distributions. The pdf of a sample matrix `X` is
  proportional to `det(X) ** (concentration - 1)`.
* <b>`input_output_cholesky`</b>: Python `bool`. If `True`, functions whose input or
  output have the semantics of samples assume inputs are in Cholesky form
  and return outputs in Cholesky form. In particular, if this flag is
  `True`, input to `log_prob` is presumed of Cholesky form and output from
  `sample` is of Cholesky form.  Setting this argument to `True` is purely
  a computational optimization and does not change the underlying
  distribution. Additionally, validation checks which are only defined on
  the multiplied-out form are omitted, even if `validate_args` is `True`.
  Default value: `False` (i.e., input/output does not have Cholesky
  semantics). WARNING: Do not set this boolean to true, when using
  <a href="../../tfp/mcmc.md"><code>tfp.mcmc</code></a>. The density is not the density of Cholesky factors of
  correlation matrices drawn via LKJ.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value `NaN` to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:


* <b>`ValueError`</b>: If `dimension` is negative.