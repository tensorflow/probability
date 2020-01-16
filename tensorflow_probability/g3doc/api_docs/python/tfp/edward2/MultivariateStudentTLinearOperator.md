<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MultivariateStudentTLinearOperator" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MultivariateStudentTLinearOperator


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for MultivariateStudentTLinearOperator.

### Aliases:

* `tfp.experimental.edward2.MultivariateStudentTLinearOperator`


``` python
tfp.edward2.MultivariateStudentTLinearOperator(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See MultivariateStudentTLinearOperator for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct Multivariate Student's t-distribution on `R^k`.

The `batch_shape` is the broadcast shape between `df`, `loc` and `scale`
arguments.

The `event_shape` is given by last dimension of the matrix implied by
`scale`. The last dimension of `loc` must broadcast with this.

Additional leading dimensions (if any) will index batches.

#### Args:


* <b>`df`</b>: A positive floating-point `Tensor`. Has shape `[B1, ..., Bb]` where `b
  >= 0`.
* <b>`loc`</b>: Floating-point `Tensor`. Has shape `[B1, ..., Bb, k]` where `k` is
  the event size.
* <b>`scale`</b>: Instance of `LinearOperator` with a floating `dtype` and shape
  `[B1, ..., Bb, k, k]`.
* <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
  with asserts. If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. If `False`, raise an
  exception if a statistic (e.g. mean/variance/etc...) is undefined for
  any batch member If `True`, batch members with valid parameters leading
  to undefined statistics will return NaN for this statistic.
* <b>`name`</b>: The name to give Ops created by the initializer.


#### Raises:


* <b>`TypeError`</b>: if not `scale.dtype.is_floating`.
* <b>`ValueError`</b>: if not `scale.is_positive_definite`.