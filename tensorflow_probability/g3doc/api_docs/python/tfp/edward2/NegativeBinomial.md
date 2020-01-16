<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.NegativeBinomial" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.NegativeBinomial


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for NegativeBinomial.

### Aliases:

* `tfp.experimental.edward2.NegativeBinomial`


``` python
tfp.edward2.NegativeBinomial(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See NegativeBinomial for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct NegativeBinomial distributions.

#### Args:


* <b>`total_count`</b>: Non-negative floating-point `Tensor` with shape
  broadcastable to `[B1,..., Bb]` with `b >= 0` and the same dtype as
  `probs` or `logits`. Defines this as a batch of `N1 x ... x Nm`
  different Negative Binomial distributions. In practice, this represents
  the number of negative Bernoulli trials to stop at (the `total_count`
  of failures). Its components should be equal to integer values.
* <b>`logits`</b>: Floating-point `Tensor` with shape broadcastable to
  `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
  Each entry represents logits for the probability of success for
  independent Negative Binomial distributions and must be in the open
  interval `(-inf, inf)`. Only one of `logits` or `probs` should be
  specified.
* <b>`probs`</b>: Positive floating-point `Tensor` with shape broadcastable to
  `[B1, ..., Bb]` where `b >= 0` indicates the number of batch dimensions.
  Each entry represents the probability of success for independent
  Negative Binomial distributions and must be in the open interval
  `(0, 1)`. Only one of `logits` or `probs` should be specified.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.