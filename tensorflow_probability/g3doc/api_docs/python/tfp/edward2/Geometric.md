Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Geometric" />
</div>

# tfp.edward2.Geometric

``` python
tfp.edward2.Geometric(
    *args,
    **kwargs
)
```

Create a random variable for Geometric.

See Geometric for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct Geometric distributions.


#### Args:

* <b>`logits`</b>: Floating-point `Tensor` with shape `[B1, ..., Bb]` where `b >= 0`
    indicates the number of batch dimensions. Each entry represents logits
    for the probability of success for independent Geometric distributions
    and must be in the range `(-inf, inf]`. Only one of `logits` or `probs`
    should be specified.
* <b>`probs`</b>: Positive floating-point `Tensor` with shape `[B1, ..., Bb]`
    where `b >= 0` indicates the number of batch dimensions. Each entry
    represents the probability of success for independent Geometric
    distributions and must be in the range `(0, 1]`. Only one of `logits`
    or `probs` should be specified.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.