<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.log_sub_exp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.log_sub_exp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Compute `log(exp(max(x, y)) - exp(min(x, y)))` in a numerically stable way.

``` python
tfp.math.log_sub_exp(
    x,
    y,
    return_sign=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Use `return_sign=True` unless `x >= y`, since we can't represent a negative in
log-space.

#### Args:


* <b>`x`</b>: Float `Tensor` broadcastable with `y`.
* <b>`y`</b>: Float `Tensor` broadcastable with `x`.
* <b>`return_sign`</b>: Whether or not to return the second output value `sign`. If
  it is known that `x >= y`, this is unnecessary.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `'log_sub_exp'`).


#### Returns:


* <b>`logsubexp`</b>: Float `Tensor` of `log(exp(max(x, y)) - exp(min(x, y)))`.
* <b>`sign`</b>: Float `Tensor` +/-1 indicating the sign of `exp(x) - exp(y)`.