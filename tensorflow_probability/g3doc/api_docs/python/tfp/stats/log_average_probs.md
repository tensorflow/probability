<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.log_average_probs" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.log_average_probs


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/stats/sample_stats.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes `log(average(to_probs(logits)))` in a numerically stable manner.

``` python
tfp.stats.log_average_probs(
    logits,
    sample_axis=0,
    event_axis=None,
    keepdims=False,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The meaning of `to_probs` is controlled by the `event_axis` argument. When
`event_axis` is `None`, `to_probs = tf.math.sigmoid` and otherwise
`to_probs = lambda x: tf.math.log_softmax(x, axis=event_axis)`.

`sample_axis` and `event_axis` should have a null intersection. This
requirement is always verified when `validate_args` is `True`.

#### Args:


* <b>`logits`</b>: A `float` `Tensor` representing logits.
* <b>`sample_axis`</b>: Scalar or vector `Tensor` designating axis holding samples, or
  `None` (meaning all axis hold samples).
  Default value: `0` (leftmost dimension).
* <b>`event_axis`</b>: Scalar or vector `Tensor` designating the axis representing
  categorical logits.
  Default value: `None` (i.e., Bernoulli logits).
* <b>`keepdims`</b>:  Boolean.  Whether to keep the sample axis as singletons.
  Default value: `False` (i.e., squeeze the reduced dimensions).
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
  Default value: `False` (i.e., do not validate args).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `'log_average_probs'`).


#### Returns:


* <b>`log_avg_probs`</b>: The natural log of the average of probs computed from logits.