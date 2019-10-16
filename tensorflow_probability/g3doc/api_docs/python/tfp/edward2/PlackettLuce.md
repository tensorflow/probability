<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.PlackettLuce" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.PlackettLuce


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for PlackettLuce.

### Aliases:

* `tfp.experimental.edward2.PlackettLuce`


``` python
tfp.edward2.PlackettLuce(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See PlackettLuce for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Initialize a batch of PlackettLuce distributions.

#### Args:


* <b>`scores`</b>: An N-D `Tensor`, `N >= 1`, representing the scores of a set of
  elements to be permuted. The first `N - 1` dimensions index into a
  batch of independent distributions and the last dimension represents a
  vector of scores for the elements.
* <b>`dtype`</b>: The type of the event samples (default: int32).
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.