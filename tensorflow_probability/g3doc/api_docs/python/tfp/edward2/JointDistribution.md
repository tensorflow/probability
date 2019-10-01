<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.JointDistribution" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.JointDistribution


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for JointDistribution.

### Aliases:

* `tfp.experimental.edward2.JointDistribution`


``` python
tfp.edward2.JointDistribution(
    *args,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

See JointDistribution for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Constructs the `Distribution`.

**This is a private method for subclass use.**

#### Args:


* <b>`dtype`</b>: The type of the event samples. `None` implies no type-enforcement.
* <b>`reparameterization_type`</b>: Instance of `ReparameterizationType`.
  If `tfd.FULLY_REPARAMETERIZED`, then samples from the distribution are
  fully reparameterized, and straight-through gradients are supported.
  If `tfd.NOT_REPARAMETERIZED`, then samples from the distribution are not
  fully reparameterized, and straight-through gradients are either
  partially unsupported or are not supported at all.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`parameters`</b>: Python `dict` of parameters used to instantiate this
  `Distribution`.
* <b>`graph_parents`</b>: Python `list` of graph prerequisites of this
  `Distribution`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class. Default:
  subclass name.


#### Raises:


* <b>`ValueError`</b>: if any member of graph_parents is `None` or not a `Tensor`.