<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.math.log_add_exp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.numpy.math.log_add_exp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/numpy/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes `log(exp(x) + exp(y))` in a numerically stable way.

### Aliases:

* `tfp.experimental.substrates.numpy.math.generic.log_add_exp`


``` python
tfp.experimental.substrates.numpy.math.log_add_exp(
    x,
    y,
    name=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`x`</b>: `float` `Tensor` broadcastable with `y`.
* <b>`y`</b>: `float` `Tensor` broadcastable with `x`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `'log_add_exp'`).


#### Returns:


* <b>`log_add_exp`</b>: `log(exp(x) + exp(y))` computed in a numerically stable way.