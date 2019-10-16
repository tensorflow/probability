<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.default_multivariate_normal_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.layers.default_multivariate_normal_fn


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/layers/util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Creates multivariate standard `Normal` distribution.

### Aliases:

* `tfp.layers.util.default_multivariate_normal_fn`


``` python
tfp.layers.default_multivariate_normal_fn(
    dtype,
    shape,
    name,
    trainable,
    add_variable_fn
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`dtype`</b>: Type of parameter's event.
* <b>`shape`</b>: Python `list`-like representing the parameter's event shape.
* <b>`name`</b>: Python `str` name prepended to any created (or existing)
  `tf.Variable`s.
* <b>`trainable`</b>: Python `bool` indicating all created `tf.Variable`s should be
  added to the graph collection `GraphKeys.TRAINABLE_VARIABLES`.
* <b>`add_variable_fn`</b>: `tf.get_variable`-like `callable` used to create (or
  access existing) `tf.Variable`s.


#### Returns:

Multivariate standard `Normal` distribution.
