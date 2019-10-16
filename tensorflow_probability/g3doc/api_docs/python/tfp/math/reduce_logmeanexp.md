<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.reduce_logmeanexp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.reduce_logmeanexp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes `log(mean(exp(input_tensor)))`.

``` python
tfp.math.reduce_logmeanexp(
    input_tensor,
    axis=None,
    keepdims=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Reduces `input_tensor` along the dimensions given in `axis`.  Unless
`keepdims` is true, the rank of the tensor is reduced by 1 for each entry in
`axis`. If `keepdims` is true, the reduced dimensions are retained with length
1.

If `axis` has no entries, all dimensions are reduced, and a tensor with a
single element is returned.

This function is more numerically stable than `log(reduce_mean(exp(input)))`.
It avoids overflows caused by taking the exp of large inputs and underflows
caused by taking the log of small inputs.

#### Args:


* <b>`input_tensor`</b>: The tensor to reduce. Should have numeric type.
* <b>`axis`</b>: The dimensions to reduce. If `None` (the default), reduces all
  dimensions. Must be in the range `[-rank(input_tensor),
  rank(input_tensor))`.
* <b>`keepdims`</b>:  Boolean.  Whether to keep the axis as singleton dimensions.
  Default value: `False` (i.e., squeeze the reduced dimensions).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., `'reduce_logmeanexp'`).


#### Returns:


* <b>`log_mean_exp`</b>: The reduced tensor.