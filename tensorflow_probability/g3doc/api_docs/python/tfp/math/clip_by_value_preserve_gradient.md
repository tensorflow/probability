<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.clip_by_value_preserve_gradient" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.clip_by_value_preserve_gradient

Clips values to a specified min and max while leaving gradient unaltered.

``` python
tfp.math.clip_by_value_preserve_gradient(
    t,
    clip_value_min,
    clip_value_max,
    name=None
)
```



Defined in [`python/math/numeric.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/numeric.py).

<!-- Placeholder for "Used in" -->

Like `tf.clip_by_value`, this function returns a tensor of the same type and
shape as input `t` but with values clamped to be no smaller than to
`clip_value_min` and no larger than `clip_value_max`. Unlike
`tf.clip_by_value`, the gradient is unaffected by this op, i.e.,

```python
tf.gradients(tfp.math.clip_by_value_preserve_gradient(x), x)[0]
# ==> ones_like(x)
```

Note: `clip_value_min` needs to be smaller or equal to `clip_value_max` for
correct results.

#### Args:


* <b>`t`</b>: A `Tensor`.
* <b>`clip_value_min`</b>: A scalar `Tensor`, or a `Tensor` with the same shape
  as `t`. The minimum value to clip by.
* <b>`clip_value_max`</b>: A scalar `Tensor`, or a `Tensor` with the same shape
  as `t`. The maximum value to clip by.
* <b>`name`</b>: A name for the operation (optional).
  Default value: `'clip_by_value_preserve_gradient'`.


#### Returns:


* <b>`clipped_t`</b>: A clipped `Tensor`.