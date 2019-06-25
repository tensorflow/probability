<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.reduce_weighted_logsumexp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.reduce_weighted_logsumexp

Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

``` python
tfp.math.reduce_weighted_logsumexp(
    logx,
    w=None,
    axis=None,
    keep_dims=False,
    return_sign=False,
    name=None
)
```



Defined in [`python/math/generic.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/generic.py).

<!-- Placeholder for "Used in" -->

If all weights `w` are known to be positive, it is more efficient to directly
use `reduce_logsumexp`, i.e., `tf.reduce_logsumexp(logx + tf.log(w))` is more
efficient than `du.reduce_weighted_logsumexp(logx, w)`.

Reduces `input_tensor` along the dimensions given in `axis`.
Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
entry in `axis`. If `keep_dims` is true, the reduced dimensions
are retained with length 1.

If `axis` has no entries, all dimensions are reduced, and a
tensor with a single element is returned.

This function is more numerically stable than log(sum(w * exp(input))). It
avoids overflows caused by taking the exp of large inputs and underflows
caused by taking the log of small inputs.

#### For example:



```python
x = tf.constant([[0., 0, 0],
                 [0, 0, 0]])

w = tf.constant([[-1., 1, 1],
                 [1, 1, 1]])

du.reduce_weighted_logsumexp(x, w)
# ==> log(-1*1 + 1*1 + 1*1 + 1*1 + 1*1 + 1*1) = log(4)

du.reduce_weighted_logsumexp(x, w, axis=0)
# ==> [log(-1+1), log(1+1), log(1+1)]

du.reduce_weighted_logsumexp(x, w, axis=1)
# ==> [log(-1+1+1), log(1+1+1)]

du.reduce_weighted_logsumexp(x, w, axis=1, keep_dims=True)
# ==> [[log(-1+1+1)], [log(1+1+1)]]

du.reduce_weighted_logsumexp(x, w, axis=[0, 1])
# ==> log(-1+5)
```

#### Args:


* <b>`logx`</b>: The tensor to reduce. Should have numeric type.
* <b>`w`</b>: The weight tensor. Should have numeric type identical to `logx`.
* <b>`axis`</b>: The dimensions to reduce. If `None` (the default), reduces all
  dimensions. Must be in the range `[-rank(input_tensor),
  rank(input_tensor))`.
* <b>`keep_dims`</b>: If true, retains reduced dimensions with length 1.
* <b>`return_sign`</b>: If `True`, returns the sign of the result.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:


* <b>`lswe`</b>: The `log(abs(sum(weight * exp(x))))` reduced tensor.
* <b>`sign`</b>: (Optional) The sign of `sum(weight * exp(x))`.