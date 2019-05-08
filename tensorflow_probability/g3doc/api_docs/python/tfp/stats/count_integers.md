<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.stats.count_integers" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.stats.count_integers

Counts the number of occurrences of each value in an integer array `arr`.

``` python
tfp.stats.count_integers(
    arr,
    weights=None,
    minlength=None,
    maxlength=None,
    axis=None,
    dtype=tf.int32,
    name=None
)
```



Defined in [`python/stats/quantiles.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/stats/quantiles.py).

<!-- Placeholder for "Used in" -->

Works like `tf.math.bincount`, but provides an `axis` kwarg that specifies
dimensions to reduce over.  With
  `~axis = [i for i in range(arr.ndim) if i not in axis]`,
this function returns a `Tensor` of shape `[K] + arr.shape[~axis]`.

If `minlength` and `maxlength` are not given, `K = tf.reduce_max(arr) + 1`
if `arr` is non-empty, and 0 otherwise.
If `weights` are non-None, then index `i` of the output stores the sum of the
value in `weights` at each index where the corresponding value in `arr` is
`i`.

#### Args:

* <b>`arr`</b>: An `int32` `Tensor` of non-negative values.
* <b>`weights`</b>: If non-None, must be the same shape as arr. For each value in
  `arr`, the bin will be incremented by the corresponding weight instead of
  1.
* <b>`minlength`</b>: If given, ensures the output has length at least `minlength`,
  padding with zeros at the end if necessary.
* <b>`maxlength`</b>: If given, skips values in `arr` that are equal or greater than
  `maxlength`, ensuring that the output has length at most `maxlength`.
* <b>`axis`</b>: A `0-D` or `1-D` `int32` `Tensor` (with static values) designating
  dimensions in `arr` to reduce over.
  `Default value:` `None`, meaning reduce over all dimensions.
* <b>`dtype`</b>: If `weights` is None, determines the type of the output bins.
* <b>`name`</b>: A name scope for the associated operations (optional).


#### Returns:

A vector with the same dtype as `weights` or the given `dtype`. The bin
values.