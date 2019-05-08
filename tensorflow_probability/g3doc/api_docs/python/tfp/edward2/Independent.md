<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Independent" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Independent

Create a random variable for Independent.

``` python
tfp.edward2.Independent(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Independent for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct a `Independent` distribution.


#### Args:

* <b>`distribution`</b>: The base distribution instance to transform. Typically an
  instance of `Distribution`.
* <b>`reinterpreted_batch_ndims`</b>: Scalar, integer number of rightmost batch dims
  which will be regarded as event dims. When `None` all but the first
  batch axis (batch axis 0) will be transferred to event dimensions
  (analogous to `tf.layers.flatten`).
* <b>`validate_args`</b>: Python `bool`.  Whether to validate input with asserts.
  If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
* <b>`name`</b>: The name for ops managed by the distribution.
  Default value: `Independent + distribution.name`.


#### Raises:

* <b>`ValueError`</b>: if `reinterpreted_batch_ndims` exceeds
  `distribution.batch_ndims`