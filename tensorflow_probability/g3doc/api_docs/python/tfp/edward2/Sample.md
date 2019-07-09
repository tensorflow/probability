<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.Sample" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.Sample

Create a random variable for Sample.

``` python
tfp.edward2.Sample(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See Sample for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct the `Sample` distribution.

#### Args:


* <b>`distribution`</b>: The base distribution instance to transform. Typically an
  instance of `Distribution`.
* <b>`sample_shape`</b>: `int` scalar or vector `Tensor` representing the shape of a
  single sample.
* <b>`validate_args`</b>: Python `bool`.  Whether to validate input with asserts.
  If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
* <b>`name`</b>: The name for ops managed by the distribution.
  Default value: `None` (i.e., `'Sample' + distribution.name`).