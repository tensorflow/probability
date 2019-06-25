<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.ConditionalTransformedDistribution" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.ConditionalTransformedDistribution

Create a random variable for ConditionalTransformedDistribution.

``` python
tfp.edward2.ConditionalTransformedDistribution(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See ConditionalTransformedDistribution for more details.

#### Returns:

RandomVariable.


#### Original Docstring for Distribution

Construct a Transformed Distribution.

#### Args:


* <b>`distribution`</b>: The base distribution instance to transform. Typically an
  instance of `Distribution`.
* <b>`bijector`</b>: The object responsible for calculating the transformation.
  Typically an instance of `Bijector`.
* <b>`batch_shape`</b>: `integer` vector `Tensor` which overrides `distribution`
  `batch_shape`; valid only if `distribution.is_scalar_batch()`.
* <b>`event_shape`</b>: `integer` vector `Tensor` which overrides `distribution`
  `event_shape`; valid only if `distribution.is_scalar_event()`.
* <b>`kwargs_split_fn`</b>: Python `callable` which takes a kwargs `dict` and returns
  a tuple of kwargs `dict`s for each of the `distribution` and `bijector`
  parameters respectively.
  Default value: `_default_kwargs_split_fn` (i.e.,
      `lambda kwargs: (kwargs.get('distribution_kwargs', {}),
                       kwargs.get('bijector_kwargs', {}))`)
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`parameters`</b>: Locals dict captured by subclass constructor, to be used for
  copy/slice re-instantiation operations.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class. Default:
  `bijector.name + distribution.name`.