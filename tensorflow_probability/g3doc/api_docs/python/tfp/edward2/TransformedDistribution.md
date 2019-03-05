<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.TransformedDistribution" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.TransformedDistribution

``` python
tfp.edward2.TransformedDistribution(
    *args,
    **kwargs
)
```

Create a random variable for TransformedDistribution.

See TransformedDistribution for more details.

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
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class. Default:
    `bijector.name + distribution.name`.