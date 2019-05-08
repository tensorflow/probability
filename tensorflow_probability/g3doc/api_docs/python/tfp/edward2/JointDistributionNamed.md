<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.JointDistributionNamed" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.JointDistributionNamed

Create a random variable for JointDistributionNamed.

``` python
tfp.edward2.JointDistributionNamed(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See JointDistributionNamed for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct the `JointDistributionNamed` distribution.


#### Args:

* <b>`model`</b>: Python `dict` or `namedtuple` of distribution-making functions each
  with required args corresponding only to other keys.
* <b>`validate_args`</b>: Python `bool`.  Whether to validate input with asserts.
  If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
  Default value: `False`.
* <b>`name`</b>: The name for ops managed by the distribution.
  Default value: `None` (i.e., `"JointDistributionNamed"`).