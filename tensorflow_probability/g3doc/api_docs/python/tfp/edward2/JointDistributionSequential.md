<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.JointDistributionSequential" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.JointDistributionSequential

Create a random variable for JointDistributionSequential.

``` python
tfp.edward2.JointDistributionSequential(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See JointDistributionSequential for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct the `JointDistributionSequential` distribution.


#### Args:

* <b>`model`</b>: Python list of either tfd.Distribution instances and/or
  lambda functions which take the `k` previous distributions and returns a
  new tfd.Distribution instance.
* <b>`validate_args`</b>: Python `bool`.  Whether to validate input with asserts.
  If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
  Default value: `False`.
* <b>`name`</b>: The name for ops managed by the distribution.
  Default value: `None` (i.e., `"JointDistributionSequential"`).