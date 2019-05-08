<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.JointDistributionCoroutine" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.JointDistributionCoroutine

Create a random variable for JointDistributionCoroutine.

``` python
tfp.edward2.JointDistributionCoroutine(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See JointDistributionCoroutine for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Construct the `JointDistributionCoroutine` distribution.


#### Args:

* <b>`model`</b>: A generator that yields a sequence of `tfd.Distribution`-like
  instances.
* <b>`sample_dtype`</b>: Samples from this distribution will be structured like
  `tf.nest.pack_sequence_as(sample_dtype, list_)`. `sample_dtype` is only
  used for `tf.nest.pack_sequence_as` structuring of outputs, never
  casting (which is the responsibility of the component distributions).
  Default value: `None` (i.e., `tuple`).
* <b>`validate_args`</b>: Python `bool`.  Whether to validate input with asserts.
  If `validate_args` is `False`, and the inputs are invalid,
  correct behavior is not guaranteed.
  Default value: `False`.
* <b>`name`</b>: The name for ops managed by the distribution.
  Default value: `None` (i.e., `"JointDistributionCoroutine"`).