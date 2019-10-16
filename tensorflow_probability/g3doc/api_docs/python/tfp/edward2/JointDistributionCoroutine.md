<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.JointDistributionCoroutine" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.JointDistributionCoroutine


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for JointDistributionCoroutine.

### Aliases:

* `tfp.experimental.edward2.JointDistributionCoroutine`


``` python
tfp.edward2.JointDistributionCoroutine(
    *args,
    **kwargs
)
```



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
  Default value: `None` (i.e., `JointDistributionCoroutine`).