<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.JointDistributionNamed" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.JointDistributionNamed


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/edward2/interceptor.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Create a random variable for JointDistributionNamed.

### Aliases:

* `tfp.experimental.edward2.JointDistributionNamed`


``` python
tfp.edward2.JointDistributionNamed(
    *args,
    **kwargs
)
```



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