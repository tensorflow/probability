<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.as_random_variable" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.as_random_variable

Wrap an existing distribution as a traceable random variable.

``` python
tfp.edward2.as_random_variable(
    distribution,
    sample_shape=(),
    value=None
)
```



Defined in [`python/edward2/generated_random_variables.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/generated_random_variables.py).

<!-- Placeholder for "Used in" -->

This enables the use of custom or user-provided distributions in
Edward models. Unlike a bare `RandomVariable` object, this method
wraps the constructor so it is included in the Edward trace and its
values can be properly intercepted and overridden.

Where possible, you should prefer the built-in constructors
(`ed.Normal`, etc); these simultaneously construct a Distribution
and a RandomVariable object so that the distribution parameters
themselves may be intercepted and overridden. RVs constructed via
`as_random_variable()` have a fixed distribution and may not support
program transformations (e.g, conjugate marginalization) that rely
on overriding distribution parameters.

#### Args:

* <b>`distribution`</b>: tfd.Distribution governing the distribution of the random
  variable, such as sampling and log-probabilities.
* <b>`sample_shape`</b>: tf.TensorShape of samples to draw from the random variable.
  Default is `()` corresponding to a single sample.
* <b>`value`</b>: Fixed tf.Tensor to associate with random variable. Must have shape
  `sample_shape + distribution.batch_shape + distribution.event_shape`.
  Default is to sample from random variable according to `sample_shape`.


#### Returns:

  rv: a `RandomVariable` wrapping the provided distribution.

#### Example

```python
from tensorflow_probability import distributions as tfd
from tensorflow_probability import edward2 as ed

def model():
  # equivalent to ed.Normal(0., 1., name='x')
  return ed.as_random_variable(tfd.Normal(0., 1., name='x'))

log_joint = ed.make_log_joint_fn(model)
output = log_joint(x=2.)
```