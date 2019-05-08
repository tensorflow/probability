<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.MultivariateNormalDiagWithSoftplusScale" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.MultivariateNormalDiagWithSoftplusScale

Create a random variable for MultivariateNormalDiagWithSoftplusScale.

``` python
tfp.edward2.MultivariateNormalDiagWithSoftplusScale(
    *args,
    **kwargs
)
```



Defined in [`python/edward2/interceptor.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/interceptor.py).

<!-- Placeholder for "Used in" -->

See MultivariateNormalDiagWithSoftplusScale for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

* <b>`Warning`</b>: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-06-05.
Instructions for updating:
MultivariateNormalDiagWithSoftplusScale is deprecated, use MultivariateNormalDiag(loc=loc, scale_diag=tf.nn.softplus(scale_diag)) instead.