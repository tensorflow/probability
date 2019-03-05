<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.ExponentialWithSoftplusRate" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.ExponentialWithSoftplusRate

``` python
tfp.edward2.ExponentialWithSoftplusRate(
    *args,
    **kwargs
)
```

Create a random variable for ExponentialWithSoftplusRate.

See ExponentialWithSoftplusRate for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
Use `tfd.Exponential(tf.nn.softplus(rate)).