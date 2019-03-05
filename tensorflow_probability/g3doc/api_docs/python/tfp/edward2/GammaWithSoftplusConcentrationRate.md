<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.GammaWithSoftplusConcentrationRate" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.GammaWithSoftplusConcentrationRate

``` python
tfp.edward2.GammaWithSoftplusConcentrationRate(
    *args,
    **kwargs
)
```

Create a random variable for GammaWithSoftplusConcentrationRate.

See GammaWithSoftplusConcentrationRate for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
Use `tfd.Gamma(tf.nn.softplus(concentration), tf.nn.softplus(rate))` instead.