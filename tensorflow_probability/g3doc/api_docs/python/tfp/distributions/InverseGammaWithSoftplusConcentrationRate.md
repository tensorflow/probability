<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.InverseGammaWithSoftplusConcentrationRate" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.distributions.InverseGammaWithSoftplusConcentrationRate

`InverseGamma` with softplus of `concentration` and `scale`. (deprecated)

``` python
tfp.distributions.InverseGammaWithSoftplusConcentrationRate(
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-06-05.
Instructions for updating:
InverseGammaWithSoftplusConcentrationRate is deprecated, use InverseGamma(concentration=tf.nn.softplus(concentration), scale=tf.nn.softplus(scale)) instead.