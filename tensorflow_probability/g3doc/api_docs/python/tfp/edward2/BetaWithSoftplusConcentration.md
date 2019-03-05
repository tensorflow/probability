<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.BetaWithSoftplusConcentration" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.BetaWithSoftplusConcentration

``` python
tfp.edward2.BetaWithSoftplusConcentration(
    *args,
    **kwargs
)
```

Create a random variable for BetaWithSoftplusConcentration.

See BetaWithSoftplusConcentration for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
Use `tfd.Beta(tf.nn.softplus(concentration1), tf.nn.softplus(concentration2))` instead.