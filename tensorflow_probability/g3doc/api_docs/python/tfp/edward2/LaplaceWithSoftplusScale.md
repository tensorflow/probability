<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.LaplaceWithSoftplusScale" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.LaplaceWithSoftplusScale

``` python
tfp.edward2.LaplaceWithSoftplusScale(
    *args,
    **kwargs
)
```

Create a random variable for LaplaceWithSoftplusScale.

See LaplaceWithSoftplusScale for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
Use `tfd.Laplace(loc, tf.nn.softplus(scale)) instead.