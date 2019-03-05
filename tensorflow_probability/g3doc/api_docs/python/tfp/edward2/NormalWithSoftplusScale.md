<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.NormalWithSoftplusScale" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.NormalWithSoftplusScale

``` python
tfp.edward2.NormalWithSoftplusScale(
    *args,
    **kwargs
)
```

Create a random variable for NormalWithSoftplusScale.

See NormalWithSoftplusScale for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
Use `tfd.Normal(loc, tf.nn.softplus(scale)) instead.