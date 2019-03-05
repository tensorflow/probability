<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.StudentTWithAbsDfSoftplusScale" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.StudentTWithAbsDfSoftplusScale

``` python
tfp.edward2.StudentTWithAbsDfSoftplusScale(
    *args,
    **kwargs
)
```

Create a random variable for StudentTWithAbsDfSoftplusScale.

See StudentTWithAbsDfSoftplusScale for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-01-01.
Instructions for updating:
Use `tfd.StudentT(tf.floor(tf.abs(df)), loc, tf.nn.softplus(scale)) instead.