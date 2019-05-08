<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.RegisterKL" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfp.distributions.RegisterKL

## Class `RegisterKL`

Decorator to register a KL divergence implementation function.



### Aliases:

* Class `tfp.distributions.RegisterKL`
* Class `tfp.layers.dense_variational_v2.kullback_leibler.RegisterKL`



Defined in [`python/distributions/kullback_leibler.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/kullback_leibler.py).

<!-- Placeholder for "Used in" -->

#### Usage:


@distributions.RegisterKL(distributions.Normal, distributions.Normal)
def _kl_normal_mvn(norm_a, norm_b):
  # Return KL(norm_a || norm_b)

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    dist_cls_a,
    dist_cls_b
)
```

Initialize the KL registrar.

#### Args:

* <b>`dist_cls_a`</b>: the class of the first argument of the KL divergence.
* <b>`dist_cls_b`</b>: the class of the second argument of the KL divergence.



## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(kl_fn)
```

Perform the KL registration.

#### Args:

* <b>`kl_fn`</b>: The function to use for the KL divergence.


#### Returns:

kl_fn


#### Raises:

* <b>`TypeError`</b>: if kl_fn is not a callable.
* <b>`ValueError`</b>: if a KL divergence function has already been registered for
  the given argument classes.



