<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.distributions.RegisterKL" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfp.experimental.substrates.jax.distributions.RegisterKL


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/distributions/kullback_leibler.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `RegisterKL`

Decorator to register a KL divergence implementation function.



<!-- Placeholder for "Used in" -->


#### Usage:



@distributions.RegisterKL(distributions.Normal, distributions.Normal)
def _kl_normal_mvn(norm_a, norm_b):
  # Return KL(norm_a || norm_b)

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/distributions/kullback_leibler.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/distributions/kullback_leibler.py">View source</a>

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



