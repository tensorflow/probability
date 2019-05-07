<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.ReparameterizationType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfp.distributions.ReparameterizationType

## Class `ReparameterizationType`

Instances of this class represent how sampling is reparameterized.





Defined in [`python/internal/reparameterization.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/reparameterization.py).

<!-- Placeholder for "Used in" -->

Two static instances exist in the distributions library, signifying
one of two possible properties for samples from a distribution:

`FULLY_REPARAMETERIZED`: Samples from the distribution are fully
  reparameterized, and straight-through gradients are supported.

`NOT_REPARAMETERIZED`: Samples from the distribution are not fully
  reparameterized, and straight-through gradients are either partially
  unsupported or are not supported at all. In this case, for purposes of
  e.g. RL or variational inference, it is generally safest to wrap the
  sample results in a `stop_gradients` call and use policy
  gradients / surrogate loss instead.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(rep_type)
```

Initialize self.  See help(type(self)) for accurate signature.



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Determine if this `ReparameterizationType` is equal to another.

Since RepaparameterizationType instances are constant static global
instances, equality checks if two instances' id() values are equal.

#### Args:

* <b>`other`</b>: Object to compare against.


#### Returns:

`self is other`.



