<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.BlockwiseInitializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="initializers"/>
<meta itemprop="property" content="sizes"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tfp.layers.BlockwiseInitializer

## Class `BlockwiseInitializer`

Initializer which concats other intializers.



### Aliases:

* Class `tfp.layers.BlockwiseInitializer`
* Class `tfp.layers.initializers.BlockwiseInitializer`



Defined in [`python/layers/initializers.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/initializers.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    initializers,
    sizes,
    validate_args=False
)
```

Creates the `BlockwiseInitializer`.


#### Arguments:


* <b>`initializers`</b>: `list` of Keras initializers, e.g., `"glorot_uniform"` or
  `tf.keras.initializers.Constant(0.5413)`.
* <b>`sizes`</b>: `list` of `int` scalars representing the number of elements
  associated with each initializer in `initializers`.
* <b>`validate_args`</b>: Python `bool` indicating we should do (possibly expensive)
  graph-time assertions, if necessary.



## Properties

<h3 id="initializers"><code>initializers</code></h3>




<h3 id="sizes"><code>sizes</code></h3>




<h3 id="validate_args"><code>validate_args</code></h3>






## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    shape,
    dtype=None
)
```

Returns a tensor object initialized as specified by the initializer.


#### Args:


* <b>`shape`</b>: Shape of the tensor.
* <b>`dtype`</b>: Optional dtype of the tensor. If not provided will return tensor
 of `tf.float32`.

<h3 id="from_config"><code>from_config</code></h3>

``` python
@classmethod
from_config(
    cls,
    config
)
```

Instantiates an initializer from a configuration dictionary.


<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```

Returns initializer configuration as a JSON-serializable dict.




