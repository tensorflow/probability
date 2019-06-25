<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.AutoregressiveNetwork" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="input_spec"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="params"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_input_at"/>
<meta itemprop="property" content="get_input_mask_at"/>
<meta itemprop="property" content="get_input_shape_at"/>
<meta itemprop="property" content="get_losses_for"/>
<meta itemprop="property" content="get_output_at"/>
<meta itemprop="property" content="get_output_mask_at"/>
<meta itemprop="property" content="get_output_shape_at"/>
<meta itemprop="property" content="get_updates_for"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfp.bijectors.AutoregressiveNetwork

## Class `AutoregressiveNetwork`

Masked Autoencoder for Distribution Estimation [Germain et al. (2015)][1].





Defined in [`python/bijectors/masked_autoregressive.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors/masked_autoregressive.py).

<!-- Placeholder for "Used in" -->

A `AutoregressiveNetwork` takes as input a Tensor of shape `[..., event_size]`
and returns a Tensor of shape `[..., event_size, params]`.

The output satisfies the autoregressive property.  That is, the layer is
configured with some permutation `ord` of `{0, ..., event_size-1}` (i.e., an
ordering of the input dimensions), and the output `output[batch_idx, i, ...]`
for input dimension `i` depends only on inputs `x[batch_idx, j]` where
`ord(j) < ord(i)`.  The autoregressive property allows us to use
`output[batch_idx, i]` to parameterize conditional distributions:
  `p(x[batch_idx, i] | x[batch_idx, ] for ord(j) < ord(i))`
which give us a tractable distribution over input `x[batch_idx]`:
  `p(x[batch_idx]) = prod_i p(x[batch_idx, ord(i)] | x[batch_idx, ord(0:i)])`

For example, when `params` is 2, the output of the layer can parameterize
the location and log-scale of an autoregressive Gaussian distribution.

#### Example

```python
# Generate data -- as in Figure 1 in [Papamakarios et al. (2017)][2]).
n = 2000
x2 = np.random.randn(n).astype(dtype=np.float32) * 2.
x1 = np.random.randn(n).astype(dtype=np.float32) + (x2 * x2 / 4.)
data = np.stack([x1, x2], axis=-1)

# Density estimation with MADE.
made = tfb.AutoregressiveNetwork(params=2, hidden_units=[10, 10])

distribution = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.MaskedAutoregressiveFlow(
        lambda x: tf.unstack(made(x), num=2, axis=-1)),
    event_shape=[2])

# Construct and fit model.
x_ = tfkl.Input(shape=(2,), dtype=tf.float32)
log_prob_ = distribution.log_prob(x_)
model = tfk.Model(x_, log_prob_)

model.compile(optimizer=tf.compat.v2.optimizers.Adam(),
              loss=lambda _, log_prob: -log_prob)

batch_size = 25
model.fit(x=data,
          y=np.zeros((n, 0), dtype=np.float32),
          batch_size=batch_size,
          epochs=1,
          steps_per_epoch=1,  # Usually `n // batch_size`.
          shuffle=True,
          verbose=True)

# Use the fitted distribution.
distribution.sample((3, 1))
distribution.log_prob(np.ones((3, 2), dtype=np.float32))
```

#### Examples: Handling Rank-2+ Tensors

`AutoregressiveNetwork` can be used as a building block to achieve different
autoregressive structures over rank-2+ tensors.  For example, suppose we want
to build an autoregressive distribution over images with dimension `[weight,
height, channels]` with `channels = 3`:

 1. We can parameterize a "fully autoregressive" distribution, with
    cross-channel and within-pixel autoregressivity:
    ```
        r0    g0   b0     r0    g0   b0       r0   g0    b0
        ^   ^      ^         ^   ^   ^         ^      ^   ^
        |  /  ____/           \  |  /           \____  \  |
        | /__/                 \ | /                 \__\ |
        r1    g1   b1     r1 <- g1   b1       r1   g1 <- b1
                                             ^          |
                                              \_________/
    ```

    as:
    ```python
    # Generate random images for training data.
    images = np.random.uniform(size=(100, 8, 8, 3)).astype(np.float32)
    n, width, height, channels = images.shape

    # Reshape images to achieve desired autoregressivity.
    event_shape = [height * width * channels]
    reshaped_images = tf.reshape(images, [n, event_shape])

    # Density estimatino with MADE.
    made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape,
                                     hidden_units=[20, 20], activation="relu")
    distribution = tfd.TransformedDistribution(
        distribution=tfd.Normal(loc=0., scale=1.),
        bijector=tfb.MaskedAutoregressiveFlow(
            lambda x: tf.unstack(made(x), num=2, axis=-1)),
        event_shape=event_shape)

    # Construct and fit model.
    x_ = tfkl.Input(shape=event_shape, dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.compat.v2.optimizers.Adam(),
                  loss=lambda _, log_prob: -log_prob)

    batch_size = 10
    model.fit(x=data,
              y=np.zeros((n, 0), dtype=np.float32),
              batch_size=batch_size,
              epochs=10,
              steps_per_epoch=n // batch_size,
              shuffle=True,
              verbose=True)

    # Use the fitted distribution.
    distribution.sample((3, 1))
    distribution.log_prob(np.ones((5, 8, 8, 3), dtype=np.float32))
    ```

 2. We can parameterize a distribution with neither cross-channel nor
    within-pixel autoregressivity:
    ```
        r0    g0   b0
        ^     ^    ^
        |     |    |
        |     |    |
        r1    g1   b1
    ```

    as:
    ```python
    # Generate fake images.
    images = np.random.choice([0, 1], size=(100, 8, 8, 3))
    n, width, height, channels = images.shape

    # Reshape images to achieve desired autoregressivity.
    reshaped_images = np.transpose(
        np.reshape(images, [n, width * height, channels]),
        axes=[0, 2, 1])

    made = tfb.AutoregressiveNetwork(params=1, event_shape=[width * height],
                                     hidden_units=[20, 20], activation="relu")

    # Density estimation with MADE.
    #
    # NOTE: Parameterize an autoregressive distribution over an event_shape of
    # [channels, width * height], with univariate Bernoulli conditional
    # distributions.
    distribution = tfd.Autoregressive(
        lambda x: tfd.Independent(
            tfd.Bernoulli(logits=tf.unstack(made(x), axis=-1)[0],
                          dtype=tf.float32),
            reinterpreted_batch_ndims=2),
        sample0=tf.zeros([channels, width * height], dtype=tf.float32))

    # Construct and fit model.
    x_ = tfkl.Input(shape=(channels, width * height), dtype=tf.float32)
    log_prob_ = distribution.log_prob(x_)
    model = tfk.Model(x_, log_prob_)

    model.compile(optimizer=tf.compat.v2.optimizers.Adam(),
                  loss=lambda _, log_prob: -log_prob)

    batch_size = 10
    model.fit(x=reshaped_images,
              y=np.zeros((n, 0), dtype=np.float32),
              batch_size=batch_size,
              epochs=10,
              steps_per_epoch=n // batch_size,
              shuffle=True,
              verbose=True)

    distribution.sample(7)
    distribution.log_prob(np.ones((4, 8, 8, 3), dtype=np.float32))
    ```

    Note that one set of weights is shared for the mapping for each channel
    from image to distribution parameters -- i.e., the mapping
    `layer(reshaped_images[..., channel, :])`, where `channel` is 0, 1, or 2.

    To use separate weights for each channel, we could construct an
    `AutoregressiveNetwork` and `TransformedDistribution` for each channel,
    and combine them with a `tfd.Blockwise` distribution.

#### References

[1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
     Masked Autoencoder for Distribution Estimation. In _International
     Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509

[2]: George Papamakarios, Theo Pavlakou, Iain Murray, Masked Autoregressive
     Flow for Density Estimation.  In _Neural Information Processing Systems_,
     2017. https://arxiv.org/abs/1705.07057

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    params,
    event_shape=None,
    hidden_units=None,
    input_order='left-to-right',
    hidden_degrees='equal',
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    validate_args=False,
    **kwargs
)
```

Constructs the MADE layer.


#### Arguments:


* <b>`params`</b>: Python integer specifying the number of parameters to output
  per input.
* <b>`event_shape`</b>: Python `list`-like of positive integers (or a single int),
  specifying the shape of the input to this layer, which is also the
  event_shape of the distribution parameterized by this layer.  Currently
  only rank-1 shapes are supported.  That is, event_shape must be a single
  integer.  If not specified, the event shape is inferred when this layer
  is first called or built.
* <b>`hidden_units`</b>: Python `list`-like of non-negative integers, specifying
  the number of units in each hidden layer.
* <b>`input_order`</b>: Order of degrees to the input units: 'random',
  'left-to-right', 'right-to-left', or an array of an explicit order. For
  example, 'left-to-right' builds an autoregressive model:
  `p(x) = p(x1) p(x2 | x1) ... p(xD | x<D)`.  Default: 'left-to-right'.
* <b>`hidden_degrees`</b>: Method for assigning degrees to the hidden units:
  'equal', 'random'.  If 'equal', hidden units in each layer are allocated
  equally (up to a remainder term) to each degree.  Default: 'equal'.
* <b>`activation`</b>: An activation function.  See `tf.keras.layers.Dense`. Default:
  `None`.
* <b>`use_bias`</b>: Whether or not the dense layers constructed in this layer
  should have a bias term.  See `tf.keras.layers.Dense`.  Default: `True`.
* <b>`kernel_initializer`</b>: Initializer for the `Dense` kernel weight
  matrices.  Default: 'glorot_uniform'.
* <b>`bias_initializer`</b>: Initializer for the `Dense` bias vectors. Default:
  'zeros'.
* <b>`kernel_regularizer`</b>: Regularizer function applied to the `Dense` kernel
  weight matrices.  Default: None.
* <b>`bias_regularizer`</b>: Regularizer function applied to the `Dense` bias
  weight vectors.  Default: None.
* <b>`kernel_constraint`</b>: Constraint function applied to the `Dense` kernel
  weight matrices.  Default: None.
* <b>`bias_constraint`</b>: Constraint function applied to the `Dense` bias
  weight vectors.  Default: None.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True`, layer
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`**kwargs`</b>: Additional keyword arguments passed to this layer (but not to
  the `tf.keras.layer.Dense` layers constructed by this layer).



## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.


<h3 id="dtype"><code>dtype</code></h3>




<h3 id="dynamic"><code>dynamic</code></h3>




<h3 id="event_shape"><code>event_shape</code></h3>




<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.
* <b>`AttributeError`</b>: If no inbound nodes are found.

<h3 id="input_mask"><code>input_mask</code></h3>

Retrieves the input mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Input mask tensor (potentially None) or list of input
mask tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.

<h3 id="input_shape"><code>input_shape</code></h3>

Retrieves the input shape(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer, or if all inputs
have the same shape.

#### Returns:

Input shape, as an integer shape tuple
(or list of shape tuples, one tuple per input tensor).



#### Raises:


* <b>`AttributeError`</b>: if the layer has no defined input_shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="input_spec"><code>input_spec</code></h3>




<h3 id="losses"><code>losses</code></h3>

Losses which are associated with this `Layer`.

Variable regularization tensors are created when this property is accessed,
so it is eager safe: accessing `losses` under a `tf.GradientTape` will
propagate gradients back to the corresponding variables.

#### Returns:

A list of tensors.


<h3 id="metrics"><code>metrics</code></h3>




<h3 id="name"><code>name</code></h3>

Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.

<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>




<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>




<h3 id="output"><code>output</code></h3>

Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.

#### Returns:

Output tensor or list of output tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to more than one incoming
  layers.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="output_mask"><code>output_mask</code></h3>

Retrieves the output mask tensor(s) of a layer.

Only applicable if the layer has exactly one inbound node,
i.e. if it is connected to one incoming layer.

#### Returns:

Output mask tensor (potentially None) or list of output
mask tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.

<h3 id="output_shape"><code>output_shape</code></h3>

Retrieves the output shape(s) of a layer.

Only applicable if the layer has one output,
or if all outputs have the same shape.

#### Returns:

Output shape, as an integer shape tuple
(or list of shape tuples, one tuple per output tensor).



#### Raises:


* <b>`AttributeError`</b>: if the layer has no defined output shape.
* <b>`RuntimeError`</b>: if called in Eager mode.

<h3 id="params"><code>params</code></h3>




<h3 id="submodules"><code>submodules</code></h3>

Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
a = tf.Module()
b = tf.Module()
c = tf.Module()
a.b = b
b.c = c
assert list(a.submodules) == [b, c]
assert list(b.submodules) == [c]
assert list(c.submodules) == []
```

#### Returns:

A sequence of all submodules.


<h3 id="trainable"><code>trainable</code></h3>




<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).


<h3 id="trainable_weights"><code>trainable_weights</code></h3>




<h3 id="updates"><code>updates</code></h3>




<h3 id="variables"><code>variables</code></h3>

Returns the list of all layer variables/weights.

Alias of `self.weights`.

#### Returns:

A list of variables.


<h3 id="weights"><code>weights</code></h3>

Returns the list of all layer variables/weights.


#### Returns:

A list of variables.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    inputs,
    *args,
    **kwargs
)
```

Wraps `call`, applying pre- and post-processing steps.


#### Arguments:


* <b>`inputs`</b>: input tensor(s).
* <b>`*args`</b>: additional positional arguments to be passed to `self.call`.
* <b>`**kwargs`</b>: additional keyword arguments to be passed to `self.call`.


#### Returns:

Output tensor(s).



#### Note:

- The following optional keyword arguments are reserved for specific uses:
  * `training`: Boolean scalar tensor of Python boolean indicating
    whether the `call` is meant for training or inference.
  * `mask`: Boolean input mask.
- If the layer's `call` method takes a `mask` argument (as some Keras
  layers do), its default value will be set to the mask generated
  for `inputs` by the previous layer (if `input` did come from
  a layer that generated a corresponding mask, i.e. if it came from
  a Keras layer with masking support.



#### Raises:


* <b>`ValueError`</b>: if the layer's `call` method returns None (an invalid value).

<h3 id="build"><code>build</code></h3>

``` python
build(input_shape)
```

See tfkl.Layer.build.


<h3 id="compute_mask"><code>compute_mask</code></h3>

``` python
compute_mask(
    inputs,
    mask=None
)
```

Computes an output mask tensor.


#### Arguments:


* <b>`inputs`</b>: Tensor or list of tensors.
* <b>`mask`</b>: Tensor or list of tensors.


#### Returns:

None or a tensor (or list of tensors,
    one per output tensor of the layer).


<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

``` python
compute_output_shape(input_shape)
```

See tfkl.Layer.compute_output_shape.


<h3 id="count_params"><code>count_params</code></h3>

``` python
count_params()
```

Count the total number of scalars composing the weights.


#### Returns:

An integer count.



#### Raises:


* <b>`ValueError`</b>: if the layer isn't yet built
  (in which case its weights aren't yet defined).

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config
)
```

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

#### Arguments:


* <b>`config`</b>: A Python dictionary, typically the
    output of get_config.


#### Returns:

A layer instance.


<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```

Returns the config of the layer.

A layer config is a Python dictionary (serializable)
containing the configuration of a layer.
The same layer can be reinstantiated later
(without its trained weights) from this configuration.

The config of a layer does not include connectivity
information, nor the layer class name. These are handled
by `Network` (one layer of abstraction above).

#### Returns:

Python dictionary.


<h3 id="get_input_at"><code>get_input_at</code></h3>

``` python
get_input_at(node_index)
```

Retrieves the input tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple inputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_input_mask_at"><code>get_input_mask_at</code></h3>

``` python
get_input_mask_at(node_index)
```

Retrieves the input mask tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple inputs).


<h3 id="get_input_shape_at"><code>get_input_shape_at</code></h3>

``` python
get_input_shape_at(node_index)
```

Retrieves the input shape(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple inputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_losses_for"><code>get_losses_for</code></h3>

``` python
get_losses_for(inputs)
```

Retrieves losses relevant to a specific set of inputs.


#### Arguments:


* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of loss tensors of the layer that depend on `inputs`.


<h3 id="get_output_at"><code>get_output_at</code></h3>

``` python
get_output_at(node_index)
```

Retrieves the output tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A tensor (or list of tensors if the layer has multiple outputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_output_mask_at"><code>get_output_mask_at</code></h3>

``` python
get_output_mask_at(node_index)
```

Retrieves the output mask tensor(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A mask tensor
(or list of tensors if the layer has multiple outputs).


<h3 id="get_output_shape_at"><code>get_output_shape_at</code></h3>

``` python
get_output_shape_at(node_index)
```

Retrieves the output shape(s) of a layer at a given node.


#### Arguments:


* <b>`node_index`</b>: Integer, index of the node
    from which to retrieve the attribute.
    E.g. `node_index=0` will correspond to the
    first time the layer was called.


#### Returns:

A shape tuple
(or list of shape tuples if the layer has multiple outputs).



#### Raises:


* <b>`RuntimeError`</b>: If called in Eager mode.

<h3 id="get_updates_for"><code>get_updates_for</code></h3>

``` python
get_updates_for(inputs)
```

Retrieves updates relevant to a specific set of inputs.


#### Arguments:


* <b>`inputs`</b>: Input tensor or list/tuple of input tensors.


#### Returns:

List of update ops of the layer that depend on `inputs`.


<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```

Returns the current weights of the layer.


#### Returns:

Weights values as a list of numpy arrays.


<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```

Sets the weights of the layer, from Numpy arrays.


#### Arguments:


* <b>`weights`</b>: a list of Numpy arrays. The number
    of arrays and their shape must match
    number of the dimensions of the weights
    of the layer (i.e. it should match the
    output of `get_weights`).


#### Raises:


* <b>`ValueError`</b>: If the provided weights list does not match the
    layer's specifications.

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

``` python
with_name_scope(
    cls,
    method
)
```

Decorator to automatically enter the module name scope.

```
class MyModule(tf.Module):
  @tf.Module.with_name_scope
  def __call__(self, x):
    if not hasattr(self, 'w'):
      self.w = tf.Variable(tf.random.normal([x.shape[1], 64]))
    return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
mod = MyModule()
mod(tf.ones([8, 32]))
# ==> <tf.Tensor: ...>
mod.w
# ==> <tf.Variable ...'my_module/w:0'>
```

#### Args:


* <b>`method`</b>: The method to wrap.


#### Returns:

The original method wrapped such that it enters the module's name scope.




