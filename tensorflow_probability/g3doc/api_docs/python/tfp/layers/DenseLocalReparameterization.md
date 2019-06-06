<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.DenseLocalReparameterization" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="activity_regularizer"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="dynamic"/>
<meta itemprop="property" content="input"/>
<meta itemprop="property" content="input_mask"/>
<meta itemprop="property" content="input_shape"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="metrics"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="output"/>
<meta itemprop="property" content="output_mask"/>
<meta itemprop="property" content="output_shape"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply"/>
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

# tfp.layers.DenseLocalReparameterization

## Class `DenseLocalReparameterization`

Densely-connected layer class with local reparameterization estimator.



### Aliases:

* Class `tfp.layers.DenseLocalReparameterization`
* Class `tfp.layers.dense_variational.DenseLocalReparameterization`



Defined in [`python/layers/dense_variational.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/dense_variational.py).

<!-- Placeholder for "Used in" -->

This layer implements the Bayesian variational inference analogue to
a dense layer by assuming the `kernel` and/or the `bias` are drawn
from distributions. By default, the layer implements a stochastic
forward pass via sampling from the kernel and bias posteriors,

```none
kernel, bias ~ posterior
outputs = activation(matmul(inputs, kernel) + bias)
```

It uses the local reparameterization estimator [(Kingma et al., 2015)][1],
which performs a Monte Carlo approximation of the distribution on the hidden
units induced by the `kernel` and `bias`. The default `kernel_posterior_fn`
is a normal distribution which factorizes across all elements of the weight
matrix and bias vector. Unlike [1]'s multiplicative parameterization, this
distribution has trainable location and scale parameters which is known as
an additive noise parameterization [(Molchanov et al., 2017)][2].

The arguments permit separate specification of the surrogate posterior
(`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
distributions.

Upon being built, this layer adds losses (accessible via the `losses`
property) representing the divergences of `kernel` and/or `bias` surrogate
posteriors and their respective priors. When doing minibatch stochastic
optimization, make sure to scale this loss such that it is applied just once
per epoch (e.g. if `kl` is the sum of `losses` for each element of the batch,
you should pass `kl / num_examples_per_epoch` to your optimizer).

You can access the `kernel` and/or `bias` posterior and prior distributions
after the layer is built via the `kernel_posterior`, `kernel_prior`,
`bias_posterior` and `bias_prior` properties.

#### Examples

We illustrate a Bayesian neural network with [variational inference](
https://en.wikipedia.org/wiki/Variational_Bayesian_methods),
assuming a dataset of `features` and `labels`.

```python
import tensorflow_probability as tfp

model = tf.keras.Sequential([
    tfp.layers.DenseReparameterization(512, activation=tf.nn.relu),
    tfp.layers.DenseReparameterization(10),
])

logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits)
kl = sum(model.losses)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)
```

It uses local reparameterization gradients to minimize the
Kullback-Leibler divergence up to a constant, also known as the
negative Evidence Lower Bound. It consists of the sum of two terms:
the expected negative log-likelihood, which we approximate via
Monte Carlo; and the KL divergence, which is added via regularizer
terms which are arguments to the layer.

#### References

[1]: Diederik Kingma, Tim Salimans, and Max Welling. Variational Dropout and
     the Local Reparameterization Trick. In _Neural Information Processing
     Systems_, 2015. https://arxiv.org/abs/1506.02557
[2]: Dmitry Molchanov, Arsenii Ashukha, Dmitry Vetrov. Variational Dropout
     Sparsifies Deep Neural Networks. In _International Conference on Machine
     Learning_, 2017. https://arxiv.org/abs/1701.05369

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    units,
    activation=None,
    activity_regularizer=None,
    trainable=True,
    kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
    kernel_posterior_tensor_fn=(lambda d: d.sample()),
    kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
    kernel_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
    bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
    bias_posterior_tensor_fn=(lambda d: d.sample()),
    bias_prior_fn=None,
    bias_divergence_fn=(lambda q, p, ignore: tfd.kl_divergence(q, p)),
    **kwargs
)
```

Construct layer.


#### Args:


* <b>`units`</b>: Integer or Long, dimensionality of the output space.
* <b>`activation`</b>: Activation function (`callable`). Set it to None to maintain a
  linear activation.
* <b>`activity_regularizer`</b>: Regularizer function for the output.
* <b>`kernel_posterior_fn`</b>: Python `callable` which creates
  `tfd.Distribution` instance representing the surrogate
  posterior of the `kernel` parameter. Default value:
  `default_mean_field_normal_fn()`.
* <b>`kernel_posterior_tensor_fn`</b>: Python `callable` which takes a
  `tfd.Distribution` instance and returns a representative
  value. Default value: `lambda d: d.sample()`.
* <b>`kernel_prior_fn`</b>: Python `callable` which creates `tfd`
  instance. See `default_mean_field_normal_fn` docstring for required
  parameter signature.
  Default value: `tfd.Normal(loc=0., scale=1.)`.
* <b>`kernel_divergence_fn`</b>: Python `callable` which takes the surrogate posterior
  distribution, prior distribution and random variate sample(s) from the
  surrogate posterior and computes or approximates the KL divergence. The
  distributions are `tfd.Distribution`-like instances and the
  sample is a `Tensor`.
* <b>`bias_posterior_fn`</b>: Python `callable` which creates
  `tfd.Distribution` instance representing the surrogate
  posterior of the `bias` parameter. Default value:
  `default_mean_field_normal_fn(is_singular=True)` (which creates an
  instance of `tfd.Deterministic`).
* <b>`bias_posterior_tensor_fn`</b>: Python `callable` which takes a
  `tfd.Distribution` instance and returns a representative
  value. Default value: `lambda d: d.sample()`.
* <b>`bias_prior_fn`</b>: Python `callable` which creates `tfd` instance.
  See `default_mean_field_normal_fn` docstring for required parameter
  signature. Default value: `None` (no prior, no variational inference)
* <b>`bias_divergence_fn`</b>: Python `callable` which takes the surrogate posterior
  distribution, prior distribution and random variate sample(s) from the
  surrogate posterior and computes or approximates the KL divergence. The
  distributions are `tfd.Distribution`-like instances and the
  sample is a `Tensor`.



## Properties

<h3 id="activity_regularizer"><code>activity_regularizer</code></h3>

Optional regularizer function for the output of this layer.


<h3 id="dtype"><code>dtype</code></h3>




<h3 id="dynamic"><code>dynamic</code></h3>




<h3 id="input"><code>input</code></h3>

Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.

#### Returns:

Input tensor or list of input tensors.



#### Raises:


* <b>`AttributeError`</b>: if the layer is connected to
more than one incoming layers.


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

<h3 id="apply"><code>apply</code></h3>

``` python
apply(
    inputs,
    *args,
    **kwargs
)
```

Apply the layer on a input.

This is an alias of `self.__call__`.

#### Arguments:


* <b>`inputs`</b>: Input tensor(s).
* <b>`*args`</b>: additional positional arguments to be passed to `self.call`.
* <b>`**kwargs`</b>: additional keyword arguments to be passed to `self.call`.


#### Returns:

Output tensor(s).


<h3 id="build"><code>build</code></h3>

``` python
build(input_shape)
```

Creates the variables of the layer (optional, for subclass implementers).

This is a method that implementers of subclasses of `Layer` or `Model`
can override if they need a state-creation step in-between
layer instantiation and layer call.

This is typically used to create the weights of `Layer` subclasses.

#### Arguments:


* <b>`input_shape`</b>: Instance of `TensorShape`, or list of instances of
  `TensorShape` if the layer expects a list of inputs
  (one instance per input).

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

Computes the output shape of the layer.


#### Args:


* <b>`input_shape`</b>: Shape tuple (tuple of integers) or list of shape tuples
  (one per output tensor of the layer). Shape tuples can include None for
  free dimensions, instead of an integer.


#### Returns:


* <b>`output_shape`</b>: A tuple representing the output shape.


#### Raises:


* <b>`ValueError`</b>: If innermost dimension of `input_shape` is not defined.

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

This method is the reverse of `get_config`, capable of instantiating the
same layer from the config dictionary.

#### Args:


* <b>`config`</b>: A Python dictionary, typically the output of `get_config`.


#### Returns:


* <b>`layer`</b>: A layer instance.

<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```

Returns the config of the layer.

A layer config is a Python dictionary (serializable) containing the
configuration of a layer. The same layer can be reinstantiated later
(without its trained weights) from this configuration.

#### Returns:


* <b>`config`</b>: A Python dictionary of class keyword arguments and their
  serialized values.

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




