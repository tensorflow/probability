<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.ConditionalBijector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="forward_min_event_ndims"/>
<meta itemprop="property" content="graph_parents"/>
<meta itemprop="property" content="inverse_min_event_ndims"/>
<meta itemprop="property" content="is_constant_jacobian"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="forward"/>
<meta itemprop="property" content="forward_event_shape"/>
<meta itemprop="property" content="forward_event_shape_tensor"/>
<meta itemprop="property" content="forward_log_det_jacobian"/>
<meta itemprop="property" content="inverse"/>
<meta itemprop="property" content="inverse_event_shape"/>
<meta itemprop="property" content="inverse_event_shape_tensor"/>
<meta itemprop="property" content="inverse_log_det_jacobian"/>
</div>

# tfp.bijectors.ConditionalBijector

## Class `ConditionalBijector`

Inherits From: [`Bijector`](../../tfp/bijectors/Bijector.md)

Conditional Bijector is a Bijector that allows intrinsic conditioning.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    graph_parents=None,
    is_constant_jacobian=False,
    validate_args=False,
    dtype=None,
    forward_min_event_ndims=None,
    inverse_min_event_ndims=None,
    name=None
)
```

Constructs Bijector.

A `Bijector` transforms random variables into new random variables.

Examples:

```python
# Create the Y = g(X) = X transform.
identity = Identity()

# Create the Y = g(X) = exp(X) transform.
exp = Exp()
```

See `Bijector` subclass docstring for more details and specific examples.

#### Args:

* <b>`graph_parents`</b>: Python list of graph prerequisites of this `Bijector`.
* <b>`is_constant_jacobian`</b>: Python `bool` indicating that the Jacobian matrix is
    not a function of the input.
* <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
    with asserts. If `validate_args` is `False`, and the inputs are invalid,
    correct behavior is not guaranteed.
* <b>`dtype`</b>: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
    enforced.
* <b>`forward_min_event_ndims`</b>: Python `integer` indicating the minimum number of
    dimensions `forward` operates on.
* <b>`inverse_min_event_ndims`</b>: Python `integer` indicating the minimum number of
    dimensions `inverse` operates on. Will be set to
    `forward_min_event_ndims` by default, if no value is provided.
* <b>`name`</b>: The name to give Ops created by the initializer.


#### Raises:

* <b>`ValueError`</b>:  If neither `forward_min_event_ndims` and
    `inverse_min_event_ndims` are specified, or if either of them is
    negative.
* <b>`ValueError`</b>:  If a member of `graph_parents` is not a `Tensor`.



## Properties

<h3 id="dtype"><code>dtype</code></h3>

dtype of `Tensor`s transformable by this distribution.

<h3 id="forward_min_event_ndims"><code>forward_min_event_ndims</code></h3>

Returns the minimal number of dimensions bijector.forward operates on.

<h3 id="graph_parents"><code>graph_parents</code></h3>

Returns this `Bijector`'s graph_parents as a Python list.

<h3 id="inverse_min_event_ndims"><code>inverse_min_event_ndims</code></h3>

Returns the minimal number of dimensions bijector.inverse operates on.

<h3 id="is_constant_jacobian"><code>is_constant_jacobian</code></h3>

Returns true iff the Jacobian matrix is not a function of x.

Note: Jacobian matrix is either constant for both forward and inverse or
neither.

#### Returns:

* <b>`is_constant_jacobian`</b>: Python `bool`.

<h3 id="name"><code>name</code></h3>

Returns the string name of this `Bijector`.

<h3 id="validate_args"><code>validate_args</code></h3>

Returns True if Tensor arguments will be validated.



## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    value,
    name=None,
    **kwargs
)
```

Applies or composes the `Bijector`, depending on input type.

This is a convenience function which applies the `Bijector` instance in
three different ways, depending on the input:

1. If the input is a `tfd.Distribution` instance, return
   `tfd.TransformedDistribution(distribution=input, bijector=self)`.
2. If the input is a `tfb.Bijector` instance, return
   `tfb.Chain([self, input])`.
3. Otherwise, return `self.forward(input)`

#### Args:

* <b>`value`</b>: A `tfd.Distribution`, `tfb.Bijector`, or a `Tensor`.
* <b>`name`</b>: Python `str` name given to ops created by this function.
* <b>`**kwargs`</b>: Additional keyword arguments passed into the created
    `tfd.TransformedDistribution`, `tfb.Bijector`, or `self.forward`.


#### Returns:

* <b>`composition`</b>: A `tfd.TransformedDistribution` if the input was a
    `tfd.Distribution`, a `tfb.Chain` if the input was a `tfb.Bijector`, or
    a `Tensor` computed by `self.forward`.

#### Examples

```python
sigmoid = tfb.Reciprocal()(
    tfb.AffineScalar(shift=1.)(
      tfb.Exp()(
        tfb.AffineScalar(scale=-1.))))
# ==> `tfb.Chain([
#         tfb.Reciprocal(),
#         tfb.AffineScalar(shift=1.),
#         tfb.Exp(),
#         tfb.AffineScalar(scale=-1.),
#      ])`  # ie, `tfb.Sigmoid()`

log_normal = tfb.Exp()(tfd.Normal(0, 1))
# ==> `tfd.TransformedDistribution(tfd.Normal(0, 1), tfb.Exp())`

tfb.Exp()([-1., 0., 1.])
# ==> tf.exp([-1., 0., 1.])
```

<h3 id="forward"><code>forward</code></h3>

``` python
forward(
    *args,
    **kwargs
)
```

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.

<h3 id="forward_event_shape"><code>forward_event_shape</code></h3>

``` python
forward_event_shape(input_shape)
```

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

#### Args:

* <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.


#### Returns:

* <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.

<h3 id="forward_event_shape_tensor"><code>forward_event_shape_tensor</code></h3>

``` python
forward_event_shape_tensor(
    input_shape,
    name='forward_event_shape_tensor'
)
```

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

#### Args:

* <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
* <b>`name`</b>: name to give to the op


#### Returns:

* <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.

<h3 id="forward_log_det_jacobian"><code>forward_log_det_jacobian</code></h3>

``` python
forward_log_det_jacobian(
    *args,
    **kwargs
)
```

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.

<h3 id="inverse"><code>inverse</code></h3>

``` python
inverse(
    *args,
    **kwargs
)
```

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.

<h3 id="inverse_event_shape"><code>inverse_event_shape</code></h3>

``` python
inverse_event_shape(output_shape)
```

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

#### Args:

* <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.


#### Returns:

* <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.

<h3 id="inverse_event_shape_tensor"><code>inverse_event_shape_tensor</code></h3>

``` python
inverse_event_shape_tensor(
    output_shape,
    name='inverse_event_shape_tensor'
)
```

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

#### Args:

* <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
* <b>`name`</b>: name to give to the op


#### Returns:

* <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.

<h3 id="inverse_log_det_jacobian"><code>inverse_log_det_jacobian</code></h3>

``` python
inverse_log_det_jacobian(
    *args,
    **kwargs
)
```

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation.



