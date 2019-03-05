<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.Reshape" />
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

# tfp.bijectors.Reshape

## Class `Reshape`

Inherits From: [`Bijector`](../../tfp/bijectors/Bijector.md)

Reshapes the `event_shape` of a `Tensor`.

The semantics generally follow that of `tf.reshape()`, with
a few differences:

* The user must provide both the input and output shape, so that
  the transformation can be inverted. If an input shape is not
  specified, the default assumes a vector-shaped input, i.e.,
  event_shape_in = (-1,).
* The `Reshape` bijector automatically broadcasts over the leftmost
  dimensions of its input (`sample_shape` and `batch_shape`); only
  the rightmost `event_ndims_in` dimensions are reshaped. The
  number of dimensions to reshape is inferred from the provided
  `event_shape_in` (`event_ndims_in = len(event_shape_in)`).

Example usage:

```python
r = tfp.bijectors.Reshape(event_shape_out=[1, -1])

r.forward([3., 4.])    # shape [2]
# ==> [[3., 4.]]       # shape [1, 2]

r.forward([[1., 2.], [3., 4.]])  # shape [2, 2]
# ==> [[[1., 2.]],
#      [[3., 4.]]]   # shape [2, 1, 2]

r.inverse([[3., 4.]])  # shape [1,2]
# ==> [3., 4.]         # shape [2]

r.forward_log_det_jacobian(any_value)
# ==> 0.

r.inverse_log_det_jacobian(any_value)
# ==> 0.
```

Note: we had to make a tricky-to-describe policy decision, which we attempt to
summarize here. At instantiation time and class method invocation time, we
validate consistency of class-level and method-level arguments. Note that
since the class-level arguments may be unspecified until graph execution time,
we had the option of deciding between two validation policies. One was,
roughly, "the earliest, most statically specified arguments take precedence".
The other was "method-level arguments must be consistent with class-level
arguments". The former policy is in a sense more optimistic about user intent,
and would enable us, in at least one particular case [1], to perform
additional inference about resulting shapes. We chose the latter policy, as it
is simpler to implement and a bit easier to articulate.

[1] The case in question is exemplified in the following snippet:

```python
bijector = tfp.bijectors.Reshape(
  event_shape_out=tf.placeholder(dtype=tf.int32, shape=[1]),
  event_shape_in= tf.placeholder(dtype=tf.int32, shape=[3]),
  validate_args=True)

bijector.forward_event_shape(tf.TensorShape([5, 2, 3, 7]))
# Chosen policy    ==> (5, None)
# Alternate policy ==> (5, 42)
```

In the chosen policy, since we don't know what `event_shape_in/out` are at the
time of the call to `forward_event_shape`, we simply fill in everything we
*do* know, which is that the last three dims will be replaced with
"something".

In the alternate policy, we would assume that the intention must be to reshape
`[5, 2, 3, 7]` such that the last three dims collapse to one, which is only
possible if the resulting shape is `[5, 42]`.

Note that the above is the *only* case in which we could do such inference; if
the output shape has more than 1 dim, we can't infer anything. E.g., we would
have

```python
bijector = tfp.bijectors.Reshape(
  event_shape_out=tf.placeholder(dtype=tf.int32, shape=[2]),
  event_shape_in= tf.placeholder(dtype=tf.int32, shape=[3]),
  validate_args=True)

bijector.forward_event_shape(tf.TensorShape([5, 2, 3, 7]))
# Either policy ==> (5, None, None)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    event_shape_out,
    event_shape_in=(-1,),
    validate_args=False,
    name=None
)
```

Creates a `Reshape` bijector.

#### Args:

* <b>`event_shape_out`</b>: An `int`-like vector-shaped `Tensor`
    representing the event shape of the transformed output.
* <b>`event_shape_in`</b>: An optional `int`-like vector-shape `Tensor`
    representing the event shape of the input. This is required in
    order to define inverse operations; the default of (-1,)
    assumes a vector-shaped input.
* <b>`validate_args`</b>: Python `bool` indicating whether arguments should
    be checked for correctness.
* <b>`name`</b>: Python `str`, name given to ops managed by this object.


#### Raises:

* <b>`TypeError`</b>: if either `event_shape_in` or `event_shape_out` has
    non-integer `dtype`.
* <b>`ValueError`</b>: if either of `event_shape_in` or `event_shape_out`
   has non-vector shape (`rank > 1`), or if their sizes do not
   match.



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
    x,
    name='forward'
)
```

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

#### Args:

* <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
* <b>`name`</b>: The name to give this op.


#### Returns:

`Tensor`.


#### Raises:

* <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
* <b>`NotImplementedError`</b>: if `_forward` is not implemented.

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
    x,
    event_ndims,
    name='forward_log_det_jacobian'
)
```

Returns both the forward_log_det_jacobian.

#### Args:

* <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian determinant evaluation.
* <b>`event_ndims`</b>: Number of dimensions in the probabilistic events being
    transformed. Must be greater than or equal to
    `self.forward_min_event_ndims`. The result is summed over the final
    dimensions to produce a scalar Jacobian determinant for each event, i.e.
    it has shape `x.shape.ndims - event_ndims` dimensions.
* <b>`name`</b>: The name to give this op.


#### Returns:

`Tensor`, if this bijector is injective.
  If not injective this is not implemented.


#### Raises:

* <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
* <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented, or
    this is a non-injective bijector.

<h3 id="inverse"><code>inverse</code></h3>

``` python
inverse(
    y,
    name='inverse'
)
```

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

#### Args:

* <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
* <b>`name`</b>: The name to give this op.


#### Returns:

`Tensor`, if this bijector is injective.
  If not injective, returns the k-tuple containing the unique
  `k` points `(x1, ..., xk)` such that `g(xi) = y`.


#### Raises:

* <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
* <b>`NotImplementedError`</b>: if `_inverse` is not implemented.

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
    y,
    event_ndims,
    name='inverse_log_det_jacobian'
)
```

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function,
evaluated at `g^{-1}(y)`.

#### Args:

* <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian determinant evaluation.
* <b>`event_ndims`</b>: Number of dimensions in the probabilistic events being
    transformed. Must be greater than or equal to
    `self.inverse_min_event_ndims`. The result is summed over the final
    dimensions to produce a scalar Jacobian determinant for each event, i.e.
    it has shape `y.shape.ndims - event_ndims` dimensions.
* <b>`name`</b>: The name to give this op.


#### Returns:

* <b>`ildj`</b>: `Tensor`, if this bijector is injective.
    If not injective, returns the tuple of local log det
    Jacobians, `log(det(Dg_i^{-1}(y)))`, where `g_i` is the restriction
    of `g` to the `ith` partition `Di`.


#### Raises:

* <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
* <b>`NotImplementedError`</b>: if `_inverse_log_det_jacobian` is not implemented.



