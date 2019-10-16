<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.bijectors.RationalQuadraticSpline" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="bin_heights"/>
<meta itemprop="property" content="bin_widths"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="forward_min_event_ndims"/>
<meta itemprop="property" content="graph_parents"/>
<meta itemprop="property" content="inverse_min_event_ndims"/>
<meta itemprop="property" content="is_constant_jacobian"/>
<meta itemprop="property" content="knot_slopes"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="range_min"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="variables"/>
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

# tfp.experimental.substrates.jax.bijectors.RationalQuadraticSpline


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/rational_quadratic_spline.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `RationalQuadraticSpline`

A piecewise rational quadratic spline, as developed in [1].

Inherits From: [`Bijector`](../../../../../tfp/experimental/substrates/jax/bijectors/Bijector.md)

<!-- Placeholder for "Used in" -->

This transformation represents a monotonically increasing piecewise rational
quadratic function. Outside of the bounds of `knot_x`/`knot_y`, the transform
behaves as an identity function.

Typically this bijector will be used as part of a chain, with splines for
trailing `x` dimensions conditioned on some of the earlier `x` dimensions, and
with the inverse then solved first for unconditioned dimensions, then using
conditioning derived from those inverses, and so forth. For example, if we
split a 15-D `xs` vector into 3 components, we may implement a forward and
inverse as follows:

```python
nsplits = 3

class SplineParams(tf.Module):

  def __init__(self, nbins=32):
    self._nbins = nbins
    self._built = False
    self._bin_widths = None
    self._bin_heights = None
    self._knot_slopes = None

  def _bin_positions(self, x):
    x = tf.reshape(x, [-1, self._nbins])
    return tf.math.softmax(x, axis=-1) * (2 - self._nbins * 1e-2) + 1e-2

  def _slopes(self, x):
    x = tf.reshape(x, [-1, self._nbins - 1])
    return tf.math.softplus(x) + 1e-2

  def __call__(self, x, nunits):
    if not self._built:
      self._bin_widths = tf.keras.layers.Dense(
          nunits * self._nbins, activation=self._bin_positions, name='w')
      self._bin_heights = tf.keras.layers.Dense(
          nunits * self._nbins, activation=self._bin_positions, name='h')
      self._knot_slopes = tf.keras.layers.Dense(
          nunits * (self._nbins - 1), activation=self._slopes, name='s')
      self._built = True
    return tfb.RationalQuadraticSpline(
        bin_widths=self._bin_widths(x),
        bin_heights=self._bin_heights(x),
        knot_slopes=self._knot_slopes(x))

xs = np.random.randn(1, 15).astype(np.float32)  # Keras won't Dense(.)(vec).
splines = [SplineParams() for _ in range(nsplits)]

def spline_flow():
  stack = tfb.Identity()
  for i in range(nsplits):
    stack = tfb.RealNVP(5 * i, bijector_fn=splines[i])(stack)
  return stack

ys = spline_flow().forward(xs)
ys_inv = spline_flow().inverse(ys)  # ys_inv ~= xs
```

For a one-at-a-time autoregressive flow as in [1], it would be profitable to
implement a mask over `xs` to parallelize either the inverse or the forward
pass and implement the other using a `tf.while_loop`. See
<a href="../../../../../tfp/bijectors/MaskedAutoregressiveFlow.md"><code>tfp.bijectors.MaskedAutoregressiveFlow</code></a> for support doing so (paired with
<a href="../../../../../tfp/bijectors/Invert.md"><code>tfp.bijectors.Invert</code></a> depending which direction should be parallel).

#### References

[1]: Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios. Neural
     Spline Flows. _arXiv preprint arXiv:1906.04032_, 2019.
     https://arxiv.org/abs/1906.04032

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/rational_quadratic_spline.py">View source</a>

``` python
__init__(
    bin_widths,
    bin_heights,
    knot_slopes,
    range_min=-1,
    validate_args=False,
    name=None
)
```

Construct a new RationalQuadraticSpline bijector.

For each argument, the innermost axis indexes bins/knots and batch axes
index axes of `x`/`y` spaces. A `RationalQuadraticSpline` with a separate
transform for each of three dimensions might have `bin_widths` shaped
`[3, 32]`. To use the same spline for each of `x`'s three dimensions we may
broadcast against `x` and use a `bin_widths` parameter shaped `[32]`.
Parameters will be broadcast against each other and against the input
`x`/`y`s, so if we want fixed slopes, we can use kwarg `knot_slopes=1`.

A typical recipe for acquiring compatible bin widths and heights would be:

```python
nbins = unconstrained_vector.shape[-1]
range_min, range_max, min_bin_size = -1, 1, 1e-2
scale = range_max - range_min - nbins * min_bin_size
bin_widths = tf.math.softmax(unconstrained_vector) * scale + min_bin_size
```

#### Args:


* <b>`bin_widths`</b>: The widths of the spans between subsequent knot `x` positions,
  a floating point `Tensor`. Must be positive, and at least 1-D. Innermost
  axis must sum to the same value as `bin_heights`. The knot `x` positions
  will be a first at `range_min`, followed by knots at `range_min +
  cumsum(bin_widths, axis=-1)`.
* <b>`bin_heights`</b>: The heights of the spans between subsequent knot `y`
  positions, a floating point `Tensor`. Must be positive, and at least
  1-D. Innermost axis must sum to the same value as `bin_widths`. The knot
  `y` positions will be a first at `range_min`, followed by knots at
  `range_min + cumsum(bin_heights, axis=-1)`.
* <b>`knot_slopes`</b>: The slope of the spline at each knot, a floating point
  `Tensor`. Must be positive. `1`s are implicitly padded for the first and
  last implicit knots corresponding to `range_min` and `range_min +
  sum(bin_widths, axis=-1)`. Innermost axis size should be 1 less than
  that of `bin_widths`/`bin_heights`, or 1 for broadcasting.
* <b>`range_min`</b>: The `x`/`y` position of the first knot, which has implicit
  slope `1`. `range_max` is implicit, and can be computed as `range_min +
  sum(bin_widths, axis=-1)`. Scalar floating point `Tensor`.
* <b>`validate_args`</b>: Toggles argument validation (can hurt performance).
* <b>`name`</b>: Optional name scope for associated ops. (Defaults to
  `'RationalQuadraticSpline'`).



## Properties

<h3 id="bin_heights"><code>bin_heights</code></h3>




<h3 id="bin_widths"><code>bin_widths</code></h3>




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

<h3 id="knot_slopes"><code>knot_slopes</code></h3>




<h3 id="name"><code>name</code></h3>

Returns the string name of this `Bijector`.


<h3 id="range_min"><code>range_min</code></h3>




<h3 id="trainable_variables"><code>trainable_variables</code></h3>




<h3 id="validate_args"><code>validate_args</code></h3>

Returns True if Tensor arguments will be validated.


<h3 id="variables"><code>variables</code></h3>






## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

``` python
forward(
    x,
    name='forward',
    **kwargs
)
```

Returns the forward `Bijector` evaluation, i.e., X = g(Y).


#### Args:


* <b>`x`</b>: `Tensor`. The input to the 'forward' evaluation.
* <b>`name`</b>: The name to give this op.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:

`Tensor`.



#### Raises:


* <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
  `self.dtype`.
* <b>`NotImplementedError`</b>: if `_forward` is not implemented.

<h3 id="forward_event_shape"><code>forward_event_shape</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

``` python
forward_log_det_jacobian(
    x,
    event_ndims,
    name='forward_log_det_jacobian',
    **kwargs
)
```

Returns both the forward_log_det_jacobian.


#### Args:


* <b>`x`</b>: `Tensor`. The input to the 'forward' Jacobian determinant evaluation.
* <b>`event_ndims`</b>: Number of dimensions in the probabilistic events being
  transformed. Must be greater than or equal to
  `self.forward_min_event_ndims`. The result is summed over the final
  dimensions to produce a scalar Jacobian determinant for each event, i.e.
  it has shape `rank(x) - event_ndims` dimensions.
* <b>`name`</b>: The name to give this op.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

``` python
inverse(
    y,
    name='inverse',
    **kwargs
)
```

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).


#### Args:


* <b>`y`</b>: `Tensor`. The input to the 'inverse' evaluation.
* <b>`name`</b>: The name to give this op.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:

`Tensor`, if this bijector is injective.
  If not injective, returns the k-tuple containing the unique
  `k` points `(x1, ..., xk)` such that `g(xi) = y`.



#### Raises:


* <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
  `self.dtype`.
* <b>`NotImplementedError`</b>: if `_inverse` is not implemented.

<h3 id="inverse_event_shape"><code>inverse_event_shape</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/bijectors/bijector.py">View source</a>

``` python
inverse_log_det_jacobian(
    y,
    event_ndims,
    name='inverse_log_det_jacobian',
    **kwargs
)
```

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function,
evaluated at `g^{-1}(y)`.

#### Args:


* <b>`y`</b>: `Tensor`. The input to the 'inverse' Jacobian determinant evaluation.
* <b>`event_ndims`</b>: Number of dimensions in the probabilistic events being
  transformed. Must be greater than or equal to
  `self.inverse_min_event_ndims`. The result is summed over the final
  dimensions to produce a scalar Jacobian determinant for each event, i.e.
  it has shape `rank(y) - event_ndims` dimensions.
* <b>`name`</b>: The name to give this op.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`ildj`</b>: `Tensor`, if this bijector is injective.
  If not injective, returns the tuple of local log det
  Jacobians, `log(det(Dg_i^{-1}(y)))`, where `g_i` is the restriction
  of `g` to the `ith` partition `Di`.


#### Raises:


* <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
  `self.dtype`.
* <b>`NotImplementedError`</b>: if `_inverse_log_det_jacobian` is not implemented.



