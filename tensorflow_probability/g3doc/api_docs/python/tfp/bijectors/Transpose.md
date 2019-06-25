<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.Transpose" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="forward_min_event_ndims"/>
<meta itemprop="property" content="graph_parents"/>
<meta itemprop="property" content="inverse_min_event_ndims"/>
<meta itemprop="property" content="is_constant_jacobian"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="perm"/>
<meta itemprop="property" content="rightmost_transposed_ndims"/>
<meta itemprop="property" content="submodules"/>
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
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfp.bijectors.Transpose

## Class `Transpose`

Compute `Y = g(X) = transpose_rightmost_dims(X, rightmost_perm)`.

Inherits From: [`Bijector`](../../tfp/bijectors/Bijector.md)



Defined in [`python/bijectors/transpose.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors/transpose.py).

<!-- Placeholder for "Used in" -->

This bijector is semantically similar to `tf.transpose` except that it
transposes only the rightmost "event" dimensions. That is, unlike
`tf.transpose` the `perm` argument is itself a permutation of
`tf.range(rightmost_transposed_ndims)` rather than `tf.range(tf.rank(x))`,
i.e., users specify the (rightmost) dimensions to permute, not all dimensions.

The actual (forward) transformation is:

```python
def forward(x, perm):
  sample_batch_ndims = tf.rank(x) - tf.size(perm)
  perm = tf.concat([
      tf.range(sample_batch_ndims),
      sample_batch_ndims + perm,
  ], axis=0)
  return tf.transpose(x, perm)
```

#### Examples

```python
tfp.bijectors.Transpose(perm=[1, 0]).forward(
    [
      [[1, 2],
       [3, 4]],
      [[5, 6],
       [7, 8]],
    ])
# ==>
#  [
#    [[1, 3],
#     [2, 4]],
#    [[5, 7],
#     [6, 8]],
#  ]

# Using `rightmost_transposed_ndims=2` means this bijector has the same
# semantics as `tf.matrix_transpose`.
tfp.bijectors.Transpose(rightmost_transposed_ndims=2).inverse(
    [
      [[1, 3],
       [2, 4]],
      [[5, 7],
       [6, 8]],
    ])
# ==>
#  [
#    [[1, 2],
#     [3, 4]],
#    [[5, 6],
#     [7, 8]],
#  ]
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    perm=None,
    rightmost_transposed_ndims=None,
    validate_args=False,
    name='transpose'
)
```

Instantiates the `Transpose` bijector.


#### Args:


* <b>`perm`</b>: Positive `int32` vector-shaped `Tensor` representing permutation of
  rightmost dims (for forward transformation).  Note that the `0`th index
  represents the first of the rightmost dims and the largest value must be
  `rightmost_transposed_ndims - 1` and corresponds to `tf.rank(x) - 1`.
  Only one of `perm` and `rightmost_transposed_ndims` can (and must) be
  specified.
  Default value:
  `tf.range(start=rightmost_transposed_ndims, limit=-1, delta=-1)`.
* <b>`rightmost_transposed_ndims`</b>: Positive `int32` scalar-shaped `Tensor`
  representing the number of rightmost dimensions to permute.
  Only one of `perm` and `rightmost_transposed_ndims` can (and must) be
  specified.
  Default value: `tf.size(perm)`.
* <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
  checked for correctness.
* <b>`name`</b>: Python `str` name given to ops managed by this object.


#### Raises:


* <b>`ValueError`</b>: if both or neither `perm` and `rightmost_transposed_ndims` are
  specified.
* <b>`NotImplementedError`</b>: if `rightmost_transposed_ndims` is not known prior to
  graph execution.



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


<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="perm"><code>perm</code></h3>




<h3 id="rightmost_transposed_ndims"><code>rightmost_transposed_ndims</code></h3>




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


<h3 id="trainable_variables"><code>trainable_variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).


<h3 id="validate_args"><code>validate_args</code></h3>

Returns True if Tensor arguments will be validated.


<h3 id="variables"><code>variables</code></h3>

Sequence of variables owned by this module and it's submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.

#### Returns:

A sequence of variables for the current module (sorted by attribute
name) followed by variables from all submodules recursively (breadth
first).




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




