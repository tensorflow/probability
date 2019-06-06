<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.bijectors.MatvecLU" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="forward_min_event_ndims"/>
<meta itemprop="property" content="graph_parents"/>
<meta itemprop="property" content="inverse_min_event_ndims"/>
<meta itemprop="property" content="is_constant_jacobian"/>
<meta itemprop="property" content="lower_upper"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="permutation"/>
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

# tfp.bijectors.MatvecLU

## Class `MatvecLU`

Matrix-vector multiply using LU decomposition.

Inherits From: [`Bijector`](../../tfp/bijectors/Bijector.md)



Defined in [`python/bijectors/matveclu.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/bijectors/matveclu.py).

<!-- Placeholder for "Used in" -->

This bijector is identical to the "Convolution1x1" used in Glow
[(Kingma and Dhariwal, 2018)[1].

Warning: this bijector never verifies the scale matrix (as parameterized by LU
decomposition) is invertible. Ensuring this is the case is the caller's
responsibility.

#### Examples

Here's an example of initialization via random weights matrix:

```python
def trainable_lu_factorization(
    event_size, batch_shape=(), seed=None, dtype=tf.float32, name=None):
  with tf.name_scope(name, 'trainable_lu_factorization',
                     [event_size, batch_shape]):
    event_size = tf.convert_to_tensor(
        event_size, preferred_dtype=tf.int32, name='event_size')
    batch_shape = tf.convert_to_tensor(
        batch_shape, preferred_dtype=event_size.dtype, name='batch_shape')
    random_matrix = tf.random_uniform(
        shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
        dtype=dtype,
        seed=seed)
    random_orthonormal = tf.linalg.qr(random_matrix)[0]
    lower_upper, permutation = tf.linalg.lu(random_orthonormal)
    lower_upper = tf.Variable(
        initial_value=lower_upper,
        trainable=True,
        use_resource=True,
        name='lower_upper')
    return lower_upper, permutation

channels = 3
conv1x1 = tfb.MatvecLU(*trainable_lu_factorization(channels),
                       validate_args=True)
x = tf.random_uniform(shape=[2, 28, 28, channels])
fwd = conv1x1.forward(x)
rev_fwd = conv1x1.inverse(fwd)
# ==> x
```

To initialize this variable outside of TensorFlow, one can also use SciPy,
e.g.,

```python
def lu_factorized_random_orthonormal_matrix(channels, dtype=np.float32):
  random_matrix = np.random.rand(channels, channels).astype(dtype)
  lower_upper = scipy.linalg.qr(random_matrix)[0]
  permutation = scipy.linalg.lu(lower_upper, overwrite_a=True)[0]
  permutation = np.argmax(permutation, axis=-2)
  return lower_upper, permutation
```

#### References

[1]: Diederik P. Kingma, Prafulla Dhariwal. Glow: Generative Flow with
     Invertible 1x1 Convolutions. _arXiv preprint arXiv:1807.03039_, 2018.
     https://arxiv.org/abs/1807.03039

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    lower_upper,
    permutation,
    validate_args=False,
    name=None
)
```

Creates the MatvecLU bijector.


#### Args:


* <b>`lower_upper`</b>: The LU factorization as returned by `tf.linalg.lu`.
* <b>`permutation`</b>: The LU factorization permutation as returned by
  `tf.linalg.lu`.
* <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
  checked for correctness.
  Default value: `False`.
* <b>`name`</b>: Python `str` name given to ops managed by this object.
  Default value: `None` (i.e., "MatvecLU").


#### Raises:


* <b>`ValueError`</b>: If both/neither `channels` and `lower_upper`/`permutation` are
  specified.



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

<h3 id="lower_upper"><code>lower_upper</code></h3>




<h3 id="name"><code>name</code></h3>

Returns the string name of this `Bijector`.


<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="permutation"><code>permutation</code></h3>




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


* <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
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


* <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian determinant evaluation.
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


* <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
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


* <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian determinant evaluation.
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




