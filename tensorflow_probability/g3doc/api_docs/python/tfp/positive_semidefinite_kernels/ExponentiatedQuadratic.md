<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.positive_semidefinite_kernels.ExponentiatedQuadratic" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="amplitude"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="feature_ndims"/>
<meta itemprop="property" content="length_scale"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="matrix"/>
<meta itemprop="property" content="tensor"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfp.positive_semidefinite_kernels.ExponentiatedQuadratic

## Class `ExponentiatedQuadratic`

The ExponentiatedQuadratic kernel.

Inherits From: [`PositiveSemidefiniteKernel`](../../tfp/positive_semidefinite_kernels/PositiveSemidefiniteKernel.md)



Defined in [`python/positive_semidefinite_kernels/exponentiated_quadratic.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/positive_semidefinite_kernels/exponentiated_quadratic.py).

<!-- Placeholder for "Used in" -->

Sometimes called the "squared exponential", "Gaussian" or "radial basis
function", this kernel function has the form

  ```none
  k(x, y) = amplitude**2 * exp(-||x - y||**2 / (2 * length_scale**2))
  ```

where the double-bars represent vector length (ie, Euclidean, or L2 norm).

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    amplitude=None,
    length_scale=None,
    feature_ndims=1,
    validate_args=False,
    name='ExponentiatedQuadratic'
)
```

Construct an ExponentiatedQuadratic kernel instance.


#### Args:


* <b>`amplitude`</b>: floating point `Tensor` that controls the maximum value
  of the kernel. Must be broadcastable with `length_scale` and inputs to
  `apply` and `matrix` methods. Must be greater than zero.
* <b>`length_scale`</b>: floating point `Tensor` that controls how sharp or wide the
  kernel shape is. This provides a characteristic "unit" of length against
  which `||x - y||` can be compared for scale. Must be broadcastable with
  `amplitude` and inputs to `apply` and `matrix` methods.
* <b>`feature_ndims`</b>: Python `int` number of rightmost dims to include in the
  squared difference norm in the exponential.
* <b>`validate_args`</b>: If `True`, parameters are checked for validity despite
  possibly degrading runtime performance
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.



## Properties

<h3 id="amplitude"><code>amplitude</code></h3>

Amplitude parameter.


<h3 id="batch_shape"><code>batch_shape</code></h3>

The batch_shape property of a PositiveSemidefiniteKernel.

This property describes the fully broadcast shape of all kernel parameters.
For example, consider an ExponentiatedQuadratic kernel, which is
parameterized by an amplitude and length_scale:

```none
exp_quad(x, x') := amplitude * exp(||x - x'||**2 / length_scale**2)
```

The batch_shape of such a kernel is derived from broadcasting the shapes of
`amplitude` and `length_scale`. E.g., if their shapes were

```python
amplitude.shape = [2, 1, 1]
length_scale.shape = [1, 4, 3]
```

then `exp_quad`'s batch_shape would be `[2, 4, 3]`.

Note that this property defers to the private _batch_shape method, which
concrete implementation sub-classes are obliged to provide.

#### Returns:

`TensorShape` instance describing the fully broadcast shape of all
kernel parameters.


<h3 id="dtype"><code>dtype</code></h3>

DType over which the kernel operates.


<h3 id="feature_ndims"><code>feature_ndims</code></h3>

The number of feature dimensions.

Kernel functions generally act on pairs of inputs from some space like

```none
R^(d1 x ... x dD)
```

or, in words: rank-`D` real-valued tensors of shape `[d1, ..., dD]`. Inputs
can be vectors in some `R^N`, but are not restricted to be. Indeed, one
might consider kernels over matrices, tensors, or even more general spaces,
like strings or graphs.

#### Returns:

The number of feature dimensions (feature rank) of this kernel.


<h3 id="length_scale"><code>length_scale</code></h3>

Length scale parameter.


<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this class.


<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


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

<h3 id="__add__"><code>__add__</code></h3>

``` python
__add__(k)
```




<h3 id="__mul__"><code>__mul__</code></h3>

``` python
__mul__(k)
```




<h3 id="apply"><code>apply</code></h3>

``` python
apply(
    x1,
    x2,
    example_ndims=0
)
```

Apply the kernel function pairs of inputs.


#### Args:


* <b>`x1`</b>: `Tensor` input to the kernel, of shape `B1 + E1 + F`, where `B1` and
  `E1` may be empty (ie, no batch/example dims, resp.) and `F` (the
  feature shape) must have rank equal to the kernel's `feature_ndims`
  property. Batch shape must broadcast with the batch shape of `x2` and
  with the kernel's batch shape. Example shape must broadcast with example
  shape of `x2`. `x1` and `x2` must have the same *number* of example dims
  (ie, same rank).
* <b>`x2`</b>: `Tensor` input to the kernel, of shape `B2 + E2 + F`, where `B2` and
  `E2` may be empty (ie, no batch/example dims, resp.) and `F` (the
  feature shape) must have rank equal to the kernel's `feature_ndims`
  property. Batch shape must broadcast with the batch shape of `x2` and
  with the kernel's batch shape. Example shape must broadcast with example
  shape of `x2`. `x1` and `x2` must have the same *number* of example
* <b>`example_ndims`</b>: A python integer, the number of example dims in the inputs.
  In essence, this parameter controls how broadcasting of the kernel's
  batch shape with input batch shapes works. The kernel batch shape will
  be broadcast against everything to the left of the combined example and
  feature dimensions in the input shapes.


#### Returns:

`Tensor` containing the results of applying the kernel function to inputs
`x1` and `x2`. If the kernel parameters' batch shape is `Bk` then the
shape of the `Tensor` resulting from this method call is
`broadcast(Bk, B1, B2) + broadcast(E1, E2)`.


Given an index set `S`, a kernel function is mathematically defined as a
real- or complex-valued function on `S` satisfying the
positive semi-definiteness constraint:

```none
sum_i sum_j (c[i]*) c[j] k(x[i], x[j]) >= 0
```

for any finite collections `{x[1], ..., x[N]}` in `S` and
`{c[1], ..., c[N]}` in the reals (or the complex plane). '*' is the complex
conjugate, in the complex case.

This method most closely resembles the function described in the
mathematical definition of a kernel. Given a PositiveSemidefiniteKernel `k`
with scalar parameters and inputs `x` and `y` in `S`, `apply(x, y)` yields a
single scalar value.

#### Examples

```python
import tensorflow_probability as tfp

# Suppose `SomeKernel` acts on vectors (rank-1 tensors)
scalar_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=.5)
scalar_kernel.batch_shape
# ==> []

# `x` and `y` are batches of five 3-D vectors:
x = np.ones([5, 3], np.float32)
y = np.ones([5, 3], np.float32)
scalar_kernel.apply(x, y).shape
# ==> [5]
```

The above output is the result of vectorized computation of the five values

```none
[k(x[0], y[0]), k(x[1], y[1]), ..., k(x[4], y[4])]
```

Now we can consider a kernel with batched parameters:

```python
batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=[.2, .5])
batch_kernel.batch_shape
# ==> [2]
batch_kernel.apply(x, y).shape
# ==> Error! [2] and [5] can't broadcast.
```

The parameter batch shape of `[2]` and the input batch shape of `[5]` can't
be broadcast together. We can fix this in either of two ways:

1. Give the parameter a shape of `[2, 1]` which will correctly
broadcast with `[5]` to yield `[2, 5]`:

```python
batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(
    param=[[.2], [.5]])
batch_kernel.batch_shape
# ==> [2, 1]
batch_kernel.apply(x, y).shape
# ==> [2, 5]
```

2. By specifying `example_ndims`, which tells the kernel to treat the `5`
in the input shape as part of the "example shape", and "pushing" the
kernel batch shape to the left:

```python
batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=[.2, .5])
batch_kernel.batch_shape
# ==> [2]
batch_kernel.apply(x, y, example_ndims=1).shape
# ==> [2, 5]

<h3 id="batch_shape_tensor"><code>batch_shape_tensor</code></h3>

``` python
batch_shape_tensor()
```

The batch_shape property of a PositiveSemidefiniteKernel as a `Tensor`.


#### Returns:

`Tensor` which evaluates to a vector of integers which are the
fully-broadcast shapes of the kernel parameters.


<h3 id="matrix"><code>matrix</code></h3>

``` python
matrix(
    x1,
    x2
)
```

Construct (batched) matrices from (batches of) collections of inputs.


#### Args:


* <b>`x1`</b>: `Tensor` input to the first positional parameter of the kernel, of
  shape `B1 + [e1] + F`, where `B1` may be empty (ie, no batch dims,
  resp.), `e1` is a single integer (ie, `x1` has example ndims exactly 1),
  and `F` (the feature shape) must have rank equal to the kernel's
  `feature_ndims` property. Batch shape must broadcast with the batch
  shape of `x2` and with the kernel's batch shape.
* <b>`x2`</b>: `Tensor` input to the second positional parameter of the kernel,
  shape `B2 + [e2] + F`, where `B2` may be empty (ie, no batch dims,
  resp.), `e2` is a single integer (ie, `x2` has example ndims exactly 1),
  and `F` (the feature shape) must have rank equal to the kernel's
  `feature_ndims` property. Batch shape must broadcast with the batch
  shape of `x1` and with the kernel's batch shape.


#### Returns:

`Tensor` containing the matrix (possibly batched) of kernel applications
to pairs from inputs `x1` and `x2`. If the kernel parameters' batch shape
is `Bk` then the shape of the `Tensor` resulting from this method call is
`broadcast(Bk, B1, B2) + [e1, e2]` (note this differs from `apply`: the
example dimensions are concatenated, whereas in `apply` the example dims
are broadcast together).


Given inputs `x1` and `x2` of shapes

```none
[b1, ..., bB, e1, f1, ..., fF]
```

and

```none
[c1, ..., cC, e2, f1, ..., fF]
```

This method computes the batch of `e1 x e2` matrices resulting from applying
the kernel function to all pairs of inputs from `x1` and `x2`. The shape
of the batch of matrices is the result of broadcasting the batch shapes of
`x1`, `x2`, and the kernel parameters (see examples below). As such, it's
required that these shapes all be broadcast compatible. However, the kernel
parameter batch shapes need not broadcast against the 'example shapes' (`e1`
and `e2` above).

When the two inputs are the (batches of) identical collections, the
resulting matrix is the so-called Gram (or Gramian) matrix
(https://en.wikipedia.org/wiki/Gramian_matrix).

#### Examples

First, consider a kernel with a single scalar parameter.

```python
import tensorflow_probability as tfp

scalar_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=.5)
scalar_kernel.batch_shape
# ==> []

# Our inputs are two lists of 3-D vectors
x = np.ones([5, 3], np.float32)
y = np.ones([4, 3], np.float32)
scalar_kernel.matrix(x, y).shape
# ==> [5, 4]
```

The result comes from applying the kernel to the entries in `x` and `y`
pairwise, across all pairs:

  ```none
  | k(x[0], y[0])    k(x[0], y[1])  ...  k(x[0], y[3]) |
  | k(x[1], y[0])    k(x[1], y[1])  ...  k(x[1], y[3]) |
  |      ...              ...                 ...      |
  | k(x[4], y[0])    k(x[4], y[1])  ...  k(x[4], y[3]) |
  ```

Now consider a kernel with batched parameters with the same inputs

```python
batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=[1., .5])
batch_kernel.batch_shape
# ==> [2]

batch_kernel.matrix(x, y).shape
# ==> [2, 5, 4]
```

This results in a batch of 2 matrices, one computed from the kernel with
`param = 1.` and the other with `param = .5`.

We also support batching of the inputs. First, let's look at that with
the scalar kernel again.

```python
# Batch of 10 lists of 5 vectors of dimension 3
x = np.ones([10, 5, 3], np.float32)

# Batch of 10 lists of 4 vectors of dimension 3
y = np.ones([10, 4, 3], np.float32)

scalar_kernel.matrix(x, y).shape
# ==> [10, 5, 4]
```

The result is a batch of 10 matrices built from the batch of 10 lists of
input vectors. These batch shapes have to be broadcastable. The following
will *not* work:

```python
x = np.ones([10, 5, 3], np.float32)
y = np.ones([20, 4, 3], np.float32)
scalar_kernel.matrix(x, y).shape
# ==> Error! [10] and [20] can't broadcast.
```

Now let's consider batches of inputs in conjunction with batches of kernel
parameters. We require that the input batch shapes be broadcastable with
the kernel parameter batch shapes, otherwise we get an error:

```python
x = np.ones([10, 5, 3], np.float32)
y = np.ones([10, 4, 3], np.float32)

batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(params=[1., .5])
batch_kernel.batch_shape
# ==> [2]
batch_kernel.matrix(x, y).shape
# ==> Error! [2] and [10] can't broadcast.
```

The fix is to make the kernel parameter shape broadcastable with `[10]` (or
reshape the inputs to be broadcastable!):

```python
x = np.ones([10, 5, 3], np.float32)
y = np.ones([10, 4, 3], np.float32)

batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(
    params=[[1.], [.5]])
batch_kernel.batch_shape
# ==> [2, 1]
batch_kernel.matrix(x, y).shape
# ==> [2, 10, 5, 4]

# Or, make the inputs broadcastable:
x = np.ones([10, 1, 5, 3], np.float32)
y = np.ones([10, 1, 4, 3], np.float32)

batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(
    params=[1., .5])
batch_kernel.batch_shape
# ==> [2]
batch_kernel.matrix(x, y).shape
# ==> [10, 2, 5, 4]
```

Here, we have the result of applying the kernel, with 2 different
parameters, to each of a batch of 10 pairs of input lists.

<h3 id="tensor"><code>tensor</code></h3>

``` python
tensor(
    x1,
    x2,
    x1_example_ndims,
    x2_example_ndims
)
```

Construct (batched) tensors from (batches of) collections of inputs.


#### Args:


* <b>`x1`</b>: `Tensor` input to the first positional parameter of the kernel, of
  shape `B1 + E1 + F`, where `B1` and `E1` arbitrary shapes which may be
  empty (ie, no batch/example dims, resp.), and `F` (the feature shape)
  must have rank equal to the kernel's `feature_ndims` property. Batch
  shape must broadcast with the batch shape of `x2` and with the kernel's
  batch shape.
* <b>`x2`</b>: `Tensor` input to the second positional parameter of the kernel,
  shape `B2 + E2 + F`, where `B2` and `E2` arbitrary shapes which may be
  empty (ie, no batch/example dims, resp.), and `F` (the feature shape)
  must have rank equal to the kernel's `feature_ndims` property. Batch
  shape must broadcast with the batch shape of `x1` and with the kernel's
  batch shape.
* <b>`x1_example_ndims`</b>: A python integer greater than or equal to 0, the number
  of example dims in the first input. This affects both the alignment of
  batch shapes and the shape of the final output of the function.
  Everything left of the feature shape and the example dims in `x1` is
  considered "batch shape", and must broadcast as specified above.
* <b>`x2_example_ndims`</b>: A python integer greater than or equal to 0, the number
  of example dims in the second input. This affects both the alignment of
  batch shapes and the shape of the final output of the function.
  Everything left of the feature shape and the example dims in `x1` is
  considered "batch shape", and must broadcast as specified above.


#### Returns:

`Tensor` containing (possibly batched) kernel applications to pairs from
inputs `x1` and `x2`. If the kernel parameters' batch shape is `Bk` then
the shape of the `Tensor` resulting from this method call is
`broadcast(Bk, B1, B2) + E1 + E2`. Note this differs from `apply`: the
example dimensions are concatenated, whereas in `apply` the example dims
are broadcast together. It also differs from `matrix`: the example shapes
are arbitrary here, and the result accrues a rank equal to the sum of the
ranks of the input example shapes.


#### Examples

First, consider a kernel with a single scalar parameter.

```python
import tensorflow_probability as tfp

scalar_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=.5)
scalar_kernel.batch_shape
# ==> []

# Our inputs are two rank-2 collections of 3-D vectors
x = np.ones([5, 6, 3], np.float32)
y = np.ones([7, 8, 3], np.float32)
scalar_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
# ==> [5, 6, 7, 8]

# Empty example shapes work too!
x = np.ones([3], np.float32)
y = np.ones([5, 3], np.float32)
scalar_kernel.tensor(x, y, x1_example_ndims=0, x2_example_ndims=1).shape
# ==> [5]
```

The result comes from applying the kernel to the entries in `x` and `y`
pairwise, across all pairs:

  ```none
  | k(x[0], y[0])    k(x[0], y[1])  ...  k(x[0], y[3]) |
  | k(x[1], y[0])    k(x[1], y[1])  ...  k(x[1], y[3]) |
  |      ...              ...                 ...      |
  | k(x[4], y[0])    k(x[4], y[1])  ...  k(x[4], y[3]) |
  ```

Now consider a kernel with batched parameters.

```python
batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(param=[1., .5])
batch_kernel.batch_shape
# ==> [2]

# Inputs are two rank-2 collections of 3-D vectors
x = np.ones([5, 6, 3], np.float32)
y = np.ones([7, 8, 3], np.float32)
scalar_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
# ==> [2, 5, 6, 7, 8]
```

We also support batching of the inputs. First, let's look at that with
the scalar kernel again.

```python
# Batch of 10 lists of 5x6 collections of dimension 3
x = np.ones([10, 5, 6, 3], np.float32)

# Batch of 10 lists of 7x8 collections of dimension 3
y = np.ones([10, 7, 8, 3], np.float32)

scalar_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
# ==> [10, 5, 6, 7, 8]
```

The result is a batch of 10 tensors built from the batch of 10 rank-2
collections of input vectors. The batch shapes have to be broadcastable.
The following will *not* work:

```python
x = np.ones([10, 5, 3], np.float32)
y = np.ones([20, 4, 3], np.float32)
scalar_kernel.tensor(x, y, x1_example_ndims=1, x2_example_ndims=1).shape
# ==> Error! [10] and [20] can't broadcast.
```

Now let's consider batches of inputs in conjunction with batches of kernel
parameters. We require that the input batch shapes be broadcastable with
the kernel parameter batch shapes, otherwise we get an error:

```python
x = np.ones([10, 5, 6, 3], np.float32)
y = np.ones([10, 7, 8, 3], np.float32)

batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(params=[1., .5])
batch_kernel.batch_shape
# ==> [2]
batch_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
# ==> Error! [2] and [10] can't broadcast.
```

The fix is to make the kernel parameter shape broadcastable with `[10]` (or
reshape the inputs to be broadcastable!):

```python
x = np.ones([10, 5, 6, 3], np.float32)
y = np.ones([10, 7, 8, 3], np.float32)

batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(
    params=[[1.], [.5]])
batch_kernel.batch_shape
# ==> [2, 1]
batch_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
# ==> [2, 10, 5, 6, 7, 8]

# Or, make the inputs broadcastable:
x = np.ones([10, 1, 5, 6, 3], np.float32)
y = np.ones([10, 1, 7, 8, 3], np.float32)

batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(
    params=[1., .5])
batch_kernel.batch_shape
# ==> [2]
batch_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
# ==> [10, 2, 5, 6, 7, 8]
```

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




