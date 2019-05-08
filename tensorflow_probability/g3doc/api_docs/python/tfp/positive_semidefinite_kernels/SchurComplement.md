<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.positive_semidefinite_kernels.SchurComplement" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="base_kernel"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="cholesky_bijector"/>
<meta itemprop="property" content="divisor_matrix"/>
<meta itemprop="property" content="divisor_matrix_cholesky"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="feature_ndims"/>
<meta itemprop="property" content="fixed_inputs"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__iadd__"/>
<meta itemprop="property" content="__imul__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="apply"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="matrix"/>
</div>

# tfp.positive_semidefinite_kernels.SchurComplement

## Class `SchurComplement`

The SchurComplement kernel.

Inherits From: [`PositiveSemidefiniteKernel`](../../tfp/positive_semidefinite_kernels/PositiveSemidefiniteKernel.md)



Defined in [`python/positive_semidefinite_kernels/schur_complement.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/positive_semidefinite_kernels/schur_complement.py).

<!-- Placeholder for "Used in" -->

Given a block matrix `M = [[A, B], [C, D]]`, the Schur complement of D in M is
written `M / D = A - B @ Inverse(D) @ C`.

This class represents a PositiveSemidefiniteKernel whose behavior is as
follows. We compute a matrix, analogous to `D` in the above definition, by
calling `base_kernel.matrix(fixed_inputs, fixed_inputs)`. Then given new input
locations `x` and `y`, we can construct the remaining pieces of `M` above, and
compute the Schur complement of `D` in `M` (see Mathematical Details, below).

Notably, this kernel uses a bijector (Invert(CholeskyOuterProduct)), as an
intermediary for the requisite matrix solve, which means we get a caching
benefit after the first use.

### Mathematical Details

Suppose we have a kernel `k` and some fixed collection of inputs
`Z = [z0, z1, ..., zN]`. Given new inputs `x` and `y`, we can form a block
matrix

 ```none
   M = [
     [k(x, y), k(x, z0), ..., k(x, zN)],
     [k(z0, y), k(z0, z0), ..., k(z0, zN)],
     ...,
     [k(zN, y), k(z0, zN), ..., k(zN, zN)],
   ]
 ```

We might write this, so as to emphasize the block structure,

 ```none
   M = [
     [xy, xZ],
     [yZ^T, ZZ],
   ],

   xy = [k(x, y)]
   xZ = [k(x, z0), ..., k(x, zN)]
   yZ = [k(y, z0), ..., k(y, zN)]
   ZZ = "the matrix of k(zi, zj)'s"
 ```

Then we have the definition of this kernel's apply method:

`schur_comp.apply(x, y) = xy - xZ @ ZZ^{-1} @ yZ^T`

and similarly, if x and y are collections of inputs.

As with other PSDKernels, the `apply` method acts as a (possibly
vectorized) scalar function of 2 inputs. Given a single `x` and `y`,
`apply` will yield a scalar output. Given two (equal size!) collections `X`
and `Y`, it will yield another (equal size!) collection of scalar outputs.

### Examples

Here's a simple example usage, with no particular motivation.

```python
from tensorflow_probability import positive_semidefinite_kernels as psd_kernel

base_kernel = psd_kernel.ExponentiatedQuadratic(amplitude=np.float64(1.))
# 3 points in 1-dimensional space (shape [3, 1]).
z = [[0.], [3.], [4.]]

schur_kernel = psd_kernel.SchurComplement(
    base_kernel=base_kernel,
    fixed_inputs=z)

# Two individual 1-d points
x = [1.]
y = [2.]
print(schur_kernel.apply(x, y))
# ==> k(x, y) - k(x, z) @ Inverse(k(z, z)) @ k(z, y)
```

A more motivating application of this kernel is in constructing a Gaussian
process that is conditioned on some observed data.

```python
from tensorflow_probability import distributions as tfd
from tensorflow_probability import positive_semidefinite_kernels as psd_kernel

base_kernel = psd_kernel.ExponentiatedQuadratic(amplitude=np.float64(1.))
observation_index_points = np.random.uniform(-1., 1., [50, 1])
observations = np.sin(2 * np.pi * observation_index_points[..., 0])

posterior_kernel = psd_kernel.SchurComplement(
    base_kernel=base_kernel,
    fixed_inputs=observation_index_points)

# Assume we use a zero prior mean, and compute the posterior mean.
def posterior_mean_fn(x):
  k_x_obs_linop = tf.linalg.LinearOperatorFullMatrix(
      base_kernel.matrix(x, observation_index_points))
  chol_linop = tf.linalg.LinearOperatorLowerTriangular(
      posterior_kernel.divisor_matrix_cholesky)

  return k_x_obs_linop.matvec(
      chol_linop.solvevec(
          chol_linop.solvevec(observations),
          adjoint=True))

# Construct the GP posterior distribution at some new points.
gp_posterior = tfp.distributions.GaussianProcess(
    index_points=np.linspace(-1., 1., 100)[..., np.newaxis],
    kernel=posterior_kernel,
    mean_fn=posterior_mean_fn)

# Draw 5 samples on the above 100-point grid
samples = gp_posterior.sample(5)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    base_kernel,
    fixed_inputs,
    diag_shift=None,
    validate_args=False,
    name='SchurComplement'
)
```

Construct a SchurComplement kernel instance.

#### Args:

* <b>`base_kernel`</b>: A `PositiveSemidefiniteKernel` instance, the kernel used to
  build the block matrices of which this kernel computes the  Schur
  complement.
* <b>`fixed_inputs`</b>: A Tensor, representing a collection of inputs. The Schur
  complement that this kernel computes comes from a block matrix, whose
  bottom-right corner is derived from `base_kernel.matrix(fixed_inputs,
  fixed_inputs)`, and whose top-right and bottom-left pieces are
  constructed by computing the base_kernel at pairs of input locations
  together with these `fixed_inputs`. `fixed_inputs` is allowed to be an
  empty collection (either `None` or having a zero shape entry), in which
  case the kernel falls back to the trivial application of `base_kernel`
  to inputs. See class-level docstring for more details on the exact
  computation this does; `fixed_inputs` correspond to the `Z` structure
  discussed there. `fixed_inputs` is assumed to have shape `[b1, ..., bB,
  N, f1, ..., fF]` where the `b`'s are batch shape entries, the `f`'s are
  feature_shape entries, and `N` is the number of fixed inputs. Use of
  this kernel entails a 1-time O(N^3) cost of computing the Cholesky
  decomposition of the k(Z, Z) matrix. The batch shape elements of
  `fixed_inputs` must be broadcast compatible with
  `base_kernel.batch_shape`.
* <b>`diag_shift`</b>: A floating point scalar to be added to the diagonal of the
  divisor_matrix before computing its Cholesky.
* <b>`validate_args`</b>: If `True`, parameters are checked for validity despite
  possibly degrading runtime performance.
  Default value: `False`
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: `"SchurComplement"`



## Properties

<h3 id="base_kernel"><code>base_kernel</code></h3>



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

<h3 id="cholesky_bijector"><code>cholesky_bijector</code></h3>



<h3 id="divisor_matrix"><code>divisor_matrix</code></h3>



<h3 id="divisor_matrix_cholesky"><code>divisor_matrix_cholesky</code></h3>



<h3 id="dtype"><code>dtype</code></h3>

DType over which the kernel operates.

<h3 id="feature_ndims"><code>feature_ndims</code></h3>

The number of feature dimensions.

Kernel functions generally act on pairs of inputs from some space like

```none
R^(d1 x ... x  dD)
```

or, in words: rank-`D` real-valued tensors of shape `[d1, ..., dD]`. Inputs
can be vectors in some `R^N`, but are not restricted to be. Indeed, one
might consider kernels over matrices, tensors, or even more general spaces,
like strings or graphs.

#### Returns:

The number of feature dimensions (feature rank) of this kernel.

<h3 id="fixed_inputs"><code>fixed_inputs</code></h3>



<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this class.



## Methods

<h3 id="__add__"><code>__add__</code></h3>

``` python
__add__(k)
```



<h3 id="__iadd__"><code>__iadd__</code></h3>

``` python
__iadd__(k)
```



<h3 id="__imul__"><code>__imul__</code></h3>

``` python
__imul__(k)
```



<h3 id="__mul__"><code>__mul__</code></h3>

``` python
__mul__(k)
```



<h3 id="apply"><code>apply</code></h3>

``` python
apply(
    x1,
    x2
)
```

Apply the kernel function to a pair of (batches of) inputs.

#### Args:

* <b>`x1`</b>: `Tensor` input to the first positional parameter of the kernel, of
  shape `[b1, ..., bB, f1, ..., fF]`, where `B` may be zero (ie, no
  batching) and `F` (number of feature dimensions) must equal the kernel's
  `feature_ndims` property. Batch shape must broadcast with the batch
  shape of `x2` and with the kernel's parameters.
* <b>`x2`</b>: `Tensor` input to the second positional parameter of the kernel,
  shape `[c1, ..., cC, f1, ..., fF]`, where `C` may be zero (ie, no
  batching) and `F` (number of feature dimensions) must equal the kernel's
  `feature_ndims` property. Batch shape must broadcast with the batch
  shape of `x1` and with the kernel's parameters.


#### Returns:

  `Tensor` containing the (batch of) results of applying the kernel function
  to inputs `x1` and `x2`. If the kernel parameters' batch shape is
  `[k1, ..., kK]` then the shape of the `Tensor` resulting from this method
  call is `broadcast([b1, ..., bB], [c1, ..., cC], [k1, ..., kK])`.

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
single scalar value. Given the same kernel and, say, batched inputs of shape
`[b1, ..., bB, f1, ..., fF]`, it will yield a batch of scalars of shape
`[b1, ..., bB]`.

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
be broadcast together. We can fix this by giving the parameter a shape of
`[2, 1]` which will correctly broadcast with `[5]` to yield `[2, 5]`:

```python
batch_kernel = tfp.positive_semidefinite_kernels.SomeKernel(
    param=[[.2], [.5]])
batch_kernel.batch_shape
# ==> [2, 1]
batch_kernel.apply(x, y).shape
# ==> [2, 5]
```

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
  shape `[b1, ..., bB, e1, f1, ..., fF]`, where `B` may be zero (ie, no
  batching), e1 is an integer greater than zero, and `F` (number of
  feature dimensions) must equal the kernel's `feature_ndims` property.
  Batch shape must broadcast with the batch shape of `x2` and with the
  kernel's parameters *after* parameter expansion (see
  `param_expansion_ndims` argument).
* <b>`x2`</b>: `Tensor` input to the second positional parameter of the kernel,
  shape `[c1, ..., cC, e2, f1, ..., fF]`, where `C` may be zero (ie, no
  batching), e2 is an integer greater than zero,  and `F` (number of
  feature dimensions) must equal the kernel's `feature_ndims` property.
  Batch shape must broadcast with the batch shape of `x1` and with the
  kernel's parameters *after* parameter expansion (see
  `param_expansion_ndims` argument).


#### Returns:

  `Tensor containing (batch of) matrices of kernel applications to pairs
  from inputs `x1` and `x2`. If the kernel parameters' batch shape is
  `[k1, ..., kK]`, then the shape of the resulting `Tensor` is
  `broadcast([b1, ..., bB], [c1, ..., cC], [k1, ..., kK]) + [e1, e2]`.

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

N.B., this method can only be used to compute the pairwise application of
the kernel function on rank-1 collections. E.g., it *does* support inputs of
shape `[e1, f]` and `[e2, f]`, yielding a matrix of shape `[e1, e2]`. It
*does not* support inputs of shape `[e1, e2, f]` and `[e3, e4, f]`, yielding
a `Tensor` of shape `[e1, e2, e3, e4]`. To do this, one should instead
reshape the inputs and pass them to `apply`, e.g.:

```python
k = tfpk.SomeKernel()
t1 = tf.placeholder([4, 4, 3], tf.float32)
t2 = tf.placeholder([5, 5, 3], tf.float32)
k.apply(
    tf.reshape(t1, [4, 4, 1, 1, 3]),
    tf.reshape(t2, [1, 1, 5, 5, 3])).shape
# ==> [4, 4, 5, 5, 3]
```

`matrix` is a special case of the above, where there is only one example
dimension; indeed, its implementation looks almost exactly like the above
(reshaped inputs passed to the private version of `_apply`).

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



