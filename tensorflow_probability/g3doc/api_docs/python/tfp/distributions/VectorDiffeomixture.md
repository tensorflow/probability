<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.VectorDiffeomixture" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="distribution"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="endpoint_affine"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="grid"/>
<meta itemprop="property" content="interpolated_affine"/>
<meta itemprop="property" content="mixture_distribution"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="cdf"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="covariance"/>
<meta itemprop="property" content="cross_entropy"/>
<meta itemprop="property" content="entropy"/>
<meta itemprop="property" content="event_shape_tensor"/>
<meta itemprop="property" content="is_scalar_batch"/>
<meta itemprop="property" content="is_scalar_event"/>
<meta itemprop="property" content="kl_divergence"/>
<meta itemprop="property" content="log_cdf"/>
<meta itemprop="property" content="log_prob"/>
<meta itemprop="property" content="log_survival_function"/>
<meta itemprop="property" content="mean"/>
<meta itemprop="property" content="mode"/>
<meta itemprop="property" content="param_shapes"/>
<meta itemprop="property" content="param_static_shapes"/>
<meta itemprop="property" content="prob"/>
<meta itemprop="property" content="quantile"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="stddev"/>
<meta itemprop="property" content="survival_function"/>
<meta itemprop="property" content="variance"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfp.distributions.VectorDiffeomixture

## Class `VectorDiffeomixture`

VectorDiffeomixture distribution.

Inherits From: [`Distribution`](../../tfp/distributions/Distribution.md)



Defined in [`python/distributions/vector_diffeomixture.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/vector_diffeomixture.py).

<!-- Placeholder for "Used in" -->

A vector diffeomixture (VDM) is a distribution parameterized by a convex
combination of `K` component `loc` vectors, `loc[k], k = 0,...,K-1`, and `K`
`scale` matrices `scale[k], k = 0,..., K-1`.  It approximates the following
[compound distribution]
(https://en.wikipedia.org/wiki/Compound_probability_distribution)

```none
p(x) = int p(x | z) p(z) dz,
where z is in the K-simplex, and
p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
```

The integral `int p(x | z) p(z) dz` is approximated with a quadrature scheme
adapted to the mixture density `p(z)`.  The `N` quadrature points `z_{N, n}`
and weights `w_{N, n}` (which are non-negative and sum to 1) are chosen
such that

```q_N(x) := sum_{n=1}^N w_{n, N} p(x | z_{N, n}) --> p(x)```

as `N --> infinity`.

Since `q_N(x)` is in fact a mixture (of `N` points), we may sample from
`q_N` exactly.  It is important to note that the VDM is *defined* as `q_N`
above, and *not* `p(x)`.  Therefore, sampling and pdf may be implemented as
exact (up to floating point error) methods.

A common choice for the conditional `p(x | z)` is a multivariate Normal.

The implemented marginal `p(z)` is the `SoftmaxNormal`, which is a
`K-1` dimensional Normal transformed by a `SoftmaxCentered` bijector, making
it a density on the `K`-simplex.  That is,

```
Z = SoftmaxCentered(X),
X = Normal(mix_loc / temperature, 1 / temperature)
```

The default quadrature scheme chooses `z_{N, n}` as `N` midpoints of
the quantiles of `p(z)` (generalized quantiles if `K > 2`).

See [Dillon and Langmore (2018)][1] for more details.

#### About `Vector` distributions in TensorFlow.

The `VectorDiffeomixture` is a non-standard distribution that has properties
particularly useful in [variational Bayesian
methods](https://en.wikipedia.org/wiki/Variational_Bayesian_methods).

Conditioned on a draw from the SoftmaxNormal, `X|z` is a vector whose
components are linear combinations of affine transformations, thus is itself
an affine transformation.

Note: The marginals `X_1|v, ..., X_d|v` are *not* generally identical to some
parameterization of `distribution`.  This is due to the fact that the sum of
draws from `distribution` are not generally itself the same `distribution`.

#### About `Diffeomixture`s and reparameterization.

The `VectorDiffeomixture` is designed to be reparameterized, i.e., its
parameters are only used to transform samples from a distribution which has no
trainable parameters. This property is important because backprop stops at
sources of stochasticity. That is, as long as the parameters are used *after*
the underlying source of stochasticity, the computed gradient is accurate.

Reparametrization means that we can use gradient-descent (via backprop) to
optimize Monte-Carlo objectives. Such objectives are a finite-sample
approximation of an expectation and arise throughout scientific computing.

WARNING: If you backprop through a VectorDiffeomixture sample and the "base"
distribution is both: not `FULLY_REPARAMETERIZED` and a function of trainable
variables, then the gradient is not guaranteed correct!

#### Examples

```python
tfd = tfp.distributions

# Create two batches of VectorDiffeomixtures, one with mix_loc=[0.],
# another with mix_loc=[1]. In both cases, `K=2` and the affine
# transformations involve:
# k=0: loc=zeros(dims)  scale=LinearOperatorScaledIdentity
# k=1: loc=[2.]*dims    scale=LinOpDiag
dims = 5
vdm = tfd.VectorDiffeomixture(
    mix_loc=[[0.], [1]],
    temperature=[1.],
    distribution=tfd.Normal(loc=0., scale=1.),
    loc=[
        None,  # Equivalent to `np.zeros(dims, dtype=np.float32)`.
        np.float32([2.]*dims),
    ],
    scale=[
        tf.linalg.LinearOperatorScaledIdentity(
          num_rows=dims,
          multiplier=np.float32(1.1),
          is_positive_definite=True),
        tf.linalg.LinearOperatorDiag(
          diag=np.linspace(2.5, 3.5, dims, dtype=np.float32),
          is_positive_definite=True),
    ],
    validate_args=True)
```

#### References

[1]: Joshua Dillon and Ian Langmore. Quadrature Compound: An approximating
     family of distributions. _arXiv preprint arXiv:1801.03080_, 2018.
     https://arxiv.org/abs/1801.03080

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    mix_loc,
    temperature,
    distribution,
    loc=None,
    scale=None,
    quadrature_size=8,
    quadrature_fn=tfp.distributions.quadrature_scheme_softmaxnormal_quantiles,
    validate_args=False,
    allow_nan_stats=True,
    name='VectorDiffeomixture'
)
```

Constructs the VectorDiffeomixture on `R^d`.

The vector diffeomixture (VDM) approximates the compound distribution

```none
p(x) = int p(x | z) p(z) dz,
where z is in the K-simplex, and
p(x | z) := p(x | loc=sum_k z[k] loc[k], scale=sum_k z[k] scale[k])
```

#### Args:


* <b>`mix_loc`</b>: `float`-like `Tensor` with shape `[b1, ..., bB, K-1]`.
  In terms of samples, larger `mix_loc[..., k]` ==>
  `Z` is more likely to put more weight on its `kth` component.
* <b>`temperature`</b>: `float`-like `Tensor`. Broadcastable with `mix_loc`.
  In terms of samples, smaller `temperature` means one component is more
  likely to dominate.  I.e., smaller `temperature` makes the VDM look more
  like a standard mixture of `K` components.
* <b>`distribution`</b>: <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>-like instance. Distribution
  from which `d` iid samples are used as input to the selected affine
  transformation. Must be a scalar-batch, scalar-event distribution.
  Typically `distribution.reparameterization_type = FULLY_REPARAMETERIZED`
  or it is a function of non-trainable parameters. WARNING: If you
  backprop through a VectorDiffeomixture sample and the `distribution`
  is not `FULLY_REPARAMETERIZED` yet is a function of trainable variables,
  then the gradient will be incorrect!
* <b>`loc`</b>: Length-`K` list of `float`-type `Tensor`s. The `k`-th element
  represents the `shift` used for the `k`-th affine transformation.  If
  the `k`-th item is `None`, `loc` is implicitly `0`.  When specified,
  must have shape `[B1, ..., Bb, d]` where `b >= 0` and `d` is the event
  size.
* <b>`scale`</b>: Length-`K` list of `LinearOperator`s. Each should be
  positive-definite and operate on a `d`-dimensional vector space. The
  `k`-th element represents the `scale` used for the `k`-th affine
  transformation. `LinearOperator`s must have shape `[B1, ..., Bb, d, d]`,
  `b >= 0`, i.e., characterizes `b`-batches of `d x d` matrices
* <b>`quadrature_size`</b>: Python `int` scalar representing number of
  quadrature points.  Larger `quadrature_size` means `q_N(x)` better
  approximates `p(x)`.
* <b>`quadrature_fn`</b>: Python callable taking `normal_loc`, `normal_scale`,
  `quadrature_size`, `validate_args` and returning `tuple(grid, probs)`
  representing the SoftmaxNormal grid and corresponding normalized weight.
  normalized) weight.
  Default value: `quadrature_scheme_softmaxnormal_quantiles`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
  statistics (e.g., mean, mode, variance) use the value "`NaN`" to
  indicate the result is undefined. When `False`, an exception is raised
  if one or more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.


#### Raises:


* <b>`ValueError`</b>: if `not scale or len(scale) < 2`.
* <b>`ValueError`</b>: if `len(loc) != len(scale)`
* <b>`ValueError`</b>: if `quadrature_grid_and_probs is not None` and
  `len(quadrature_grid_and_probs[0]) != len(quadrature_grid_and_probs[1])`
* <b>`ValueError`</b>: if `validate_args` and any not scale.is_positive_definite.
* <b>`TypeError`</b>: if any scale.dtype != scale[0].dtype.
* <b>`TypeError`</b>: if any loc.dtype != scale[0].dtype.
* <b>`NotImplementedError`</b>: if `len(scale) != 2`.
* <b>`ValueError`</b>: if `not distribution.is_scalar_batch`.
* <b>`ValueError`</b>: if `not distribution.is_scalar_event`.



## Properties

<h3 id="allow_nan_stats"><code>allow_nan_stats</code></h3>

Python `bool` describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense. E.g., the variance of a
Cauchy distribution is infinity. However, sometimes the statistic is
undefined, e.g., if a distribution's pdf does not achieve a maximum within
the support of the distribution, the mode is undefined. If the mean is
undefined, then by definition the variance is undefined. E.g. the mean for
Student's T for df = 1 is undefined (no clear way to say it is either + or -
infinity), so the variance = E[(X - mean)**2] is also undefined.

#### Returns:


* <b>`allow_nan_stats`</b>: Python `bool`.

<h3 id="batch_shape"><code>batch_shape</code></h3>

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

#### Returns:


* <b>`batch_shape`</b>: `TensorShape`, possibly unknown.

<h3 id="distribution"><code>distribution</code></h3>

Base scalar-event, scalar-batch distribution.


<h3 id="dtype"><code>dtype</code></h3>

The `DType` of `Tensor`s handled by this `Distribution`.


<h3 id="endpoint_affine"><code>endpoint_affine</code></h3>

Affine transformation for each of `K` components.


<h3 id="event_shape"><code>event_shape</code></h3>

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

#### Returns:


* <b>`event_shape`</b>: `TensorShape`, possibly unknown.

<h3 id="grid"><code>grid</code></h3>

Grid of mixing probabilities, one for each grid point.


<h3 id="interpolated_affine"><code>interpolated_affine</code></h3>

Affine transformation for each convex combination of `K` components.


<h3 id="mixture_distribution"><code>mixture_distribution</code></h3>

Distribution used to select a convex combination of affine transforms.


<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this `Distribution`.


<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="parameters"><code>parameters</code></h3>

Dictionary of parameters used to instantiate this `Distribution`.


<h3 id="reparameterization_type"><code>reparameterization_type</code></h3>

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

#### Returns:

An instance of `ReparameterizationType`.


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

Python `bool` indicating possibly expensive checks are enabled.


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

<h3 id="__getitem__"><code>__getitem__</code></h3>

``` python
__getitem__(slices)
```

Slices the batch axes of this distribution, returning a new instance.

```python
b = tfd.Bernoulli(logits=tf.zeros([3, 5, 7, 9]))
b.batch_shape  # => [3, 5, 7, 9]
b2 = b[:, tf.newaxis, ..., -2:, 1::2]
b2.batch_shape  # => [3, 1, 5, 2, 4]

x = tf.random.normal([5, 3, 2, 2])
cov = tf.matmul(x, x, transpose_b=True)
chol = tf.cholesky(cov)
loc = tf.random.normal([4, 1, 3, 1])
mvn = tfd.MultivariateNormalTriL(loc, chol)
mvn.batch_shape  # => [4, 5, 3]
mvn.event_shape  # => [2]
mvn2 = mvn[:, 3:, ..., ::-1, tf.newaxis]
mvn2.batch_shape  # => [4, 2, 3, 1]
mvn2.event_shape  # => [2]
```

#### Args:


* <b>`slices`</b>: slices from the [] operator


#### Returns:


* <b>`dist`</b>: A new `tfd.Distribution` instance with sliced parameters.

<h3 id="__iter__"><code>__iter__</code></h3>

``` python
__iter__()
```




<h3 id="batch_shape_tensor"><code>batch_shape_tensor</code></h3>

``` python
batch_shape_tensor(name='batch_shape_tensor')
```

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

#### Args:


* <b>`name`</b>: name to give to the op


#### Returns:


* <b>`batch_shape`</b>: `Tensor`.

<h3 id="cdf"><code>cdf</code></h3>

``` python
cdf(
    value,
    name='cdf',
    **kwargs
)
```

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```none
cdf(x) := P[X <= x]
```

#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
  values of type `self.dtype`.

<h3 id="copy"><code>copy</code></h3>

``` python
copy(**override_parameters_kwargs)
```

Creates a deep copy of the distribution.

Note: the copy distribution may continue to depend on the original
initialization arguments.

#### Args:


* <b>`**override_parameters_kwargs`</b>: String/value dictionary of initialization
  arguments to override with new values.


#### Returns:


* <b>`distribution`</b>: A new instance of `type(self)` initialized from the union
  of self.parameters and override_parameters_kwargs, i.e.,
  `dict(self.parameters, **override_parameters_kwargs)`.

<h3 id="covariance"><code>covariance</code></h3>

``` python
covariance(
    name='covariance',
    **kwargs
)
```

Covariance.

Covariance is (possibly) defined only for non-scalar-event distributions.

For example, for a length-`k`, vector-valued distribution, it is calculated
as,

```none
Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
```

where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
denotes expectation.

Alternatively, for non-vector, multivariate distributions (e.g.,
matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
under some vectorization of the events, i.e.,

```none
Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
```

where `Cov` is a (batch of) `k' x k'` matrices,
`0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
mapping indices of this distribution's event dimensions to indices of a
length-`k'` vector.

#### Args:


* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`covariance`</b>: Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
  where the first `n` dimensions are batch coordinates and
  `k' = reduce_prod(self.event_shape)`.

<h3 id="cross_entropy"><code>cross_entropy</code></h3>

``` python
cross_entropy(
    other,
    name='cross_entropy'
)
```

Computes the (Shannon) cross entropy.

Denote this distribution (`self`) by `P` and the `other` distribution by
`Q`. Assuming `P, Q` are absolutely continuous with respect to
one another and permit densities `p(x) dr(x)` and `q(x) dr(x)`, (Shannon)
cross entropy is defined as:

```none
H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
```

where `F` denotes the support of the random variable `X ~ P`.

#### Args:


* <b>`other`</b>: <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a> instance.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.


#### Returns:


* <b>`cross_entropy`</b>: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
  representing `n` different calculations of (Shannon) cross entropy.

<h3 id="entropy"><code>entropy</code></h3>

``` python
entropy(
    name='entropy',
    **kwargs
)
```

Shannon entropy in nats.


<h3 id="event_shape_tensor"><code>event_shape_tensor</code></h3>

``` python
event_shape_tensor(name='event_shape_tensor')
```

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.


#### Args:


* <b>`name`</b>: name to give to the op


#### Returns:


* <b>`event_shape`</b>: `Tensor`.

<h3 id="is_scalar_batch"><code>is_scalar_batch</code></h3>

``` python
is_scalar_batch(name='is_scalar_batch')
```

Indicates that `batch_shape == []`.


#### Args:


* <b>`name`</b>: Python `str` prepended to names of ops created by this function.


#### Returns:


* <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.

<h3 id="is_scalar_event"><code>is_scalar_event</code></h3>

``` python
is_scalar_event(name='is_scalar_event')
```

Indicates that `event_shape == []`.


#### Args:


* <b>`name`</b>: Python `str` prepended to names of ops created by this function.


#### Returns:


* <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.

<h3 id="kl_divergence"><code>kl_divergence</code></h3>

``` python
kl_divergence(
    other,
    name='kl_divergence'
)
```

Computes the Kullback--Leibler divergence.

Denote this distribution (`self`) by `p` and the `other` distribution by
`q`. Assuming `p, q` are absolutely continuous with respect to reference
measure `r`, the KL divergence is defined as:

```none
KL[p, q] = E_p[log(p(X)/q(X))]
         = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x)
         = H[p, q] - H[p]
```

where `F` denotes the support of the random variable `X ~ p`, `H[., .]`
denotes (Shannon) cross entropy, and `H[.]` denotes (Shannon) entropy.

#### Args:


* <b>`other`</b>: <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a> instance.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.


#### Returns:


* <b>`kl_divergence`</b>: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
  representing `n` different calculations of the Kullback-Leibler
  divergence.

<h3 id="log_cdf"><code>log_cdf</code></h3>

``` python
log_cdf(
    value,
    name='log_cdf',
    **kwargs
)
```

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```none
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.

#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
  values of type `self.dtype`.

<h3 id="log_prob"><code>log_prob</code></h3>

``` python
log_prob(
    value,
    name='log_prob',
    **kwargs
)
```

Log probability density/mass function.


#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
  values of type `self.dtype`.

<h3 id="log_survival_function"><code>log_survival_function</code></h3>

``` python
log_survival_function(
    value,
    name='log_survival_function',
    **kwargs
)
```

Log survival function.

Given random variable `X`, the survival function is defined:

```none
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:

`Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
  `self.dtype`.


<h3 id="mean"><code>mean</code></h3>

``` python
mean(
    name='mean',
    **kwargs
)
```

Mean.


<h3 id="mode"><code>mode</code></h3>

``` python
mode(
    name='mode',
    **kwargs
)
```

Mode.


<h3 id="param_shapes"><code>param_shapes</code></h3>

``` python
param_shapes(
    cls,
    sample_shape,
    name='DistributionParamShapes'
)
```

Shapes of parameters given the desired shape of a call to `sample()`.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`.

Subclasses should override class method `_param_shapes`.

#### Args:


* <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
  `sample()`.
* <b>`name`</b>: name to prepend ops with.


#### Returns:

`dict` of parameter name to `Tensor` shapes.


<h3 id="param_static_shapes"><code>param_static_shapes</code></h3>

``` python
param_static_shapes(
    cls,
    sample_shape
)
```

param_shapes with static (i.e. `TensorShape`) shapes.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`. Assumes that the sample's
shape is known statically.

Subclasses should override class method `_param_shapes` to return
constant-valued tensors when constant values are fed.

#### Args:


* <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
  to `sample()`.


#### Returns:

`dict` of parameter name to `TensorShape`.



#### Raises:


* <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.

<h3 id="prob"><code>prob</code></h3>

``` python
prob(
    value,
    name='prob',
    **kwargs
)
```

Probability density/mass function.


#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
  values of type `self.dtype`.

<h3 id="quantile"><code>quantile</code></h3>

``` python
quantile(
    value,
    name='quantile',
    **kwargs
)
```

Quantile function. Aka "inverse cdf" or "percent point function".

Given random variable `X` and `p in [0, 1]`, the `quantile` is:

```none
quantile(p) := x such that P[X <= x] == p
```

#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`quantile`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
  values of type `self.dtype`.

<h3 id="sample"><code>sample</code></h3>

``` python
sample(
    sample_shape=(),
    seed=None,
    name='sample',
    **kwargs
)
```

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

#### Args:


* <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
* <b>`seed`</b>: Python integer seed for RNG
* <b>`name`</b>: name to give to the op.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.

<h3 id="stddev"><code>stddev</code></h3>

``` python
stddev(
    name='stddev',
    **kwargs
)
```

Standard deviation.

Standard deviation is defined as,

```none
stddev = E[(X - E[X])**2]**0.5
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `stddev.shape = batch_shape + event_shape`.

#### Args:


* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`stddev`</b>: Floating-point `Tensor` with shape identical to
  `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.

<h3 id="survival_function"><code>survival_function</code></h3>

``` python
survival_function(
    value,
    name='survival_function',
    **kwargs
)
```

Survival function.

Given random variable `X`, the survival function is defined:

```none
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```

#### Args:


* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:

`Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
  `self.dtype`.


<h3 id="variance"><code>variance</code></h3>

``` python
variance(
    name='variance',
    **kwargs
)
```

Variance.

Variance is defined as,

```none
Var = E[(X - E[X])**2]
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `Var.shape = batch_shape + event_shape`.

#### Args:


* <b>`name`</b>: Python `str` prepended to names of ops created by this function.
* <b>`**kwargs`</b>: Named arguments forwarded to subclass implementation.


#### Returns:


* <b>`variance`</b>: Floating-point `Tensor` with shape identical to
  `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.

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




