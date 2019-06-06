<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.Beta" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="concentration0"/>
<meta itemprop="property" content="concentration1"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="total_concentration"/>
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

# tfp.distributions.Beta

## Class `Beta`

Beta distribution.

Inherits From: [`Distribution`](../../tfp/distributions/Distribution.md)



Defined in [`python/distributions/beta.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/beta.py).

<!-- Placeholder for "Used in" -->

The Beta distribution is defined over the `(0, 1)` interval using parameters
`concentration1` (aka "alpha") and `concentration0` (aka "beta").

#### Mathematical Details

The probability density function (pdf) is,

```none
pdf(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
```

where:

* `concentration1 = alpha`,
* `concentration0 = beta`,
* `Z` is the normalization constant, and,
* `Gamma` is the [gamma function](
  https://en.wikipedia.org/wiki/Gamma_function).

The concentration parameters represent mean total counts of a `1` or a `0`,
i.e.,

```none
concentration1 = alpha = mean * total_concentration
concentration0 = beta  = (1. - mean) * total_concentration
```

where `mean` in `(0, 1)` and `total_concentration` is a positive real number
representing a mean `total_count = concentration1 + concentration0`.

Distribution parameters are automatically broadcast in all functions; see
examples for details.

Warning: The samples can be zero due to finite precision.
This happens more often when some of the concentrations are very small.
Make sure to round the samples to `np.finfo(dtype).tiny` before computing the
density.

Samples of this distribution are reparameterized (pathwise differentiable).
The derivatives are computed using the approach described in the paper

[Michael Figurnov, Shakir Mohamed, Andriy Mnih.
Implicit Reparameterization Gradients, 2018](https://arxiv.org/abs/1805.08498)

#### Examples

```python
import tensorflow_probability as tfp
tfd = tfp.distributions

# Create a batch of three Beta distributions.
alpha = [1, 2, 3]
beta = [1, 2, 3]
dist = tfd.Beta(alpha, beta)

dist.sample([4, 5])  # Shape [4, 5, 3]

# `x` has three batch entries, each with two samples.
x = [[.1, .4, .5],
     [.2, .3, .5]]
# Calculate the probability of each pair of samples under the corresponding
# distribution in `dist`.
dist.prob(x)         # Shape [2, 3]
```

```python
# Create batch_shape=[2, 3] via parameter broadcast:
alpha = [[1.], [2]]      # Shape [2, 1]
beta = [3., 4, 5]        # Shape [3]
dist = tfd.Beta(alpha, beta)

# alpha broadcast as: [[1., 1, 1,],
#                      [2, 2, 2]]
# beta broadcast as:  [[3., 4, 5],
#                      [3, 4, 5]]
# batch_Shape [2, 3]
dist.sample([4, 5])  # Shape [4, 5, 2, 3]

x = [.2, .3, .5]
# x will be broadcast as [[.2, .3, .5],
#                         [.2, .3, .5]],
# thus matching batch_shape [2, 3].
dist.prob(x)         # Shape [2, 3]
```

Compute the gradients of samples w.r.t. the parameters:

```python
alpha = tf.constant(1.0)
beta = tf.constant(2.0)
dist = tfd.Beta(alpha, beta)
samples = dist.sample(5)  # Shape [5]
loss = tf.reduce_mean(tf.square(samples))  # Arbitrary loss function
# Unbiased stochastic gradients of the loss function
grads = tf.gradients(loss, [alpha, beta])
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    concentration1,
    concentration0,
    validate_args=False,
    allow_nan_stats=True,
    name='Beta'
)
```

Initialize a batch of Beta distributions.


#### Args:


* <b>`concentration1`</b>: Positive floating-point `Tensor` indicating mean
  number of successes; aka "alpha". Implies `self.dtype` and
  `self.batch_shape`, i.e.,
  `concentration1.shape = [N1, N2, ..., Nm] = self.batch_shape`.
* <b>`concentration0`</b>: Positive floating-point `Tensor` indicating mean
  number of failures; aka "beta". Otherwise has same semantics as
  `concentration1`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
  parameters are checked for validity despite possibly degrading runtime
  performance. When `False` invalid inputs may silently render incorrect
  outputs.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
  (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
  result is undefined. When `False`, an exception is raised if one or
  more of the statistic's batch members are undefined.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.



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

<h3 id="concentration0"><code>concentration0</code></h3>

Concentration parameter associated with a `0` outcome.


<h3 id="concentration1"><code>concentration1</code></h3>

Concentration parameter associated with a `1` outcome.


<h3 id="dtype"><code>dtype</code></h3>

The `DType` of `Tensor`s handled by this `Distribution`.


<h3 id="event_shape"><code>event_shape</code></h3>

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

#### Returns:


* <b>`event_shape`</b>: `TensorShape`, possibly unknown.

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


<h3 id="total_concentration"><code>total_concentration</code></h3>

Sum of concentration parameters.


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


Additional documentation from `Beta`:

Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`.

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

`other` types with built-in registrations: `Beta`

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

`other` types with built-in registrations: `Beta`

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


Additional documentation from `Beta`:

Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`.

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


Additional documentation from `Beta`:

Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`.

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

Additional documentation from `Beta`:

Note: The mode is undefined when `concentration1 <= 1` or
`concentration0 <= 1`. If `self.allow_nan_stats` is `True`, `NaN`
is used for undefined modes. If `self.allow_nan_stats` is `False` an
exception is raised when one or more modes are undefined.

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


Additional documentation from `Beta`:

Note: `x` must have dtype `self.dtype` and be in
`[0, 1].` It must have a shape compatible with `self.batch_shape()`.

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




