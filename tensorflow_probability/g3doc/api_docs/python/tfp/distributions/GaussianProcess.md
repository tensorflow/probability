<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.GaussianProcess" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="bijector"/>
<meta itemprop="property" content="distribution"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="index_points"/>
<meta itemprop="property" content="jitter"/>
<meta itemprop="property" content="kernel"/>
<meta itemprop="property" content="loc"/>
<meta itemprop="property" content="mean_fn"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="observation_noise_variance"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="scale"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="__init__"/>
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
</div>

# tfp.distributions.GaussianProcess

## Class `GaussianProcess`

Inherits From: [`MultivariateNormalLinearOperator`](../../tfp/distributions/MultivariateNormalLinearOperator.md)

Marginal distribution of a Gaussian process at finitely many points.

A Gaussian process (GP) is an indexed collection of random variables, any
finite collection of which are jointly Gaussian. While this definition applies
to finite index sets, it is typically implicit that the index set is infinite;
in applications, it is often some finite dimensional real or complex vector
space. In such cases, the GP may be thought of as a distribution over
(real- or complex-valued) functions defined over the index set.

Just as Gaussian distriubtions are fully specified by their first and second
moments, a Gaussian process can be completely specified by a mean and
covariance function. Let `S` denote the index set and `K` the space in which
each indexed random variable takes its values (again, often R or C). The mean
function is then a map `m: S -> K`, and the covariance function, or kernel, is
a positive-definite function `k: (S x S) -> K`. The properties of functions
drawn from a GP are entirely dictated (up to translation) by the form of the
kernel function.

This `Distribution` represents the marginal joint distribution over function
values at a given finite collection of points `[x[1], ..., x[N]]` from the
index set `S`. By definition, this marginal distribution is just a
multivariate normal distribution, whose mean is given by the vector
`[ m(x[1]), ..., m(x[N]) ]` and whose covariance matrix is constructed from
pairwise applications of the kernel function to the given inputs:

```none
    | k(x[1], x[1])    k(x[1], x[2])  ...  k(x[1], x[N]) |
    | k(x[2], x[1])    k(x[2], x[2])  ...  k(x[2], x[N]) |
    |      ...              ...                 ...      |
    | k(x[N], x[1])    k(x[N], x[2])  ...  k(x[N], x[N]) |
```

For this to be a valid covariance matrix, it must be symmetric and positive
definite; hence the requirement that `k` be a positive definite function
(which, by definition, says that the above procedure will yield PD matrices).

We also support the inclusion of zero-mean Gaussian noise in the model, via
the `observation_noise_variance` parameter. This augments the generative model
to

```none
f ~ GP(m, k)
(y[i] | f, x[i]) ~ Normal(f(x[i]), s)
```

where

  * `m` is the mean function
  * `k` is the covariance kernel function
  * `f` is the function drawn from the GP
  * `x[i]` are the index points at which the function is observed
  * `y[i]` are the observed values at the index points
  * `s` is the scale of the observation noise.

Note that this class represents an *unconditional* Gaussian process; it does
not implement posterior inference conditional on observed function
evaluations. This class is useful, for example, if one wishes to combine a GP
prior with a non-conjugate likelihood using MCMC to sample from the posterior.

#### Mathematical Details

The probability density function (pdf) is a multivariate normal whose
parameters are derived from the GP's properties:

```none
pdf(x; index_points, mean_fn, kernel) = exp(-0.5 * y) / Z
K = (kernel.matrix(index_points, index_points) +
     (observation_noise_variance + jitter) * eye(N))
y = (x - mean_fn(index_points))^T @ K @ (x - mean_fn(index_points))
Z = (2 * pi)**(.5 * N) |det(K)|**(.5)
```

where:

* `index_points` are points in the index set over which the GP is defined,
* `mean_fn` is a callable mapping the index set to the GP's mean values,
* `kernel` is `PositiveSemidefiniteKernel`-like and represents the covariance
  function of the GP,
* `observation_noise_variance` represents (optional) observation noise.
* `jitter` is added to the diagonal to ensure positive definiteness up to
   machine precision (otherwise Cholesky-decomposition is prone to failure),
* `eye(N)` is an N-by-N identity matrix.

#### Examples

##### Draw joint samples from a GP prior

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

num_points = 100
# Index points should be a collection (100, here) of feature vectors. In this
# example, we're using 1-d vectors, so we just need to reshape the output from
# np.linspace, to give a shape of (100, 1).
index_points = np.expand_dims(np.linspace(-1., 1., num_points), -1)

# Define a kernel with default parameters.
kernel = psd_kernels.ExponentiatedQuadratic()

gp = tfd.GaussianProcess(kernel, index_points)

samples = gp.sample(10)
# ==> 10 independently drawn, joint samples at `index_points`

noisy_gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=index_points,
    observation_noise_variance=.05)
noisy_samples = noise_gp.sample(10)
# ==> 10 independently drawn, noisy joint samples at `index_points`
```

##### Optimize kernel parameters via maximum marginal likelihood.

```python
# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observed_index_points = np.expand_dims(np.random.uniform(-1., 1., 50), -1)
# Squeeze to take the shape from [50, 1] to [50].
observed_values = f(observed_index_points)

# Define a kernel with trainable parameters.
kernel = psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.get_variable('amplitude', np.float32),
    length_scale=tf.get_variable('length_scale', np.float32))

gp = tfp.GaussianProcess(kernel, observed_index_points)
neg_log_likelihood = -gp.log_prob(observed_values)

optimize = tf.train.AdamOptimize().minimize(neg_log_likelihood)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(1000):
    _, nll_ = sess.run([optimize, nll])
    if i % 100 == 0:
      print("Step {}: NLL = {}".format(i, nll_))
  print("Final NLL = {}".format(nll_))
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    kernel,
    index_points,
    mean_fn=None,
    observation_noise_variance=0.0,
    jitter=1e-06,
    validate_args=False,
    allow_nan_stats=False,
    name='GaussianProcess'
)
```

Instantiate a GaussianProcess Distribution.

#### Args:

* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
    GP's covariance function.
* <b>`index_points`</b>: `float` `Tensor` representing finite (batch of) vector(s) of
    points in the index set over which the GP is defined. Shape has the form
    `[b1, ..., bB, e, f1, ..., fF]` where `F` is the number of feature
    dimensions and must equal `kernel.feature_ndims` and `e` is the number
    (size) of index points in each batch. Ultimately this distribution
    corresponds to a `e`-dimensional multivariate normal. The batch shape
    must be broadcastable with `kernel.batch_shape` and any batch dims
    yielded by `mean_fn`.
* <b>`mean_fn`</b>: Python `callable` that acts on `index_points` to produce a (batch
    of) vector(s) of mean values at `index_points`. Takes a `Tensor` of
    shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
    broadcastable with `[b1, ..., bB]`. Default value: `None` implies
    constant zero function.
* <b>`observation_noise_variance`</b>: `float` `Tensor` representing the variance
    of the noise in the Normal likelihood distribution of the model. May be
    batched, in which case the batch shape must be broadcastable with the
    shapes of all other batched parameters (`kernel.batch_shape`,
    `index_points`, etc.).
    Default value: `0.`
* <b>`jitter`</b>: `float` scalar `Tensor` added to the diagonal of the covariance
    matrix to ensure positive definiteness of the covariance matrix.
    Default value: `1e-6`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
    statistics (e.g., mean, mode, variance) use the value "`NaN`" to
    indicate the result is undefined. When `False`, an exception is raised
    if one or more of the statistic's batch members are undefined.
    Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
    Default value: "GaussianProcess".


#### Raises:

* <b>`ValueError`</b>: if `mean_fn` is not `None` and is not callable.



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

<h3 id="bijector"><code>bijector</code></h3>

Function transforming x => y.

<h3 id="distribution"><code>distribution</code></h3>

Base distribution, p(x).

<h3 id="dtype"><code>dtype</code></h3>

The `DType` of `Tensor`s handled by this `Distribution`.

<h3 id="event_shape"><code>event_shape</code></h3>

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

#### Returns:

* <b>`event_shape`</b>: `TensorShape`, possibly unknown.

<h3 id="index_points"><code>index_points</code></h3>



<h3 id="jitter"><code>jitter</code></h3>



<h3 id="kernel"><code>kernel</code></h3>



<h3 id="loc"><code>loc</code></h3>

The `loc` `Tensor` in `Y = scale @ X + loc`.

<h3 id="mean_fn"><code>mean_fn</code></h3>



<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this `Distribution`.

<h3 id="observation_noise_variance"><code>observation_noise_variance</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Dictionary of parameters used to instantiate this `Distribution`.

<h3 id="reparameterization_type"><code>reparameterization_type</code></h3>

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

#### Returns:

An instance of `ReparameterizationType`.

<h3 id="scale"><code>scale</code></h3>

The `scale` `LinearOperator` in `Y = scale @ X + loc`.

<h3 id="validate_args"><code>validate_args</code></h3>

Python `bool` indicating possibly expensive checks are enabled.



## Methods

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
    name='cdf'
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
covariance(name='covariance')
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
one another and permit densities `p(x) dr(x)` and `q(x) dr(x)`, (Shanon)
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
    representing `n` different calculations of (Shanon) cross entropy.

<h3 id="entropy"><code>entropy</code></h3>

``` python
entropy(name='entropy')
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
denotes (Shanon) cross entropy, and `H[.]` denotes (Shanon) entropy.

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
    name='log_cdf'
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


#### Returns:

* <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

<h3 id="log_prob"><code>log_prob</code></h3>

``` python
log_prob(
    value,
    name='log_prob'
)
```

Log probability density/mass function.


Additional documentation from `MultivariateNormalLinearOperator`:

`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

#### Args:

* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.


#### Returns:

* <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

<h3 id="log_survival_function"><code>log_survival_function</code></h3>

``` python
log_survival_function(
    value,
    name='log_survival_function'
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


#### Returns:

`Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
  `self.dtype`.

<h3 id="mean"><code>mean</code></h3>

``` python
mean(name='mean')
```

Mean.

<h3 id="mode"><code>mode</code></h3>

``` python
mode(name='mode')
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
    name='prob'
)
```

Probability density/mass function.


Additional documentation from `MultivariateNormalLinearOperator`:

`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

#### Args:

* <b>`value`</b>: `float` or `double` `Tensor`.
* <b>`name`</b>: Python `str` prepended to names of ops created by this function.


#### Returns:

* <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

<h3 id="quantile"><code>quantile</code></h3>

``` python
quantile(
    value,
    name='quantile'
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


#### Returns:

* <b>`quantile`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

<h3 id="sample"><code>sample</code></h3>

``` python
sample(
    sample_shape=(),
    seed=None,
    name='sample'
)
```

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

#### Args:

* <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
* <b>`seed`</b>: Python integer seed for RNG
* <b>`name`</b>: name to give to the op.


#### Returns:

* <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.

<h3 id="stddev"><code>stddev</code></h3>

``` python
stddev(name='stddev')
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


#### Returns:

* <b>`stddev`</b>: Floating-point `Tensor` with shape identical to
    `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.

<h3 id="survival_function"><code>survival_function</code></h3>

``` python
survival_function(
    value,
    name='survival_function'
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


#### Returns:

`Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
  `self.dtype`.

<h3 id="variance"><code>variance</code></h3>

``` python
variance(name='variance')
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


#### Returns:

* <b>`variance`</b>: Floating-point `Tensor` with shape identical to
    `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.



