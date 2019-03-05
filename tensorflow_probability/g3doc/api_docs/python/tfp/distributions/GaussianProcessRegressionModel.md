<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.GaussianProcessRegressionModel" />
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
<meta itemprop="property" content="observation_index_points"/>
<meta itemprop="property" content="observation_noise_variance"/>
<meta itemprop="property" content="observations"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="predictive_noise_variance"/>
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

# tfp.distributions.GaussianProcessRegressionModel

## Class `GaussianProcessRegressionModel`

Inherits From: [`MultivariateNormalLinearOperator`](../../tfp/distributions/MultivariateNormalLinearOperator.md)

Posterior predictive distribution in a conjugate GP regression model.

This class represents the distribution over function values at a set of points
in some index set, conditioned on noisy observations at some other set of
points. More specifically, we assume a Gaussian process prior, `f ~ GP(m, k)`
with IID normal noise on observations of function values. In this model
posterior inference can be done analytically. This `Distribution` is
parameterized by

  * the mean and covariance functions of the GP prior,
  * the set of (noisy) observations and index points to which they correspond,
  * the set of index points at which the resulting posterior predictive
    distribution over function values is defined,
  * the observation noise variance,
  * jitter, to compensate for numerical instability of Cholesky decomposition,

in addition to the usual params like `validate_args` and `allow_nan_stats`.

#### Mathematical Details

Gaussian process regression (GPR) assumes a Gaussian process (GP) prior and a
normal likelihood as a generative model for data. Given GP mean function `m`,
covariance kernel `k`, and observation noise variance `v`, we have

```none
  f ~ GP(m, k)

                   iid
  (y[i] | f, x[i])  ~  Normal(f(x[i]), v),   i = 1, ... , N
```

where `y[i]` are the noisy observations of function values at points `x[i]`.

In practice, `f` is an infinite object (eg, a function over `R^n`) which can't
be realized on a finite machine, but fortunately the marginal distribution
over function values at a finite set of points is just a multivariate normal
with mean and covariance given by the mean and covariance functions applied at
our finite set of points (see [Rasmussen and Williams, 2006][1] for a more
extensive discussion of these facts).

We spell out the generative model in detail below, but first, a digression on
notation. In what follows we drop the indices on vectorial objects such as
`x[i]`, it being implied that we are generally considering finite collections
of index points and corresponding function values and noisy observations
thereof. Thus `x` should be considered to stand for a collection of index
points (indeed, themselves often vectorial). Furthermore:

  * `f(x)` refers to the collection of function values at the index points in
    the collection `x`",
  * `m(t)` refers to the collection of values of the mean function at the
    index points in the collection `t`, and
  * `k(x, t)` refers to the *matrix* whose entries are values of the kernel
    function `k` at all pairs of index points from `x` and `t`.

With these conventions in place, we may write

```none
  (f(x) | x) ~ MVN(m(x), k(x, x))

  (y | f(x), x) ~ Normal(f(x), v)
```

When we condition on observed data `y` at the points `x`, we can derive the
posterior distribution over function values `f(x)` at those points. We can
then compute the posterior predictive distribution over function values `f(t)`
at a new set of points `t`, conditional on those observed data.

```none
  (f(t) | t, x, f(x)) ~ MVN(loc, cov)

  where

  loc = k(t, x) @ inv(k(x, x) + v * I) @ (y - loc)
  cov = k(t, t) - k(t, x) @ inv(k(x, x) + v * I) @ k(x, t)
```

where `I` is the identity matrix of appropriate dimension. Finally, the
distribution over noisy observations at the new set of points `t` is obtained
by adding IID noise from `Normal(0., observation_noise_variance)`.

#### Examples

##### Draw joint samples from the posterior predictive distribution in a GP
regression model

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

# Generate noisy observations from a known function at some random points.
observation_noise_variance = .5
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
observations = (f(observation_index_points) +
                np.random.normal(0., np.sqrt(observation_noise_variance)))

index_points = np.linspace(-1., 1., 100)[..., np.newaxis]

kernel = psd_kernels.MaternFiveHalves()

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=index_points,
    observation_index_points=observation_index_points,
    observations=observations,
    observation_noise_variance=observation_noise_variance)

samples = gprm.sample(10)
# ==> 10 independently drawn, joint samples at `index_points`.
```

Above, we have used the kernel with default parameters, which are unlikely to
be good. Instead, we can train the kernel hyperparameters on the data, as in
the next example.

##### Optimize model parameters via maximum marginal likelihood

Here we learn the kernel parameters as well as the observation noise variance
using gradient descent on the maximum marginal likelihood.

```python
# Suppose we have some data from a known function. Note the index points in
# general have shape `[b1, ..., bB, f1, ..., fF]` (here we assume `F == 1`),
# so we need to explicitly consume the feature dimensions (just the last one
# here).
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)

observation_index_points = np.random.uniform(-1., 1., 50)[..., np.newaxis]
observations = f(observation_index_points) + np.random.normal(0., .05, 50)

# Define a kernel with trainable parameters. Note we transform the trainable
# variables to apply a positivity constraint.
amplitude = tf.exp(tf.Variable(np.float64(0)), name='amplitude')
length_scale = tf.exp(tf.Variable(np.float64(0)), name='length_scale')
kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

observation_noise_variance = tf.exp(
    tf.Variable(np.float64(-5)), name='observation_noise_variance')

# We'll use an unconditioned GP to train the kernel parameters.
gp = tfd.GaussianProcess(
    kernel=kernel,
    index_points=observation_index_points,
    observation_noise_variance=observation_noise_variance)
neg_log_likelihood = -gp.log_prob(observations)

optimizer = tf.train.AdamOptimizer(learning_rate=.05, beta1=.5, beta2=.99)
optimize = optimizer.minimize(neg_log_likelihood)

# We can construct the posterior at a new set of `index_points` using the same
# kernel (with the same parameters, which we'll optimize below).
index_points = np.linspace(-1., 1., 100)[..., np.newaxis]
gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=index_points,
    observation_index_points=observation_index_points,
    observations=observations,
    observation_noise_variance=observation_noise_variance)

samples = gprm.sample(10)
# ==> 10 independently drawn, joint samples at `index_points`.

# Now execute the above ops in a Session, first training the model
# parameters, then drawing and plotting posterior samples.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for i in range(1000):
    _, neg_log_likelihood_ = sess.run([optimize, neg_log_likelihood])
    if i % 100 == 0:
      print("Step {}: NLL = {}".format(i, neg_log_likelihood_))

  print("Final NLL = {}".format(neg_log_likelihood_))
  samples_ = sess.run(samples)

  plt.scatter(np.squeeze(observation_index_points), observations)
  plt.plot(np.stack([index_points[:, 0]]*10).T, samples_.T, c='r', alpha=.2)
```

##### Marginalization of model hyperparameters

Here we use TensorFlow Probability's MCMC functionality to perform
marginalization of the model hyperparameters: kernel params as well as
observation noise variance.

```python
f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
observation_index_points = np.random.uniform(-1., 1., 25)[..., np.newaxis]
observations = np.random.normal(f(observation_index_points), .05)

def joint_log_prob(
    index_points, observations, amplitude, length_scale, noise_variance):

  # Hyperparameter Distributions.
  rv_amplitude = tfd.LogNormal(np.float64(0.), np.float64(1))
  rv_length_scale = tfd.LogNormal(np.float64(0.), np.float64(1))
  rv_noise_variance = tfd.LogNormal(np.float64(0.), np.float64(1))

  gp = tfd.GaussianProcess(
      kernel=psd_kernels.ExponentiatedQuadratic(amplitude, length_scale),
      index_points=index_points,
      observation_noise_variance=noise_variance)

  return (
      rv_amplitude.log_prob(amplitude) +
      rv_length_scale.log_prob(length_scale) +
      rv_noise_variance.log_prob(noise_variance) +
      gp.log_prob(observations)
  )

initial_chain_states = [
    1e-1 * tf.ones([], dtype=np.float64, name='init_amplitude'),
    1e-1 * tf.ones([], dtype=np.float64, name='init_length_scale'),
    1e-1 * tf.ones([], dtype=np.float64, name='init_obs_noise_variance')
]

# Since HMC operates over unconstrained space, we need to transform the
# samples so they live in real-space.
unconstraining_bijectors = [
    tfp.bijectors.Softplus(),
    tfp.bijectors.Softplus(),
    tfp.bijectors.Softplus(),
]

def unnormalized_log_posterior(amplitude, length_scale, noise_variance):
  return joint_log_prob(
      observation_index_points, observations, amplitude, length_scale,
      noise_variance)

num_results = 200
[
    amplitudes,
    length_scales,
    observation_noise_variances
], kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=500,
    num_steps_between_results=3,
    current_state=initial_chain_states,
    kernel=tfp.mcmc.TransformedTransitionKernel(
        inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_log_posterior,
            step_size=[np.float64(.15)],
            num_leapfrog_steps=3),
        bijector=unconstraining_bijectors))

# Now we can sample from the posterior predictive distribution at a new set
# of index points.
gprm = tfd.GaussianProcessRegressionModel(
    # Batch of `num_results` kernels parameterized by the MCMC samples.
    kernel=psd_kernels.ExponentiatedQuadratic(amplitudes, length_scales),
    index_points=np.linspace(-2., 2., 200)[..., np.newaxis],
    observation_index_points=observation_index_points,
    observations=observations,
    # We reshape this to align batch dimensions.
    observation_noise_variance=observation_noise_variances[..., np.newaxis])
samples = gprm.sample()

with tf.Session() as sess:
  kernel_results_, samples_ = sess.run([kernel_results, samples])

  print("Acceptance rate: {}".format(
      np.mean(kernel_results_.inner_results.is_accepted)))

  # Plot posterior samples and their mean, target function, and observations.
  plt.plot(np.stack([index_points[:, 0]]*num_results).T,
           samples_.T,
           c='r',
           alpha=.01)
  plt.plot(index_points[:, 0], np.mean(samples_, axis=0), c='k')
  plt.plot(index_points[:, 0], f(index_points))
  plt.scatter(observation_index_points[:, 0], observations)
```

#### References
[1]: Carl Rasmussen, Chris Williams. Gaussian Processes For Machine Learning,
     2006.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    kernel,
    index_points,
    observation_index_points=None,
    observations=None,
    observation_noise_variance=0.0,
    predictive_noise_variance=None,
    mean_fn=None,
    jitter=1e-06,
    validate_args=False,
    allow_nan_stats=False,
    name='GaussianProcessRegressionModel'
)
```

Construct a GaussianProcessRegressionModel instance.

#### Args:

* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
    GP's covariance function.
* <b>`index_points`</b>: `float` `Tensor` representing finite collection, or batch of
    collections, of points in the index set over which the GP is defined.
    Shape has the form `[b1, ..., bB, e, f1, ..., fF]` where `F` is the
    number of feature dimensions and must equal `kernel.feature_ndims` and
    `e` is the number (size) of index points in each batch. Ultimately this
    distribution corresponds to an `e`-dimensional multivariate normal. The
    batch shape must be broadcastable with `kernel.batch_shape` and any
    batch dims yielded by `mean_fn`.
* <b>`observation_index_points`</b>: `float` `Tensor` representing finite collection,
    or batch of collections, of points in the index set for which some data
    has been observed. Shape has the form `[b1, ..., bB, e, f1, ..., fF]`
    where `F` is the number of feature dimensions and must equal
    `kernel.feature_ndims`, and `e` is the number (size) of index points in
    each batch. `[b1, ..., bB, e]` must be broadcastable with the shape of
    `observations`, and `[b1, ..., bB]` must be broadcastable with the
    shapes of all other batched parameters (`kernel.batch_shape`,
    `index_points`, etc). The default value is `None`, which corresponds to
    the empty set of observations, and simply results in the prior
    predictive model (a GP with noise of variance
    `predictive_noise_variance`).
* <b>`observations`</b>: `float` `Tensor` representing collection, or batch of
    collections, of observations corresponding to
    `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
    must be brodcastable with the batch and example shapes of
    `observation_index_points`. The batch shape `[b1, ..., bB]` must be
    broadcastable with the shapes of all other batched parameters
    (`kernel.batch_shape`, `index_points`, etc.). The default value is
    `None`, which corresponds to the empty set of observations, and simply
    results in the prior predictive model (a GP with noise of variance
    `predictive_noise_variance`).
* <b>`observation_noise_variance`</b>: `float` `Tensor` representing the variance
    of the noise in the Normal likelihood distribution of the model. May be
    batched, in which case the batch shape must be broadcastable with the
    shapes of all other batched parameters (`kernel.batch_shape`,
    `index_points`, etc.).
    Default value: `0.`
* <b>`predictive_noise_variance`</b>: `float` `Tensor` representing the variance in
    the posterior predictive model. If `None`, we simply re-use
    `observation_noise_variance` for the posterior predictive noise. If set
    explicitly, however, we use this value. This allows us, for example, to
    omit predictive noise variance (by setting this to zero) to obtain
    noiseless posterior predictions of function values, conditioned on noisy
    observations.
* <b>`mean_fn`</b>: Python `callable` that acts on `index_points` to produce a
    collection, or batch of collections, of mean values at `index_points`.
    Takes a `Tensor` of shape `[b1, ..., bB, f1, ..., fF]` and returns a
    `Tensor` whose shape is broadcastable with `[b1, ..., bB]`.
    Default value: `None` implies the constant zero function.
* <b>`jitter`</b>: `float` scalar `Tensor` added to the diagonal of the covariance
    matrix to ensure positive definiteness of the covariance matrix.
    Default value: `1e-6`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
    statistics (e.g., mean, mode, variance) use the value `NaN` to
    indicate the result is undefined. When `False`, an exception is raised
    if one or more of the statistic's batch members are undefined.
    Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
    Default value: 'GaussianProcessRegressionModel'.


#### Raises:

* <b>`ValueError`</b>: if either
    - only one of `observations` and `observation_index_points` is given, or
    - `mean_fn` is not `None` and not callable.



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

<h3 id="observation_index_points"><code>observation_index_points</code></h3>



<h3 id="observation_noise_variance"><code>observation_noise_variance</code></h3>



<h3 id="observations"><code>observations</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Dictionary of parameters used to instantiate this `Distribution`.

<h3 id="predictive_noise_variance"><code>predictive_noise_variance</code></h3>



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



