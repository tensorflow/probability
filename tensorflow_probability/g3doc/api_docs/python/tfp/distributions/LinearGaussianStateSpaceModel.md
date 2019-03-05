<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.LinearGaussianStateSpaceModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="backward_smoothing_pass"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="cdf"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="covariance"/>
<meta itemprop="property" content="cross_entropy"/>
<meta itemprop="property" content="entropy"/>
<meta itemprop="property" content="event_shape_tensor"/>
<meta itemprop="property" content="forward_filter"/>
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
<meta itemprop="property" content="posterior_marginals"/>
<meta itemprop="property" content="prob"/>
<meta itemprop="property" content="quantile"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="stddev"/>
<meta itemprop="property" content="survival_function"/>
<meta itemprop="property" content="variance"/>
</div>

# tfp.distributions.LinearGaussianStateSpaceModel

## Class `LinearGaussianStateSpaceModel`

Inherits From: [`Distribution`](../../tfp/distributions/Distribution.md)

Observation distribution from a linear Gaussian state space model.

The state space model, sometimes called a Kalman filter, posits a
latent state vector `z_t` of dimension `latent_size` that evolves
over time following linear Gaussian transitions,

```z_{t+1} = F * z_t + N(b; Q)```

for transition matrix `F`, bias `b` and covariance matrix
`Q`. At each timestep, we observe a noisy projection of the
latent state `x_t = H * z_t + N(c; R)`. The transition and
observation models may be fixed or may vary between timesteps.

This Distribution represents the marginal distribution on
observations, `p(x)`. The marginal `log_prob` is computed by
Kalman filtering [1], and `sample` by an efficient forward
recursion. Both operations require time linear in `T`, the total
number of timesteps.

#### Shapes

The event shape is `[num_timesteps, observation_size]`, where
`observation_size` is the dimension of each observation `x_t`.
The observation and transition models must return consistent
shapes.

This implementation supports vectorized computation over a batch of
models. All of the parameters (prior distribution, transition and
observation operators and noise models) must have a consistent
batch shape.

#### Time-varying processes

Any of the model-defining parameters (prior distribution, transition
and observation operators and noise models) may be specified as a
callable taking an integer timestep `t` and returning a
time-dependent value. The dimensionality (`latent_size` and
`observation_size`) must be the same at all timesteps.

Importantly, the timestep is passed as a `Tensor`, not a Python
integer, so any conditional behavior must occur *inside* the
TensorFlow graph. For example, suppose we want to use a different
transition model on even days than odd days. It does *not* work to
write

```python
def transition_matrix(t):
  if t % 2 == 0:
    return even_day_matrix
  else:
    return odd_day_matrix
```

since the value of `t` is not fixed at graph-construction
time. Instead we need to write

```python
def transition_matrix(t):
  return tf.cond(tf.equal(tf.mod(t, 2), 0),
                 lambda : even_day_matrix,
                 lambda : odd_day_matrix)
```

so that TensorFlow can switch between operators appropriately at
runtime.

#### Examples

Consider a simple tracking model. The two-dimensional latent state
represents the true position of a vehicle, and at each timestep we
see a noisy observation of this position (e.g., a GPS reading). The
vehicle is assumed to move by a random walk with standard deviation
`step_std` at each step, and observation noise level `std`. We build
the distribution over noisy observations by

```python
ndims = 2
step_std = 1.0
noise_std = 5.0
model = LinearGaussianStateSpaceModel(
  num_timesteps=100,
  transition_matrix=tfl.LinearOperatorIdentity(ndims),
  transition_noise=tfd.MultivariateNormalDiag(
   scale_diag=step_std**2 * tf.ones([ndims])),
  observation_matrix=tfl.LinearOperatorIdentity(ndims),
  observation_noise=tfd.MultivariateNormalDiag(
   scale_diag=noise_std**2 * tf.ones([ndims])),
  initial_state_prior=tfd.MultivariateNormalDiag(
   scale_diag=tf.ones([ndims])))
)
```

using the identity matrix for the transition and observation
operators. We can then use this model to generate samples,
compute marginal likelihood of observed sequences, and
perform posterior inference.

```python
x = model.sample(5) # Sample from the prior on sequences of observations.
lp = model.log_prob(x) # Marginal likelihood of a (batch of) observations.

# Compute the filtered posterior on latent states given observations,
# and extract the mean and covariance for the current (final) timestep.
_, filtered_means, filtered_covs, _, _ = model.forward_filter(x)
final_step = tfd.MultivariateNormalFullCovariance(
              loc=filtered_means[..., -1, :],
              scale=filtered_covs[..., -1, :])
```

* TODO(davmre): implement and describe full posterior inference / smoothing.

* TODO(davmre): show example of fitting parameters.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_timesteps,
    transition_matrix,
    transition_noise,
    observation_matrix,
    observation_noise,
    initial_state_prior,
    initial_step=0,
    validate_args=False,
    allow_nan_stats=True,
    name='LinearGaussianStateSpaceModel'
)
```

Initialize a `LinearGaussianStateSpaceModel.

#### Args:

* <b>`num_timesteps`</b>: Integer `Tensor` total number of timesteps.
* <b>`transition_matrix`</b>: A transition operator, represented by a Tensor or
    LinearOperator of shape `[latent_size, latent_size]`, or by a
    callable taking as argument a scalar integer Tensor `t` and
    returning a Tensor or LinearOperator representing the transition
    operator from latent state at time `t` to time `t + 1`.
* <b>`transition_noise`</b>: An instance of
    `tfd.MultivariateNormalLinearOperator` with event shape
    `[latent_size]`, representing the mean and covariance of the
    transition noise model, or a callable taking as argument a
    scalar integer Tensor `t` and returning such a distribution
    representing the noise in the transition from time `t` to time `t + 1`.
* <b>`observation_matrix`</b>: An observation operator, represented by a Tensor
    or LinearOperator of shape `[observation_size, latent_size]`,
    or by a callable taking as argument a scalar integer Tensor
    `t` and returning a timestep-specific Tensor or
    LinearOperator.
* <b>`observation_noise`</b>: An instance of
    `tfd.MultivariateNormalLinearOperator` with event shape
    `[observation_size]`, representing the mean and covariance of
    the observation noise model, or a callable taking as argument
    a scalar integer Tensor `t` and returning a timestep-specific
    noise model.
* <b>`initial_state_prior`</b>: An instance of `MultivariateNormalLinearOperator`
    representing the prior distribution on latent states; must
    have event shape `[latent_size]`.
* <b>`initial_step`</b>: optional `int` specifying the time of the first
    modeled timestep.  This is added as an offset when passing
    timesteps `t` to (optional) callables specifying
    timestep-specific transition and observation models.
* <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
    with asserts. If `validate_args` is `False`, and the inputs are
    invalid, correct behavior is not guaranteed.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
* <b>`name`</b>: The name to give Ops created by the initializer.



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

<h3 id="dtype"><code>dtype</code></h3>

The `DType` of `Tensor`s handled by this `Distribution`.

<h3 id="event_shape"><code>event_shape</code></h3>

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

#### Returns:

* <b>`event_shape`</b>: `TensorShape`, possibly unknown.

<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this `Distribution`.

<h3 id="parameters"><code>parameters</code></h3>

Dictionary of parameters used to instantiate this `Distribution`.

<h3 id="reparameterization_type"><code>reparameterization_type</code></h3>

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

#### Returns:

An instance of `ReparameterizationType`.

<h3 id="validate_args"><code>validate_args</code></h3>

Python `bool` indicating possibly expensive checks are enabled.



## Methods

<h3 id="backward_smoothing_pass"><code>backward_smoothing_pass</code></h3>

``` python
backward_smoothing_pass(
    filtered_means,
    filtered_covs,
    predicted_means,
    predicted_covs
)
```

Run the backward pass in Kalman smoother.

The backward smoothing is using Rauch, Tung and Striebel smoother as
as discussed in section 18.3.2 of Kevin P. Murphy, 2012, Machine Learning:
A Probabilistic Perspective, The MIT Press. The inputs are returned by
`forward_filter` function.

#### Args:

* <b>`filtered_means`</b>: Means of the per-timestep filtered marginal
    distributions p(z_t | x_{:t}), as a Tensor of shape
    `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`.
* <b>`filtered_covs`</b>: Covariances of the per-timestep filtered marginal
    distributions p(z_t | x_{:t}), as a Tensor of shape
    `batch_shape + [num_timesteps, latent_size, latent_size]`.
* <b>`predicted_means`</b>: Means of the per-timestep predictive
     distributions over latent states, p(z_{t+1} | x_{:t}), as a
     Tensor of shape `sample_shape(x) + batch_shape +
     [num_timesteps, latent_size]`.
* <b>`predicted_covs`</b>: Covariances of the per-timestep predictive
     distributions over latent states, p(z_{t+1} | x_{:t}), as a
     Tensor of shape `batch_shape + [num_timesteps, latent_size,
     latent_size]`.


#### Returns:

* <b>`posterior_means`</b>: Means of the smoothed marginal distributions
    p(z_t | x_{1:T}), as a Tensor of shape
    `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`,
    which is of the same shape as filtered_means.
* <b>`posterior_covs`</b>: Covariances of the smoothed marginal distributions
    p(z_t | x_{1:T}), as a Tensor of shape
    `batch_shape + [num_timesteps, latent_size, latent_size]`.
    which is of the same shape as filtered_covs.

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

<h3 id="forward_filter"><code>forward_filter</code></h3>

``` python
forward_filter(x)
```

Run a Kalman filter over a provided sequence of outputs.

Note that the returned values `filtered_means`, `predicted_means`, and
`observation_means` depend on the observed time series `x`, while the
corresponding covariances are independent of the observed series; i.e., they
depend only on the model itself. This means that the mean values have shape
`concat([sample_shape(x), batch_shape, [num_timesteps,
{latent/observation}_size]])`, while the covariances have shape
`concat[(batch_shape, [num_timesteps, {latent/observation}_size,
{latent/observation}_size]])`, which does not depend on the sample shape.

#### Args:

* <b>`x`</b>: a float-type `Tensor` with rightmost dimensions
    `[num_timesteps, observation_size]` matching
    `self.event_shape`. Additional dimensions must match or be
    broadcastable to `self.batch_shape`; any further dimensions
    are interpreted as a sample shape.


#### Returns:

* <b>`log_likelihoods`</b>: Per-timestep log marginal likelihoods `log
    p(x_t | x_{:t-1})` evaluated at the input `x`, as a `Tensor`
    of shape `sample_shape(x) + batch_shape + [num_timesteps].`
* <b>`filtered_means`</b>: Means of the per-timestep filtered marginal
     distributions p(z_t | x_{:t}), as a Tensor of shape
    `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`.
* <b>`filtered_covs`</b>: Covariances of the per-timestep filtered marginal
     distributions p(z_t | x_{:t}), as a Tensor of shape
    `batch_shape + [num_timesteps, latent_size, latent_size]`.
* <b>`predicted_means`</b>: Means of the per-timestep predictive
     distributions over latent states, p(z_{t+1} | x_{:t}), as a
     Tensor of shape `sample_shape(x) + batch_shape +
     [num_timesteps, latent_size]`.
* <b>`predicted_covs`</b>: Covariances of the per-timestep predictive
     distributions over latent states, p(z_{t+1} | x_{:t}), as a
     Tensor of shape `batch_shape + [num_timesteps, latent_size,
     latent_size]`.
* <b>`observation_means`</b>: Means of the per-timestep predictive
     distributions over observations, p(x_{t} | x_{:t-1}), as a
     Tensor of shape `sample_shape(x) + batch_shape +
     [num_timesteps, observation_size]`.
* <b>`observation_covs`</b>: Covariances of the per-timestep predictive
     distributions over observations, p(x_{t} | x_{:t-1}), as a
     Tensor of shape `batch_shape + [num_timesteps,
     observation_size, observation_size]`.

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

<h3 id="posterior_marginals"><code>posterior_marginals</code></h3>

``` python
posterior_marginals(x)
```

Run a Kalman smoother to return posterior mean and cov.

Note that the returned values `smoothed_means` depend on the observed
time series `x`, while the `smoothed_covs` are independent
of the observed series; i.e., they depend only on the model itself.
This means that the mean values have shape `concat([sample_shape(x),
batch_shape, [num_timesteps, {latent/observation}_size]])`,
while the covariances have shape `concat[(batch_shape, [num_timesteps,
{latent/observation}_size, {latent/observation}_size]])`, which
does not depend on the sample shape.

This function only performs smoothing. If the user wants the
intermediate values, which are returned by filtering pass `forward_filter`,
one could get it by:
```
(log_likelihoods,
 filtered_means, filtered_covs,
 predicted_means, predicted_covs,
 observation_means, observation_covs) = model.forward_filter(x)
smoothed_means, smoothed_covs = model.backward_smoothing_pass(x)
```
where `x` is an observation sequence.

#### Args:

* <b>`x`</b>: a float-type `Tensor` with rightmost dimensions
    `[num_timesteps, observation_size]` matching
    `self.event_shape`. Additional dimensions must match or be
    broadcastable to `self.batch_shape`; any further dimensions
    are interpreted as a sample shape.


#### Returns:

* <b>`smoothed_means`</b>: Means of the per-timestep smoothed
     distributions over latent states, p(x_{t} | x_{:T}), as a
     Tensor of shape `sample_shape(x) + batch_shape +
     [num_timesteps, observation_size]`.
* <b>`smoothed_covs`</b>: Covariances of the per-timestep smoothed
     distributions over latent states, p(x_{t} | x_{:T}), as a
     Tensor of shape `batch_shape + [num_timesteps,
     observation_size, observation_size]`.

<h3 id="prob"><code>prob</code></h3>

``` python
prob(
    value,
    name='prob'
)
```

Probability density/mass function.

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



