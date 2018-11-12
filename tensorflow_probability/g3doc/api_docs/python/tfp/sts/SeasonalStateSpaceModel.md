<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.SeasonalStateSpaceModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="drift_scale"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_seasons"/>
<meta itemprop="property" content="num_steps_per_season"/>
<meta itemprop="property" content="observation_noise_scale"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="__init__"/>
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
<meta itemprop="property" content="prob"/>
<meta itemprop="property" content="quantile"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="stddev"/>
<meta itemprop="property" content="survival_function"/>
<meta itemprop="property" content="variance"/>
</div>

# tfp.sts.SeasonalStateSpaceModel

## Class `SeasonalStateSpaceModel`

Inherits From: [`LinearGaussianStateSpaceModel`](../../tfp/distributions/LinearGaussianStateSpaceModel.md)

State space model for a seasonal effect.

A state space model (SSM) posits a set of latent (unobserved) variables that
evolve over time with dynamics specified by a probabilistic transition model
`p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
observation model conditioned on the current state, `p(x[t] | z[t])`. The
special case where both the transition and observation models are Gaussians
with mean specified as a linear function of the inputs, is known as a linear
Gaussian state space model and supports tractable exact probabilistic
calculations; see <a href="../../tfp/distributions/LinearGaussianStateSpaceModel.md"><code>tfp.distributions.LinearGaussianStateSpaceModel</code></a> for
details.

A seasonal effect model is a special case of a linear Gaussian SSM. The
latent states represent an unknown effect from each of several 'seasons';
these are generally not meteorological seasons, but represent regular
recurring patterns such as hour-of-day or day-of-week effects. The effect of
each season drifts from one occurrence to the next, following a Gaussian
random walk:

```python
effects[season, occurrence[i]] = (
  effects[season, occurrence[i-1]] + Normal(loc=0., scale=drift_scale))
```

The latent state has dimension `num_seasons`, containing one effect for each
seasonal component. The parameters `drift_scale` and
`observation_noise_scale` are each (a batch of) scalars. The batch shape of
this `Distribution` is the broadcast batch shape of these parameters and of
the `initial_state_prior`.

#### Mathematical Details

The seasonal effect model implements a
<a href="../../tfp/distributions/LinearGaussianStateSpaceModel.md"><code>tfp.distributions.LinearGaussianStateSpaceModel</code></a> with
`latent_size = num_seasons` and `observation_size = 1`. The latent state
is organized so that the *current* seasonal effect is always in the first
(zeroth) dimension. The transition model rotates the latent state to shift
to a new effect at the end of each season:

```
transition_matrix[t] = (permutation_matrix([1, 2, ..., num_seasons-1, 0])
                        if season_is_changing(t)
                        else eye(num_seasons)
transition_noise[t] ~ Normal(loc=0., scale_diag=(
                             [drift_scale, 0, ..., 0]
                             if season_is_changing(t)
                             else [0, 0, ..., 0]))
```

where `season_is_changing(t)` is `True` if ``t `mod`
sum(num_steps_per_season)`` is in the set of final days for each season,
given by `cumsum(num_steps_per_season) - 1`. The observation model always
picks out the effect for the current season, i.e., the first element of
the latent state:

```
observation_matrix = [[1., 0., ..., 0.]]
observation_noise ~ Normal(loc=0, scale=observation_noise_scale)
```

#### Examples

A state-space model with day-of-week seasonality on hourly data:

```python
day_of_week = SeasonalStateSpaceModel(
  num_timesteps=30,
  num_seasons=7,
  drift_scale=0.1,
  initial_state_prior=tfd.MultivariateNormalDiag(
    scale_diag=tf.ones([7], dtype=tf.float32),
  num_steps_per_season=24)
```

A model with basic month-of-year seasonality on daily data, demonstrating
seasons of varying length:

```python
month_of_year = SeasonalStateSpaceModel(
  num_timesteps=2 * 365,  # 2 years
  num_seasons=12,
  drift_scale=0.1,
  initial_state_prior=tfd.MultivariateNormalDiag(
    scale_diag=tf.ones([12], dtype=tf.float32)),
  num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
  initial_step=22)
```

Note that we've used `initial_step=22` to denote that the model begins
on January 23 (steps are zero-indexed). A general implementation of
month-of-year seasonality would require additional logic; this
version works over time periods not involving a leap year.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_timesteps,
    num_seasons,
    drift_scale,
    initial_state_prior,
    observation_noise_scale=0.0,
    num_steps_per_season=1,
    initial_step=0,
    validate_args=False,
    allow_nan_stats=True,
    name=None
)
```

Build a state space model implementing seasonal effects.

#### Args:

* <b>`num_timesteps`</b>: Scalar `int` `Tensor` number of timesteps to model
    with this distribution.
* <b>`num_seasons`</b>: Scalar Python `int` number of seasons.
* <b>`drift_scale`</b>: Scalar (any additional dimensions are treated as batch
    dimensions) `float` `Tensor` indicating the standard deviation of the
    change in effect between consecutive occurrences of a given season.
    This is assumed to be the same for all seasons.
* <b>`initial_state_prior`</b>: instance of `tfd.MultivariateNormal`
    representing the prior distribution on latent states; must
    have event shape `[num_seasons]`.
* <b>`observation_noise_scale`</b>: Scalar (any additional dimensions are
    treated as batch dimensions) `float` `Tensor` indicating the standard
    deviation of the observation noise.
    Default value: 0.
* <b>`num_steps_per_season`</b>: Python `int` number of steps in each
    season. This may be either a scalar (shape `[]`), in which case all
    seasons have the same length, or a NumPy array of shape `[num_seasons]`.
    Default value: 1.
* <b>`initial_step`</b>: Optional scalar `int` `Tensor` specifying the starting
    timestep.
    Default value: 0.
* <b>`validate_args`</b>: Python `bool`. Whether to validate input
    with asserts. If `validate_args` is `False`, and the inputs are
    invalid, correct behavior is not guaranteed.
    Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`. If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member. If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
    Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this class.
    Default value: "SeasonalStateSpaceModel".


#### Raises:

* <b>`ValueError`</b>: if `num_steps_per_season` has invalid shape (neither
    scalar nor `[num_seasons]`).



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

<h3 id="drift_scale"><code>drift_scale</code></h3>

Standard deviation of the drift in effects between seasonal cycles.

<h3 id="dtype"><code>dtype</code></h3>

The `DType` of `Tensor`s handled by this `Distribution`.

<h3 id="event_shape"><code>event_shape</code></h3>

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

#### Returns:

* <b>`event_shape`</b>: `TensorShape`, possibly unknown.

<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this `Distribution`.

<h3 id="num_seasons"><code>num_seasons</code></h3>

Number of seasons.

<h3 id="num_steps_per_season"><code>num_steps_per_season</code></h3>

Number of steps in each season.

<h3 id="observation_noise_scale"><code>observation_noise_scale</code></h3>

Standard deviation of the observation noise.

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



