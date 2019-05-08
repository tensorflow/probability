<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.AutoregressiveStateSpaceModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="coefficients"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="level_scale"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="order"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
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
<meta itemprop="property" content="latents_to_observations"/>
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

# tfp.sts.AutoregressiveStateSpaceModel

## Class `AutoregressiveStateSpaceModel`

State space model for an autoregressive process.

Inherits From: [`LinearGaussianStateSpaceModel`](../../tfp/distributions/LinearGaussianStateSpaceModel.md)



Defined in [`python/sts/autoregressive.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/autoregressive.py).

<!-- Placeholder for "Used in" -->

A state space model (SSM) posits a set of latent (unobserved) variables that
evolve over time with dynamics specified by a probabilistic transition model
`p(z[t+1] | z[t])`. At each timestep, we observe a value sampled from an
observation model conditioned on the current state, `p(x[t] | z[t])`. The
special case where both the transition and observation models are Gaussians
with mean specified as a linear function of the inputs, is known as a linear
Gaussian state space model and supports tractable exact probabilistic
calculations; see <a href="../../tfp/distributions/LinearGaussianStateSpaceModel.md"><code>tfp.distributions.LinearGaussianStateSpaceModel</code></a> for
details.

In an autoregressive process, the expected level at each timestep is a linear
function of previous levels, with added Gaussian noise:

```python
level[t+1] = (sum(coefficients * levels[t:t-order:-1]) +
              Normal(0., level_scale))
 ```

The process is characterized by a vector `coefficients` whose size determines
the order of the process (how many previous values it looks at), and by
`level_scale`, the standard deviation of the noise added at each step.

This is formulated as a state space model by letting the latent state encode
the most recent values; see 'Mathematical Details' below.

The parameters `level_scale` and `observation_noise_scale` are each (a batch
of) scalars, and `coefficients` is a (batch) vector of size `[order]`. The
batch shape of this `Distribution` is the broadcast batch
shape of these parameters and of the `initial_state_prior`.

#### Mathematical Details

The autoregressive model implements a
<a href="../../tfp/distributions/LinearGaussianStateSpaceModel.md"><code>tfp.distributions.LinearGaussianStateSpaceModel</code></a> with `latent_size = order`
and `observation_size = 1`. The latent state vector encodes the recent history
of the process, with the current value in the topmost dimension. At each
timestep, the transition sums the previous values to produce the new expected
value, shifts all other values down by a dimension, and adds noise to the
current value. This is formally encoded by the transition model:

```
transition_matrix = [ coefs[0], coefs[1], ..., coefs[order]
                      1.,       0 ,       ..., 0.
                      0.,       1.,       ..., 0.
                      ...
                      0.,       0.,  ...,  1.,  0.            ]
transition_noise ~ N(loc=0., scale=diag([level_scale, 0., 0., ..., 0.]))
```

The observation model simply extracts the current (topmost) value, and
optionally adds independent noise at each step:

```
observation_matrix = [[1., 0., ..., 0.]]
observation_noise ~ N(loc=0, scale=observation_noise_scale)
```

Models with `observation_noise_scale = 0.` are AR processes in the formal
sense. Setting `observation_noise_scale` to a nonzero value corresponds to a
latent AR process observed under an iid noise model.

#### Examples

A simple model definition:

```python
ar_model = AutoregressiveStateSpaceModel(
    num_timesteps=50,
    coefficients=[0.8, -0.1],
    level_scale=0.5,
    initial_state_prior=tfd.MultivariateNormalDiag(
      scale_diag=[1., 1.]))

y = ar_model.sample() # y has shape [50, 1]
lp = ar_model.log_prob(y) # log_prob is scalar
```

Passing additional parameter dimensions constructs a batch of models. The
overall batch shape is the broadcast batch shape of the parameters:

```python
ar_model = AutoregressiveStateSpaceModel(
    num_timesteps=50,
    coefficients=[0.8, -0.1],
    level_scale=tf.ones([10]),
    initial_state_prior=tfd.MultivariateNormalDiag(
      scale_diag=tf.ones([10, 10, 2])))

y = ar_model.sample(5) # y has shape [5, 10, 10, 50, 1]
lp = ar_model.log_prob(y) # has shape [5, 10, 10]
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_timesteps,
    coefficients,
    level_scale,
    initial_state_prior,
    observation_noise_scale=0.0,
    initial_step=0,
    validate_args=False,
    name=None
)
```

Build a state space model implementing an autoregressive process.

#### Args:

* <b>`num_timesteps`</b>: Scalar `int` `Tensor` number of timesteps to model
  with this distribution.
* <b>`coefficients`</b>: `float` `Tensor` of shape `concat(batch_shape, [order])`
  defining  the autoregressive coefficients. The coefficients are defined
  backwards in time: `coefficients[0] * level[t] + coefficients[1] *
  level[t-1] + ... + coefficients[order-1] * level[t-order+1]`.
* <b>`level_scale`</b>: Scalar (any additional dimensions are treated as batch
  dimensions) `float` `Tensor` indicating the standard deviation of the
  transition noise at each step.
* <b>`initial_state_prior`</b>: instance of `tfd.MultivariateNormal`
  representing the prior distribution on latent states.  Must have
  event shape `[order]`.
* <b>`observation_noise_scale`</b>: Scalar (any additional dimensions are
  treated as batch dimensions) `float` `Tensor` indicating the standard
  deviation of the observation noise.
  Default value: 0.
* <b>`initial_step`</b>: Optional scalar `int` `Tensor` specifying the starting
  timestep.
  Default value: 0.
* <b>`validate_args`</b>: Python `bool`. Whether to validate input
  with asserts. If `validate_args` is `False`, and the inputs are
  invalid, correct behavior is not guaranteed.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this class.
  Default value: "AutoregressiveStateSpaceModel".



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

<h3 id="coefficients"><code>coefficients</code></h3>



<h3 id="dtype"><code>dtype</code></h3>

The `DType` of `Tensor`s handled by this `Distribution`.

<h3 id="event_shape"><code>event_shape</code></h3>

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

#### Returns:

* <b>`event_shape`</b>: `TensorShape`, possibly unknown.

<h3 id="level_scale"><code>level_scale</code></h3>



<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this `Distribution`.

<h3 id="order"><code>order</code></h3>



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

<h3 id="forward_filter"><code>forward_filter</code></h3>

``` python
forward_filter(
    x,
    mask=None
)
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
* <b>`mask`</b>: optional bool-type `Tensor` with rightmost dimension
  `[num_timesteps]`; `True` values specify that the value of `x`
  at that timestep is masked, i.e., not conditioned on. Additional
  dimensions must match or be broadcastable to `self.batch_shape`; any
  further dimensions must match or be broadcastable to the sample
  shape of `x`.
  Default value: `None`.


#### Returns:

* <b>`log_likelihoods`</b>: Per-timestep log marginal likelihoods `log
  p(x_t | x_{:t-1})` evaluated at the input `x`, as a `Tensor`
  of shape `sample_shape(x) + batch_shape + [num_timesteps].`
* <b>`filtered_means`</b>: Means of the per-timestep filtered marginal
   distributions p(z_t | x_{:t}), as a Tensor of shape
  `sample_shape(x) + batch_shape + [num_timesteps, latent_size]`.
* <b>`filtered_covs`</b>: Covariances of the per-timestep filtered marginal
   distributions p(z_t | x_{:t}), as a Tensor of shape
  `sample_shape(mask) + batch_shape + [num_timesteps, latent_size,
  latent_size]`. Note that the covariances depend only on the model and
  the mask, not on the data, so this may have fewer dimensions than
  `filtered_means`.
* <b>`predicted_means`</b>: Means of the per-timestep predictive
   distributions over latent states, p(z_{t+1} | x_{:t}), as a
   Tensor of shape `sample_shape(x) + batch_shape +
   [num_timesteps, latent_size]`.
* <b>`predicted_covs`</b>: Covariances of the per-timestep predictive
   distributions over latent states, p(z_{t+1} | x_{:t}), as a
   Tensor of shape `sample_shape(mask) + batch_shape +
   [num_timesteps, latent_size, latent_size]`. Note that the covariances
   depend only on the model and the mask, not on the data, so this may
   have fewer dimensions than `predicted_means`.
* <b>`observation_means`</b>: Means of the per-timestep predictive
   distributions over observations, p(x_{t} | x_{:t-1}), as a
   Tensor of shape `sample_shape(x) + batch_shape +
   [num_timesteps, observation_size]`.
* <b>`observation_covs`</b>: Covariances of the per-timestep predictive
   distributions over observations, p(x_{t} | x_{:t-1}), as a
   Tensor of shape `sample_shape(mask) + batch_shape + [num_timesteps,
   observation_size, observation_size]`. Note that the covariances depend
   only on the model and the mask, not on the data, so this may have fewer
   dimensions than `observation_means`.

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

<h3 id="latents_to_observations"><code>latents_to_observations</code></h3>

``` python
latents_to_observations(
    latent_means,
    latent_covs
)
```

Push latent means and covariances forward through the observation model.

#### Args:

* <b>`latent_means`</b>: float `Tensor` of shape `[..., num_timesteps, latent_size]`
* <b>`latent_covs`</b>: float `Tensor` of shape
  `[..., num_timesteps, latent_size, latent_size]`.


#### Returns:

* <b>`observation_means`</b>: float `Tensor` of shape
  `[..., num_timesteps, observation_size]`
* <b>`observation_covs`</b>: float `Tensor` of shape
  `[..., num_timesteps, observation_size, observation_size]`

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


Additional documentation from `LinearGaussianStateSpaceModel`:

##### `kwargs`:

*  `mask`: optional bool-type `Tensor` with rightmost dimension `[num_timesteps]`; `True` values specify that the value of `x` at that timestep is masked, i.e., not conditioned on. Additional dimensions must match or be broadcastable to `self.batch_shape`; any further dimensions must match or be broadcastable to the sample shape of `x`. Default value: `None`.

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

<h3 id="posterior_marginals"><code>posterior_marginals</code></h3>

``` python
posterior_marginals(
    x,
    mask=None
)
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
* <b>`mask`</b>: optional bool-type `Tensor` with rightmost dimension
  `[num_timesteps]`; `True` values specify that the value of `x`
  at that timestep is masked, i.e., not conditioned on. Additional
  dimensions must match or be broadcastable to `self.batch_shape`; any
  further dimensions must match or be broadcastable to the sample
  shape of `x`.
  Default value: `None`.


#### Returns:

* <b>`smoothed_means`</b>: Means of the per-timestep smoothed
   distributions over latent states, p(x_{t} | x_{:T}), as a
   Tensor of shape `sample_shape(x) + batch_shape +
   [num_timesteps, observation_size]`.
* <b>`smoothed_covs`</b>: Covariances of the per-timestep smoothed
   distributions over latent states, p(x_{t} | x_{:T}), as a
   Tensor of shape `sample_shape(mask) + batch_shape + [num_timesteps,
   observation_size, observation_size]`. Note that the covariances depend
   only on the model and the mask, not on the data, so this may have fewer
   dimensions than `filtered_means`.

<h3 id="prob"><code>prob</code></h3>

``` python
prob(
    value,
    name='prob',
    **kwargs
)
```

Probability density/mass function.


Additional documentation from `LinearGaussianStateSpaceModel`:

##### `kwargs`:

*  `mask`: optional bool-type `Tensor` with rightmost dimension `[num_timesteps]`; `True` values specify that the value of `x` at that timestep is masked, i.e., not conditioned on. Additional dimensions must match or be broadcastable to `self.batch_shape`; any further dimensions must match or be broadcastable to the sample shape of `x`. Default value: `None`.

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



