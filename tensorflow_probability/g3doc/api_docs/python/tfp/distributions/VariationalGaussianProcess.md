<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.distributions.VariationalGaussianProcess" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_nan_stats"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="bijector"/>
<meta itemprop="property" content="distribution"/>
<meta itemprop="property" content="dtype"/>
<meta itemprop="property" content="event_shape"/>
<meta itemprop="property" content="index_points"/>
<meta itemprop="property" content="inducing_index_points"/>
<meta itemprop="property" content="jitter"/>
<meta itemprop="property" content="kernel"/>
<meta itemprop="property" content="loc"/>
<meta itemprop="property" content="mean_fn"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="name_scope"/>
<meta itemprop="property" content="observation_noise_variance"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="predictive_noise_variance"/>
<meta itemprop="property" content="reparameterization_type"/>
<meta itemprop="property" content="scale"/>
<meta itemprop="property" content="submodules"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="validate_args"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="variational_inducing_observations_loc"/>
<meta itemprop="property" content="variational_inducing_observations_scale"/>
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
<meta itemprop="property" content="optimal_variational_posterior"/>
<meta itemprop="property" content="param_shapes"/>
<meta itemprop="property" content="param_static_shapes"/>
<meta itemprop="property" content="prob"/>
<meta itemprop="property" content="quantile"/>
<meta itemprop="property" content="sample"/>
<meta itemprop="property" content="stddev"/>
<meta itemprop="property" content="survival_function"/>
<meta itemprop="property" content="variance"/>
<meta itemprop="property" content="variational_loss"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# tfp.distributions.VariationalGaussianProcess

## Class `VariationalGaussianProcess`

Posterior predictive of a variational Gaussian process.

Inherits From: [`MultivariateNormalLinearOperator`](../../tfp/distributions/MultivariateNormalLinearOperator.md)



Defined in [`python/distributions/variational_gaussian_process.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/distributions/variational_gaussian_process.py).

<!-- Placeholder for "Used in" -->

This distribution implements the variational Gaussian process (VGP), as
described in [Titsias, 2009][1] and [Hensman, 2013][2]. The VGP is an
inducing point-based approximation of an exact GP posterior
(see Mathematical Details, below). Ultimately, this Distribution class
represents a marginal distrbution over function values at a
collection of `index_points`. It is parameterized by

  - a kernel function,
  - a mean function,
  - the (scalar) observation noise variance of the normal likelihood,
  - a set of index points,
  - a set of inducing index points, and
  - the parameters of the (full-rank, Gaussian) variational posterior
    distribution over function values at the inducing points, conditional on
    some observations.

A VGP is "trained" by selecting any kernel parameters, the locations of the
inducing index points, and the variational parameters. [Titsias, 2009][1] and
[Hensman, 2013][2] describe a variational lower bound on the marginal log
likelihood of observed data, which this class offers through the
`variational_loss` method (this is the negative lower bound, for convenience
when plugging into a TF Optimizer's `minimize` function).
Training may be done in minibatches.

[Titsias, 2009][1] describes a closed form for the optimal variational
parameters, in the case of sufficiently small observational data (ie,
small enough to fit in memory but big enough to warrant approximating the GP
posterior). A method to compute these optimal parameters in terms of the full
observational data set is provided as a staticmethod,
`optimal_variational_posterior`. It returns a
`MultivariateNormalLinearOperator` instance with optimal location and
scale parameters.

#### Mathematical Details

##### Notation

We will in general be concerned about three collections of index points, and
it'll be good to give them names:

  * `x[1], ..., x[N]`: observation index points -- locations of our observed
    data.
  * `z[1], ..., z[M]`: inducing index points  -- locations of the
    "summarizing" inducing points
  * `t[1], ..., t[P]`: predictive index points -- locations where we are
    making posterior predictions based on observations and the variational
    parameters.

To lighten notation, we'll use `X, Z, T` to denote the above collections.
Similarly, we'll denote by `f(X)` the collection of function values at each of
the `x[i]`, and by `Y`, the collection of (noisy) observed data at each `x[i].
We'll denote kernel matrices generated from pairs of index points as `K_tt`,
`K_xt`, `K_tz`, etc, e.g.,

```none
         | k(t[1], z[1])    k(t[1], z[2])  ...  k(t[1], z[M]) |
  K_tz = | k(t[2], z[1])    k(t[2], z[2])  ...  k(t[2], z[M]) |
         |      ...              ...                 ...      |
         | k(t[P], z[1])    k(t[P], z[2])  ...  k(t[P], z[M]) |
```

##### Preliminaries

A Gaussian process is an indexed collection of random variables, any finite
collection of which are jointly Gaussian. Typically, the index set is some
finite-dimensional, real vector space, and indeed we make this assumption in
what follows. The GP may then be thought of as a distribution over functions
on the index set. Samples from the GP are functions *on the whole index set*;
these can't be represented in finite compute memory, so one typically works
with the marginals at a finite collection of index points. The properties of
the GP are entirely determined by its mean function `m` and covariance
function `k`. The generative process, assuming a mean-zero normal likelihood
with stddev `sigma`, is

```none
  f ~ GP(m, k)

  Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
```

In finite terms (ie, marginalizing out all but a finite number of f(X)'sigma),
we can write

```none
  f(X) ~ MVN(loc=m(X), cov=K_xx)

  Y | f(X) ~ Normal(f(X), sigma),   i = 1, ... , N
```

Posterior inference is possible in analytical closed form but becomes
intractible as data sizes get large. See [Rasmussen, 2006][3] for details.

##### The VGP

The VGP is an inducing point-based approximation of an exact GP posterior,
where two approximating assumptions have been made:

  1. function values at non-inducing points are mutually independent
     conditioned on function values at the inducing points,
  2. the (expensive) posterior over function values at inducing points
     conditional on obseravtions is replaced with an arbitrary (learnable)
     full-rank Gaussian distribution,

     ```none
       q(f(Z)) = MVN(loc=m, scale=S),
     ```

     where `m` and `S` are parameters to be chosen by optimizing an evidence
     lower bound (ELBO).

The posterior predictive distribution becomes

```none
  q(f(T)) = integral df(Z) p(f(T) | f(Z)) q(f(Z))
          = MVN(loc = A @ m, scale = B^(1/2))
```

where

```none
  A = K_tz @ K_zz^-1
  B = K_tt - A @ (K_zz - S S^T) A^T
```

***The approximate posterior predictive distribution `q(f(T))` is what the
`VariationalGaussianProcess` class represents.***

Model selection in this framework entails choosing the kernel parameters,
inducing point locations, and variational parameters. We do this by optimizing
a variational lower bound on the marginal log likelihood of observed data. The
lower bound takes the following form (see [Titsias, 2009][1] and
[Hensman, 2013][2] for details on the derivation):

```none
  L(Z, m, S, Y) = (
      MVN(loc=(K_zx @ K_zz^-1) @ m, scale_diag=sigma).log_prob(Y) -
      (Tr(K_xx - K_zx @ K_zz^-1 @ K_xz) +
       Tr(S @ S^T @ K_zz^1 @ K_zx @ K_xz @ K_zz^-1)) / (2 * sigma^2) -
      KL(q(f(Z)) || p(f(Z))))
```

where in the final KL term, `p(f(Z))` is the GP prior on inducing point
function values. This variational lower bound can be computed on minibatches
of the full data set `(X, Y)`. A method to compute the *negative* variational
lower bound is implemented as `VariationalGaussianProcess.variational_loss`.

##### Optimal variational parameters

As described in [Titsias, 2009][1], a closed form optimum for the variational
location and scale parameters, `m` and `S`, can be computed when the
observational data are not prohibitively voluminous. The
`optimal_variational_posterior` function to computes the optimal variational
posterior distribution over inducing point function values in terms of the GP
parameters (mean and kernel functions), inducing point locations, observation
index points, and observations. Note that the inducing index point locations
must still be optimized even when these parameters are known functions of the
inducing index points. The optimal parameters are computed as follows:

```none
  C = sigma^-2 (K_zz + K_zx @ K_xz)^-1

  optimal Gaussian covariance: K_zz @ C @ K_zz
  optimal Gaussian location: sigma^-2 K_zz @ C @ K_zx @ Y
```

#### Usage Examples

Here's an example of defining and training a VariationalGaussianProcess on
some toy generated data.

```python
# We'll use double precision throughout for better numerics.
dtype = np.float64

# Generate noisy data from a known function.
f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
true_observation_noise_variance_ = dtype(1e-1) ** 2

num_training_points_ = 100
x_train_ = np.stack(
    [np.random.uniform(-6., 0., [num_training_points_/ 2 , 1]),
     np.random.uniform(1., 10., [num_training_points_/ 2 , 1])],
    axis=0).astype(dtype)
y_train_ = (f(x_train_) +
            np.random.normal(
                0., np.sqrt(true_observation_noise_variance_),
                [num_training_points_]).astype(dtype))

# Create kernel with trainable parameters, and trainable observation noise
# variance variable. Each of these is constrained to be positive.
amplitude = (tf.nn.softplus(tf.Variable(-1., dtype=dtype, name='amplitude')))
length_scale = (1e-5 +
                tf.nn.softplus(
                    tf.Variable(-3., dtype=dtype, name='length_scale')))
kernel = tfk.ExponentiatedQuadratic(
    amplitude=amplitude,
    length_scale=length_scale)

observation_noise_variance = tf.nn.softplus(
    tf.Variable(0, dtype=dtype, name='observation_noise_variance'))

# Create trainable inducing point locations and variational parameters.
num_inducing_points_ = 20

inducing_index_points = tf.Variable(
    initial_inducing_points_, dtype=dtype,
    name='inducing_index_points')
variational_inducing_observations_loc = tf.Variable(
    np.zeros([num_inducing_points_], dtype=dtype),
    name='variational_inducing_observations_loc')
variational_inducing_observations_scale = tf.Variable(
    np.eye(num_inducing_points_, dtype=dtype),
    name='variational_inducing_observations_scale')

# These are the index point locations over which we'll construct the
# (approximate) posterior predictive distribution.
num_predictive_index_points_ = 500
index_points_ = np.linspace(-13, 13,
                            num_predictive_index_points_,
                            dtype=dtype)[..., np.newaxis]


# Construct our variational GP Distribution instance.
vgp = tfd.VariationalGaussianProcess(
    kernel,
    index_points=index_points_,
    inducing_index_points=inducing_index_points,
    variational_inducing_observations_loc=variational_inducing_observations_loc,
    variational_inducing_observations_scale=variational_inducing_observations_scale,
    observation_noise_variance=observation_noise_variance)

# For training, we use some simplistic numpy-based minibatching.
batch_size = 64
x_train_batch = tf.placeholder(dtype, [batch_size, 1], name='x_train_batch')
y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

# Create the loss function we want to optimize.
loss = vgp.variational_loss(
    observations=y_train_batch,
    observation_index_points=x_train_batch,
    kl_weight=float(batch_size) / float(num_training_points_))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

num_iters = 10000
num_logs = 10
with tf.Session() as sess:
  for i in range(num_iters):
    batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
    x_train_batch_ = x_train_[batch_idxs, ...]
    y_train_batch_ = y_train_[batch_idxs]

    [_, loss_] = sess.run([train_op, loss],
                          feed_dict={x_train_batch: x_train_batch_,
                                     y_train_batch: y_train_batch_})
    if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
      print(i, loss_)

# Generate a plot with
#   - the posterior predictive mean
#   - training data
#   - inducing index points (plotted vertically at the mean of the variational
#     posterior over inducing point function values)
#   - 50 posterior predictive samples

num_samples = 50
[
    samples_,
    mean_,
    inducing_index_points_,
    variational_loc_,
] = sess.run([
    vgp.sample(num_samples),
    vgp.mean(),
    inducing_index_points,
    variational_inducing_observations_loc
])
plt.figure(figsize=(15, 5))
plt.scatter(inducing_index_points_[..., 0], variational_loc_
            marker='x', s=50, color='k', zorder=10)
plt.scatter(x_train_[..., 0], y_train_, color='#00ff00', zorder=9)
plt.plot(np.tile(index_points_[..., 0], num_samples),
         samples_.T, color='r', alpha=.1)
plt.plot(index_points_, mean_, color='k')
plt.plot(index_points_, f(index_points_), color='b')
```

# Here we use the same data setup, but compute the optimal variational
# parameters instead of training them.
```python
# We'll use double precision throughout for better numerics.
dtype = np.float64

# Generate noisy data from a known function.
f = lambda x: np.exp(-x[..., 0]**2 / 20.) * np.sin(1. * x[..., 0])
true_observation_noise_variance_ = dtype(1e-1) ** 2

num_training_points_ = 1000
x_train_ = np.random.uniform(-10., 10., [num_training_points_, 1])
y_train_ = (f(x_train_) +
            np.random.normal(
                0., np.sqrt(true_observation_noise_variance_),
                [num_training_points_]))

# Create kernel with trainable parameters, and trainable observation noise
# variance variable. Each of these is constrained to be positive.
amplitude = (tf.nn.softplus(
  tf.Variable(.54, dtype=dtype, name='amplitude', use_resource=True)))
length_scale = (
  1e-5 +
  tf.nn.softplus(
    tf.Variable(.54, dtype=dtype, name='length_scale', use_resource=True)))
kernel = tfk.ExponentiatedQuadratic(
    amplitude=amplitude,
    length_scale=length_scale)

observation_noise_variance = tf.nn.softplus(
    tf.Variable(
      .54, dtype=dtype, name='observation_noise_variance', use_resource=True))

# Create trainable inducing point locations and variational parameters.
num_inducing_points_ = 10

inducing_index_points = tf.Variable(
    np.linspace(-10., 10., num_inducing_points_)[..., np.newaxis],
    dtype=dtype, name='inducing_index_points', use_resource=True)

variational_loc, variational_scale = (
    tfd.VariationalGaussianProcess.optimal_variational_posterior(
        kernel=kernel,
        inducing_index_points=inducing_index_points,
        observation_index_points=x_train_,
        observations=y_train_,
        observation_noise_variance=observation_noise_variance))

# These are the index point locations over which we'll construct the
# (approximate) posterior predictive distribution.
num_predictive_index_points_ = 500
index_points_ = np.linspace(-13, 13,
                            num_predictive_index_points_,
                            dtype=dtype)[..., np.newaxis]

# Construct our variational GP Distribution instance.
vgp = tfd.VariationalGaussianProcess(
    kernel,
    index_points=index_points_,
    inducing_index_points=inducing_index_points,
    variational_inducing_observations_loc=variational_loc,
    variational_inducing_observations_scale=variational_scale,
    observation_noise_variance=observation_noise_variance)

# For training, we use some simplistic numpy-based minibatching.
batch_size = 64
x_train_batch = tf.placeholder(dtype, [batch_size, 1], name='x_train_batch')
y_train_batch = tf.placeholder(dtype, [batch_size], name='y_train_batch')

# Create the loss function we want to optimize.
loss = vgp.variational_loss(
    observations=y_train_batch,
    observation_index_points=x_train_batch,
    kl_weight=float(batch_size) / float(num_training_points_))

optimizer = tf.train.AdamOptimizer(learning_rate=.01)
train_op = optimizer.minimize(loss)

num_iters = 300
num_logs = 10
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(num_iters):
    batch_idxs = np.random.randint(num_training_points_, size=[batch_size])
    x_train_batch_ = x_train_[batch_idxs, ...]
    y_train_batch_ = y_train_[batch_idxs]

    [_, loss_] = sess.run([train_op, loss],
                          feed_dict={x_train_batch: x_train_batch_,
                                     y_train_batch: y_train_batch_})
    if i % (num_iters / num_logs) == 0 or i + 1 == num_iters:
      print(i, loss_)

  # Generate a plot with
  #   - the posterior predictive mean
  #   - training data
  #   - inducing index points (plotted vertically at the mean of the
  #     variational posterior over inducing point function values)
  #   - 50 posterior predictive samples

  num_samples = 50
  [
      samples_,
      mean_,
      inducing_index_points_,
      variational_loc_,
  ] = sess.run([
      vgp.sample(num_samples),
      vgp.mean(),
      inducing_index_points,
      variational_loc
  ])
  plt.figure(figsize=(15, 5))
  plt.scatter(inducing_index_points_[..., 0], variational_loc_,
              marker='x', s=50, color='k', zorder=10)
  plt.scatter(x_train_[..., 0], y_train_, color='#00ff00', alpha=.1, zorder=9)
  plt.plot(np.tile(index_points_, num_samples),
           samples_.T, color='r', alpha=.1)
  plt.plot(index_points_, mean_, color='k')
  plt.plot(index_points_, f(index_points_), color='b')

```

#### References

[1]: Titsias, M. "Variational Model Selection for Sparse Gaussian Process
     Regression", 2009.
     http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf
[2]: Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013
     https://arxiv.org/abs/1309.6835
[3]: Carl Rasmussen, Chris Williams. Gaussian Processes For Machine Learning,
     2006. http://www.gaussianprocess.org/gpml/

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    kernel,
    index_points,
    inducing_index_points,
    variational_inducing_observations_loc,
    variational_inducing_observations_scale,
    mean_fn=None,
    observation_noise_variance=0.0,
    predictive_noise_variance=0.0,
    jitter=1e-06,
    validate_args=False,
    allow_nan_stats=False,
    name='VariataionalGaussianProcess'
)
```

Instantiate a VariationalGaussianProcess Distribution.


#### Args:


* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
  GP's covariance function.
* <b>`index_points`</b>: `float` `Tensor` representing finite (batch of) vector(s) of
  points in the index set over which the VGP is defined. Shape has the
  form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
  dimensions and must equal `kernel.feature_ndims` and `e1` is the number
  (size) of index points in each batch (we denote it `e1` to distinguish
  it from the numer of inducing index points, denoted `e2` below).
  Ultimately the VariationalGaussianProcess distribution corresponds to an
  `e1`-dimensional multivariate normal. The batch shape must be
  broadcastable with `kernel.batch_shape`, the batch shape of
  `inducing_index_points`, and any batch dims yielded by `mean_fn`.
* <b>`inducing_index_points`</b>: `float` `Tensor` of locations of inducing points in
  the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
  like `index_points`. The batch shape components needn't be identical to
  those of `index_points`, but must be broadcast compatible with them.
* <b>`variational_inducing_observations_loc`</b>: `float` `Tensor`; the mean of the
  (full-rank Gaussian) variational posterior over function values at the
  inducing points, conditional on observed data. Shape has the form `[b1,
  ..., bB, e2]`, where `b1, ..., bB` is broadcast compatible with other
  parameters' batch shapes, and `e2` is the number of inducing points.
* <b>`variational_inducing_observations_scale`</b>: `float` `Tensor`; the scale
  matrix of the (full-rank Gaussian) variational posterior over function
  values at the inducing points, conditional on observed data. Shape has
  the form `[b1, ..., bB, e2, e2]`, where `b1, ..., bB` is broadcast
  compatible with other parameters and `e2` is the number of inducing
  points.
* <b>`mean_fn`</b>: Python `callable` that acts on index points to produce a (batch
  of) vector(s) of mean values at those index points. Takes a `Tensor` of
  shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
  (broadcastable with) `[b1, ..., bB]`. Default value: `None` implies
  constant zero function.
* <b>`observation_noise_variance`</b>: `float` `Tensor` representing the variance
  of the noise in the Normal likelihood distribution of the model. May be
  batched, in which case the batch shape must be broadcastable with the
  shapes of all other batched parameters (`kernel.batch_shape`,
  `index_points`, etc.).
  Default value: `0.`
* <b>`predictive_noise_variance`</b>: `float` `Tensor` representing additional
  variance in the posterior predictive model. If `None`, we simply re-use
  `observation_noise_variance` for the posterior predictive noise. If set
  explicitly, however, we use the given value. This allows us, for
  example, to omit predictive noise variance (by setting this to zero) to
  obtain noiseless posterior predictions of function values, conditioned
  on noisy observations.
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
  Default value: "VariationalGaussianProcess".


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




<h3 id="inducing_index_points"><code>inducing_index_points</code></h3>




<h3 id="jitter"><code>jitter</code></h3>




<h3 id="kernel"><code>kernel</code></h3>




<h3 id="loc"><code>loc</code></h3>

The `loc` `Tensor` in `Y = scale @ X + loc`.


<h3 id="mean_fn"><code>mean_fn</code></h3>




<h3 id="name"><code>name</code></h3>

Name prepended to all ops created by this `Distribution`.


<h3 id="name_scope"><code>name_scope</code></h3>

Returns a `tf.name_scope` instance for this class.


<h3 id="observation_noise_variance"><code>observation_noise_variance</code></h3>




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


<h3 id="variational_inducing_observations_loc"><code>variational_inducing_observations_loc</code></h3>




<h3 id="variational_inducing_observations_scale"><code>variational_inducing_observations_scale</code></h3>






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

`other` types with built-in registrations: `GaussianProcess`, `GaussianProcessRegressionModel`, `MultivariateNormalDiag`, `MultivariateNormalDiagPlusLowRank`, `MultivariateNormalDiagWithSoftplusScale`, `MultivariateNormalFullCovariance`, `MultivariateNormalLinearOperator`, `MultivariateNormalTriL`, `VariationalGaussianProcess`

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

`other` types with built-in registrations: `GaussianProcess`, `GaussianProcessRegressionModel`, `MultivariateNormalDiag`, `MultivariateNormalDiagPlusLowRank`, `MultivariateNormalDiagWithSoftplusScale`, `MultivariateNormalFullCovariance`, `MultivariateNormalLinearOperator`, `MultivariateNormalTriL`, `VariationalGaussianProcess`

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


<h3 id="optimal_variational_posterior"><code>optimal_variational_posterior</code></h3>

``` python
@staticmethod
optimal_variational_posterior(
    kernel,
    inducing_index_points,
    observation_index_points,
    observations,
    observation_noise_variance,
    mean_fn=None,
    jitter=1e-06,
    name=None
)
```

Model selection for optimal variational hyperparameters.

Given the full training set (parameterized by `observations` and
`observation_index_points`), compute the optimal variational
location and scale for the VGP. This is based of the method suggested
in [Titsias, 2009][1].

#### Args:


* <b>`kernel`</b>: `PositiveSemidefiniteKernel`-like instance representing the
  GP's covariance function.
* <b>`inducing_index_points`</b>: `float` `Tensor` of locations of inducing points in
  the index set. Shape has the form `[b1, ..., bB, e2, f1, ..., fF]`, just
  like `observation_index_points`. The batch shape components needn't be
  identical to those of `observation_index_points`, but must be broadcast
  compatible with them.
* <b>`observation_index_points`</b>: `float` `Tensor` representing finite (batch of)
  vector(s) of points where observations are defined. Shape has the
  form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
  dimensions and must equal `kernel.feature_ndims` and `e1` is the number
  (size) of index points in each batch (we denote it `e1` to distinguish
  it from the numer of inducing index points, denoted `e2` below).
* <b>`observations`</b>: `float` `Tensor` representing collection, or batch of
  collections, of observations corresponding to
  `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
  must be brodcastable with the batch and example shapes of
  `observation_index_points`. The batch shape `[b1, ..., bB]` must be
  broadcastable with the shapes of all other batched parameters
  (`kernel.batch_shape`, `observation_index_points`, etc.).
* <b>`observation_noise_variance`</b>: `float` `Tensor` representing the variance
  of the noise in the Normal likelihood distribution of the model. May be
  batched, in which case the batch shape must be broadcastable with the
  shapes of all other batched parameters (`kernel.batch_shape`,
  `index_points`, etc.).
  Default value: `0.`
* <b>`mean_fn`</b>: Python `callable` that acts on index points to produce a (batch
  of) vector(s) of mean values at those index points. Takes a `Tensor` of
  shape `[b1, ..., bB, f1, ..., fF]` and returns a `Tensor` whose shape is
  (broadcastable with) `[b1, ..., bB]`. Default value: `None` implies
  constant zero function.
* <b>`jitter`</b>: `float` scalar `Tensor` added to the diagonal of the covariance
  matrix to ensure positive definiteness of the covariance matrix.
  Default value: `1e-6`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: "optimal_variational_posterior".

#### Returns:

loc, scale: Tuple representing the variational location and scale.


#### Raises:


* <b>`ValueError`</b>: if `mean_fn` is not `None` and is not callable.

#### References

[1]: Titsias, M. "Variational Model Selection for Sparse Gaussian Process
     Regression", 2009.
     http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf

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

<h3 id="variational_loss"><code>variational_loss</code></h3>

``` python
variational_loss(
    observations,
    observation_index_points=None,
    kl_weight=1.0,
    name='variational_loss'
)
```

Variational loss for the VGP.

Given `observations` and `observation_index_points`, compute the
negative variational lower bound as specified in [Hensman, 2013][1].

#### Args:


* <b>`observations`</b>: `float` `Tensor` representing collection, or batch of
  collections, of observations corresponding to
  `observation_index_points`. Shape has the form `[b1, ..., bB, e]`, which
  must be brodcastable with the batch and example shapes of
  `observation_index_points`. The batch shape `[b1, ..., bB]` must be
  broadcastable with the shapes of all other batched parameters
  (`kernel.batch_shape`, `observation_index_points`, etc.).
* <b>`observation_index_points`</b>: `float` `Tensor` representing finite (batch of)
  vector(s) of points where observations are defined. Shape has the
  form `[b1, ..., bB, e1, f1, ..., fF]` where `F` is the number of feature
  dimensions and must equal `kernel.feature_ndims` and `e1` is the number
  (size) of index points in each batch (we denote it `e1` to distinguish
  it from the numer of inducing index points, denoted `e2` below). If
  set to `None` uses `index_points` as the origin for observations.
  Default value: None.
* <b>`kl_weight`</b>: Amount by which to scale the KL divergence loss between prior
  and posterior.
  Default value: 1.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
  Default value: "GaussianProcess".

#### Returns:


* <b>`loss`</b>: Scalar tensor representing the negative variational lower bound.
  Can be directly used in a `tf.Optimizer`.

#### Raises:


* <b>`ValueError`</b>: if `mean_fn` is not `None` and is not callable.

#### References

[1]: Hensman, J., Lawrence, N. "Gaussian Processes for Big Data", 2013
     https://arxiv.org/abs/1309.6835

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




