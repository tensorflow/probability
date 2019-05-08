<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.LinearRegression" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="design_matrix"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.LinearRegression

## Class `LinearRegression`

Formal representation of a linear regression from provided covariates.

Inherits From: [`StructuralTimeSeries`](../../tfp/sts/StructuralTimeSeries.md)



Defined in [`python/sts/regression.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/regression.py).

<!-- Placeholder for "Used in" -->

This model defines a time series given by a linear combination of
covariate time series provided in a design matrix:

```python
observed_time_series = matmul(design_matrix, weights)
```

The design matrix has shape `[num_timesteps, num_features]`. The weights
are treated as an unknown random variable of size `[num_features]` (both
components also support batch shape), and are integrated over using the same
approximate inference tools as other model parameters, i.e., generally HMC or
variational inference.

This component does not itself include observation noise; it defines a
deterministic distribution with mass at the point
`matmul(design_matrix, weights)`. In practice, it should be combined with
observation noise from another component such as <a href="../../tfp/sts/Sum.md"><code>tfp.sts.Sum</code></a>, as
demonstrated below.

#### Examples

Given `series1`, `series2` as `Tensors` each of shape `[num_timesteps]`
representing covariate time series, we create a regression model that
conditions on these covariates:

```python
regression = tfp.sts.LinearRegression(
  design_matrix=tf.stack([series1, series2], axis=-1),
  weights_prior=tfd.Normal(loc=0., scale=1.))
```

Here we've also demonstrated specifying a custom prior, using an informative
`Normal(0., 1.)` prior instead of the default weakly-informative prior.

As a more advanced application, we might use the design matrix to encode
holiday effects. For example, suppose we are modeling data from the month of
December. We can combine day-of-week seasonality with special effects for
Christmas Eve (Dec 24), Christmas (Dec 25), and New Year's Eve (Dec 31),
by constructing a design matrix with indicators for those dates.

```python
holiday_indicators = np.zeros([31, 3])
holiday_indicators[23, 0] = 1  # Christmas Eve
holiday_indicators[24, 1] = 1  # Christmas Day
holiday_indicators[30, 2] = 1  # New Year's Eve

holidays = tfp.sts.LinearRegression(design_matrix=holiday_indicators,
                                    name='holidays')
day_of_week = tfp.sts.Seasonal(num_seasons=7,
                               observed_time_series=observed_time_series,
                               name='day_of_week')
model = tfp.sts.Sum(components=[holidays, seasonal],
                    observed_time_series=observed_time_series)
```

Note that the `Sum` component in the above model also incorporates observation
noise, with prior scale heuristically inferred from `observed_time_series`.

In these examples, we've used a single design matrix, but batching is
also supported. If the design matrix has batch shape, the default behavior
constructs weights with matching batch shape, which will fit a separate
regression for each design matrix. This can be overridden by passing an
explicit weights prior with appropriate batch shape. For example, if each
design matrix in a batch contains features with the same semantics
(e.g., if they represent per-group or per-observation covariates), we might
choose to share statistical strength by fitting a single weight vector that
broadcasts across all design matrices:

```python
design_matrix = get_batch_of_inputs()
design_matrix.shape  # => concat([batch_shape, [num_timesteps, num_features]])

# Construct a prior with batch shape `[]` and event shape `[num_features]`,
# so that it describes a single vector of weights.
weights_prior = tfd.Independent(
    tfd.StudentT(df=5,
                 loc=tf.zeros([num_features]),
                 scale=tf.ones([num_features])),
    reinterpreted_batch_ndims=1)
linear_regression = LinearRegression(design_matrix=design_matrix,
                                     weights_prior=weights_prior)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    design_matrix,
    weights_prior=None,
    name=None
)
```

Specify a linear regression model.

Note: the statistical behavior of the regression is determined by
the broadcasting behavior of the `weights` `Tensor`:

* `weights_prior.batch_shape == []`: shares a single set of weights across
  all design matrices and observed time series. This may make sense if
  the features in each design matrix have the same semantics (e.g.,
  grouping observations by country, with per-country design matrices
  capturing the same set of national economic indicators per country).
* `weights_prior.batch_shape == `design_matrix.batch_shape`: fits separate
  weights for each design matrix. If there are multiple observed time series
  for each design matrix, this shares statistical strength over those
  observations.
* `weights_prior.batch_shape == `observed_time_series.batch_shape`: fits a
  separate regression for each individual time series.

When modeling batches of time series, you should think carefully about
which behavior makes sense, and specify `weights_prior` accordingly:
the defaults may not do what you want!

#### Args:

* <b>`design_matrix`</b>: float `Tensor` of shape `concat([batch_shape,
  [num_timesteps, num_features]])`. This may also optionally be
  an instance of `tf.linalg.LinearOperator`.
* <b>`weights_prior`</b>: `tfd.Distribution` representing a prior over the regression
  weights. Must have event shape `[num_features]` and batch shape
  broadcastable to the design matrix's `batch_shape`. Alternately,
  `event_shape` may be scalar (`[]`), in which case the prior is
  internally broadcast as `TransformedDistribution(weights_prior,
  tfb.Identity(), event_shape=[num_features],
  batch_shape=design_matrix.batch_shape)`. If `None`,
  defaults to `StudentT(df=5, loc=0., scale=10.)`, a weakly-informative
  prior loosely inspired by the [Stan prior choice recommendations](
  https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations).
  Default value: `None`.
* <b>`name`</b>: the name of this model component.
  Default value: 'LinearRegression'.



## Properties

<h3 id="batch_shape"><code>batch_shape</code></h3>

Static batch shape of models represented by this component.

#### Returns:

* <b>`batch_shape`</b>: A `tf.TensorShape` giving the broadcast batch shape of
  all model parameters. This should match the batch shape of
  derived state space models, i.e.,
  `self.make_state_space_model(...).batch_shape`. It may be partially
  defined or unknown.

<h3 id="design_matrix"><code>design_matrix</code></h3>

LinearOperator representing the design matrix.

<h3 id="latent_size"><code>latent_size</code></h3>

Python `int` dimensionality of the latent space in this model.

<h3 id="name"><code>name</code></h3>

Name of this model component.

<h3 id="parameters"><code>parameters</code></h3>

List of Parameter(name, prior, bijector) namedtuples for this model.



## Methods

<h3 id="batch_shape_tensor"><code>batch_shape_tensor</code></h3>

``` python
batch_shape_tensor()
```

Runtime batch shape of models represented by this component.

#### Returns:

* <b>`batch_shape`</b>: `int` `Tensor` giving the broadcast batch shape of
  all model parameters. This should match the batch shape of
  derived state space models, i.e.,
  `self.make_state_space_model(...).batch_shape_tensor()`.

<h3 id="joint_log_prob"><code>joint_log_prob</code></h3>

``` python
joint_log_prob(observed_time_series)
```

Build the joint density `log p(params) + log p(y|params)` as a callable.

#### Args:

* <b>`observed_time_series`</b>: Observed `Tensor` trajectories of shape
  `sample_shape + batch_shape + [num_timesteps, 1]` (the trailing
  `1` dimension is optional if `num_timesteps > 1`), where
  `batch_shape` should match `self.batch_shape` (the broadcast batch
  shape of all priors on parameters for this structural time series
  model). May optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>,
  which includes a mask `Tensor` to specify timesteps with missing
  observations.


#### Returns:

* <b>`log_joint_fn`</b>: A function taking a `Tensor` argument for each model
  parameter, in canonical order, and returning a `Tensor` log probability
  of shape `batch_shape`. Note that, *unlike* `tfp.Distributions`
  `log_prob` methods, the `log_joint` sums over the `sample_shape` from y,
  so that `sample_shape` does not appear in the output log_prob. This
  corresponds to viewing multiple samples in `y` as iid observations from a
  single model, which is typically the desired behavior for parameter
  inference.

<h3 id="make_state_space_model"><code>make_state_space_model</code></h3>

``` python
make_state_space_model(
    num_timesteps,
    param_vals=None,
    initial_state_prior=None,
    initial_step=0
)
```

Instantiate this model as a Distribution over specified `num_timesteps`.

#### Args:

* <b>`num_timesteps`</b>: Python `int` number of timesteps to model.
* <b>`param_vals`</b>: a list of `Tensor` parameter values in order corresponding to
  `self.parameters`, or a dict mapping from parameter names to values.
* <b>`initial_state_prior`</b>: an optional `Distribution` instance overriding the
  default prior on the model's initial state. This is used in forecasting
  ("today's prior is yesterday's posterior").
* <b>`initial_step`</b>: optional `int` specifying the initial timestep to model.
  This is relevant when the model contains time-varying components,
  e.g., holidays or seasonality.


#### Returns:

* <b>`dist`</b>: a `LinearGaussianStateSpaceModel` Distribution object.

<h3 id="prior_sample"><code>prior_sample</code></h3>

``` python
prior_sample(
    num_timesteps,
    initial_step=0,
    params_sample_shape=(),
    trajectories_sample_shape=(),
    seed=None
)
```

Sample from the joint prior over model parameters and trajectories.

#### Args:

* <b>`num_timesteps`</b>: Scalar `int` `Tensor` number of timesteps to model.
* <b>`initial_step`</b>: Optional scalar `int` `Tensor` specifying the starting
  timestep.
    Default value: 0.
* <b>`params_sample_shape`</b>: Number of possible worlds to sample iid from the
  parameter prior, or more generally, `Tensor` `int` shape to fill with
  iid samples.
    Default value: [] (i.e., draw a single sample and don't expand the
    shape).
* <b>`trajectories_sample_shape`</b>: For each sampled set of parameters, number
  of trajectories to sample, or more generally, `Tensor` `int` shape to
  fill with iid samples.
  Default value: [] (i.e., draw a single sample and don't expand the
    shape).
* <b>`seed`</b>: Python `int` random seed.


#### Returns:

* <b>`trajectories`</b>: `float` `Tensor` of shape
  `trajectories_sample_shape + params_sample_shape + [num_timesteps, 1]`
  containing all sampled trajectories.
* <b>`param_samples`</b>: list of sampled parameter value `Tensor`s, in order
  corresponding to `self.parameters`, each of shape
  `params_sample_shape + prior.batch_shape + prior.event_shape`.



