<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.Sum" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="components"/>
<meta itemprop="property" content="components_by_name"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.Sum

## Class `Sum`

Inherits From: [`StructuralTimeSeries`](../../tfp/sts/StructuralTimeSeries.md)

Sum of structural time series components.

This class enables compositional specification of a structural time series
model from basic components. Given a list of component models, it represents
an additive model, i.e., a model of time series that may be decomposed into a
sum of terms corresponding to the component models.

Formally, the additive model represents a random process
`g[t] = f1[t] + f2[t] + ... + fN[t] + eps[t]`, where the `f`'s are the
random processes represented by the components, and
`eps[t] ~ Normal(loc=0, scale=observation_noise_scale)` is an observation
noise term. See the `AdditiveStateSpaceModel` documentation for mathematical
details.

This model inherits the parameters (with priors) of its components, and
adds an `observation_noise_scale` parameter governing the level of noise in
the observed time series.

#### Examples

To construct a model combining a local linear trend with a day-of-week effect:

```
  local_trend = tfp.sts.LocalLinearTrend(
      observed_time_series=observed_time_series,
      name='local_trend')
  day_of_week_effect = tfp.sts.Seasonal(
      num_seasons=7,
      observed_time_series=observed_time_series,
      name='day_of_week_effect')
  additive_model = tfp.sts.Sum(
      components=[local_trend, day_of_week_effect],
      observed_time_series=observed_time_series)

  print([p.name for p in additive_model.parameters])
  # => `[observation_noise_scale,
  #      local_trend_level_scale,
  #      local_trend_slope_scale,
  #      day_of_week_effect_drift_scale`]

  print(local_trend.latent_size,
        seasonal.latent_size,
        additive_model.latent_size)
  # => `2`, `7`, `9`
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    components,
    observation_noise_scale_prior=None,
    observed_time_series=None,
    name=None
)
```

Specify a structural time series model representing a sum of components.

#### Args:

* <b>`components`</b>: Python `list` of one or more StructuralTimeSeries instances.
    These must have unique names.
* <b>`observation_noise_scale_prior`</b>: optional `tfd.Distribution` instance
    specifying a prior on `observation_noise_scale`. If `None`, a heuristic
    default prior is constructed based on the provided
    `observed_time_series`.
    Default value: `None`.
* <b>`observed_time_series`</b>: optional `float` `Tensor` of shape
    `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
    supported when `T > 1`), specifying an observed time series. This is
    used only if `observation_noise_scale_prior` is not provided, to
    construct a default heuristic prior.
    Default value: `None`.
* <b>`name`</b>: Python `str` name of this model component; used as `name_scope`
    for ops created by this class.
    Default value: 'Sum'.


#### Raises:

* <b>`ValueError`</b>: if components do not have unique names.



## Properties

<h3 id="batch_shape"><code>batch_shape</code></h3>

Static batch shape of models represented by this component.

#### Returns:

* <b>`batch_shape`</b>: A `tf.TensorShape` giving the broadcast batch shape of
    all model parameters. This should match the batch shape of
    derived state space models, i.e.,
    `self.make_state_space_model(...).batch_shape`. It may be partially
    defined or unknown.

<h3 id="components"><code>components</code></h3>

List of component `StructuralTimeSeries` models.

<h3 id="components_by_name"><code>components_by_name</code></h3>

OrderedDict mapping component names to components.

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
    model).


#### Returns:

log_joint_fn: A function taking a `Tensor` argument for each model
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



