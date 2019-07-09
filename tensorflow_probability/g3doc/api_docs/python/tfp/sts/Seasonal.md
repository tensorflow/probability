<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.Seasonal" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="constrain_mean_effect_to_zero"/>
<meta itemprop="property" content="initial_state_prior"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_seasons"/>
<meta itemprop="property" content="num_steps_per_season"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.Seasonal

## Class `Seasonal`

Formal representation of a seasonal effect model.

Inherits From: [`StructuralTimeSeries`](../../tfp/sts/StructuralTimeSeries.md)



Defined in [`python/sts/seasonal.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/seasonal.py).

<!-- Placeholder for "Used in" -->

A seasonal effect model posits a fixed set of recurring, discrete 'seasons',
each of which is active for a fixed number of timesteps and, while active,
contributes a different effect to the time series. These are generally not
meteorological seasons, but represent regular recurring patterns such as
hour-of-day or day-of-week effects. Each season lasts for a fixed number of
timesteps. The effect of each season drifts from one occurrence to the next
following a Gaussian random walk:

```python
effects[season, occurrence[i]] = (
  effects[season, occurrence[i-1]] + Normal(loc=0., scale=drift_scale))
```

The `drift_scale` parameter governs the standard deviation of the random walk;
for example, in a day-of-week model it governs the change in effect from this
Monday to next Monday.

#### Examples

A seasonal effect model representing day-of-week seasonality on hourly data:

```python
day_of_week = tfp.sts.Seasonal(num_seasons=7,
                               num_steps_per_season=24,
                               observed_time_series=y,
                               name='day_of_week')
```

A seasonal effect model representing month-of-year seasonality on daily data,
with explicit priors:

```python
month_of_year = tfp.sts.Seasonal(
  num_seasons=12,
  num_steps_per_season=[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
  drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
  initial_effect_prior=tfd.Normal(loc=0., scale=5.),
  name='month_of_year')
```

Note that this version works over time periods not involving a leap year. A
general implementation of month-of-year seasonality would require additional
logic:

```python
num_days_per_month = np.array(
  [[31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
   [31, 29, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],  # year with leap day
   [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31],
   [31, 28, 31, 30, 30, 31, 31, 31, 30, 31, 30, 31]])

month_of_year = tfp.sts.Seasonal(
  num_seasons=12,
  num_steps_per_season=num_days_per_month,
  drift_scale_prior=tfd.LogNormal(loc=-1., scale=0.1),
  initial_effect_prior=tfd.Normal(loc=0., scale=5.),
  name='month_of_year')
```

A model representing both day-of-week and hour-of-day seasonality, on hourly
data:

```
day_of_week = tfp.sts.Seasonal(num_seasons=7,
                               num_steps_per_season=24,
                               observed_time_series=y,
                               name='day_of_week')
hour_of_day = tfp.sts.Seasonal(num_seasons=24,
                               num_steps_per_season=1,
                               observed_time_series=y,
                               name='hour_of_day')
model = tfp.sts.Sum(components=[day_of_week, hour_of_day],
                    observed_time_series=y)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    num_seasons,
    num_steps_per_season=1,
    drift_scale_prior=None,
    initial_effect_prior=None,
    constrain_mean_effect_to_zero=True,
    observed_time_series=None,
    name=None
)
```

Specify a seasonal effects model.


#### Args:


* <b>`num_seasons`</b>: Scalar Python `int` number of seasons.
* <b>`num_steps_per_season`</b>: Python `int` number of steps in each
  season. This may be either a scalar (shape `[]`), in which case all
  seasons have the same length, or a NumPy array of shape `[num_seasons]`,
  in which seasons have different length, but remain constant around
  different cycles, or a NumPy array of shape `[num_cycles, num_seasons]`,
  in which num_steps_per_season for each season also varies in different
  cycle (e.g., a 4 years cycle with leap day).
  Default value: 1.
* <b>`drift_scale_prior`</b>: optional `tfd.Distribution` instance specifying a prior
  on the `drift_scale` parameter. If `None`, a heuristic default prior is
  constructed based on the provided `observed_time_series`.
  Default value: `None`.
* <b>`initial_effect_prior`</b>: optional `tfd.Distribution` instance specifying a
  normal prior on the initial effect of each season. This may be either
  a scalar `tfd.Normal` prior, in which case it applies independently to
  every season, or it may be multivariate normal (e.g.,
  `tfd.MultivariateNormalDiag`) with event shape `[num_seasons]`, in
  which case it specifies a joint prior across all seasons. If `None`, a
  heuristic default prior is constructed based on the provided
  `observed_time_series`.
  Default value: `None`.
* <b>`constrain_mean_effect_to_zero`</b>: if `True`, use a model parameterization
  that constrains the mean effect across all seasons to be zero. This
  constraint is generally helpful in identifying the contributions of
  different model components and can lead to more interpretable
  posterior decompositions. It may be undesirable if you plan to directly
  examine the latent space of the underlying state space model.
  Default value: `True`.
* <b>`observed_time_series`</b>: optional `float` `Tensor` of shape
  `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
  supported when `T > 1`), specifying an observed time series.
  Any priors not explicitly set will be given default values according to
  the scale of the observed time series (or batch of time series). May
  optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>, which includes
  a mask `Tensor` to specify timesteps with missing observations.
  Default value: `None`.
* <b>`name`</b>: the name of this model component.
  Default value: 'Seasonal'.



## Properties

<h3 id="batch_shape"><code>batch_shape</code></h3>

Static batch shape of models represented by this component.


#### Returns:


* <b>`batch_shape`</b>: A `tf.TensorShape` giving the broadcast batch shape of
  all model parameters. This should match the batch shape of
  derived state space models, i.e.,
  `self.make_state_space_model(...).batch_shape`. It may be partially
  defined or unknown.

<h3 id="constrain_mean_effect_to_zero"><code>constrain_mean_effect_to_zero</code></h3>

Whether to constrain the mean effect to zero.


<h3 id="initial_state_prior"><code>initial_state_prior</code></h3>

Prior distribution on the initial latent state (level and scale).


<h3 id="latent_size"><code>latent_size</code></h3>

Python `int` dimensionality of the latent space in this model.


<h3 id="name"><code>name</code></h3>

Name of this model component.


<h3 id="num_seasons"><code>num_seasons</code></h3>

Number of seasons.


<h3 id="num_steps_per_season"><code>num_steps_per_season</code></h3>

Number of steps per season.


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



