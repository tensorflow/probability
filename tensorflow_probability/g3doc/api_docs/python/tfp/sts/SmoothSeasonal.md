<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.SmoothSeasonal" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="allow_drift"/>
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="frequency_multipliers"/>
<meta itemprop="property" content="initial_state_prior"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="period"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.SmoothSeasonal


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/smooth_seasonal.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `SmoothSeasonal`

Formal representation of a smooth seasonal effect model.

Inherits From: [`StructuralTimeSeries`](../../tfp/sts/StructuralTimeSeries.md)

<!-- Placeholder for "Used in" -->

The smooth seasonal model uses a set of trigonometric terms in order to
capture a recurring pattern whereby adjacent (in time) effects are
similar. The model uses `frequencies` calculated via:

```python
frequencies[j] = 2. * pi * frequency_multipliers[j] / period
```

and then posits two latent states for each `frequency`. The two latent states
associated with frequency `j` drift over time via:

```python
effect[t] = (effect[t-1] * cos(frequencies[j]) +
             auxiliary[t-] * sin(frequencies[j]) +
             Normal(0., drift_scale))

auxiliary[t] = (-effect[t-1] * sin(frequencies[j]) +
                auxiliary[t-] * cos(frequencies[j]) +
                Normal(0., drift_scale))
```

where `effect` is the smooth seasonal effect and `auxiliary` only appears as a
matter of construction. The interpretation of `auxiliary` is thus not
particularly important.

#### Examples

A smooth seasonal effect model representing smooth weekly seasonality on daily
data:

```python
component = SmoothSeasonal(
    period=7,
    frequency_multipliers=[1, 2, 3],
    initial_state_prior=tfd.MultivariateNormalDiag(scale_diag=tf.ones([6])),
)
```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/smooth_seasonal.py">View source</a>

``` python
__init__(
    period,
    frequency_multipliers,
    allow_drift=True,
    drift_scale_prior=None,
    initial_state_prior=None,
    observed_time_series=None,
    name=None
)
```

Specify a smooth seasonal effects model.


#### Args:


* <b>`period`</b>: positive scalar `float` `Tensor` giving the number of timesteps
  required for the longest cyclic effect to repeat.
* <b>`frequency_multipliers`</b>: One-dimensional `float` `Tensor` listing the
  frequencies (cyclic components) included in the model, as multipliers of
  the base/fundamental frequency `2. * pi / period`. Each component is
  specified by the number of times it repeats per period, and adds two
  latent dimensions to the model. A smooth seasonal model that can
  represent any periodic function is given by `frequency_multipliers = [1,
  2, ..., floor(period / 2)]`. However, it is often desirable to enforce a
  smoothness assumption (and reduce the computational burden) by dropping
  some of the higher frequencies.
* <b>`allow_drift`</b>: optional Python `bool` specifying whether the seasonal
  effects can drift over time.  Setting this to `False`
  removes the `drift_scale` parameter from the model. This is
  mathematically equivalent to
  `drift_scale_prior = tfd.Deterministic(0.)`, but removing drift
  directly is preferred because it avoids the use of a degenerate prior.
  Default value: `True`.
* <b>`drift_scale_prior`</b>: optional `tfd.Distribution` instance specifying a prior
  on the `drift_scale` parameter. If `None`, a heuristic default prior is
  constructed based on the provided `observed_time_series`.
  Default value: `None`.
* <b>`initial_state_prior`</b>: instance of `tfd.MultivariateNormal` representing
  the prior distribution on the latent states. Must have event shape
  `[2 * len(frequency_multipliers)]`. If `None`, a heuristic default prior
  is constructed based on the provided `observed_time_series`.
* <b>`observed_time_series`</b>: optional `float` `Tensor` of shape
  `batch_shape + [T, 1]` (omitting the trailing unit dimension is also
  supported when `T > 1`), specifying an observed time series.
  Any priors not explicitly set will be given default values according to
  the scale of the observed time series (or batch of time series). May
  optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>, which includes
  a mask `Tensor` to specify timesteps with missing observations.
  Default value: `None`.
* <b>`name`</b>: the name of this model component.
  Default value: 'SmoothSeasonal'.



## Properties

<h3 id="allow_drift"><code>allow_drift</code></h3>

Whether the seasonal effects are allowed to drift over time.


<h3 id="batch_shape"><code>batch_shape</code></h3>

Static batch shape of models represented by this component.


#### Returns:


* <b>`batch_shape`</b>: A `tf.TensorShape` giving the broadcast batch shape of
  all model parameters. This should match the batch shape of
  derived state space models, i.e.,
  `self.make_state_space_model(...).batch_shape`. It may be partially
  defined or unknown.

<h3 id="frequency_multipliers"><code>frequency_multipliers</code></h3>

Multipliers of the fundamental frequency.


<h3 id="initial_state_prior"><code>initial_state_prior</code></h3>

Prior distribution on the initial latent states.


<h3 id="latent_size"><code>latent_size</code></h3>

Python `int` dimensionality of the latent space in this model.


<h3 id="name"><code>name</code></h3>

Name of this model component.


<h3 id="parameters"><code>parameters</code></h3>

List of Parameter(name, prior, bijector) namedtuples for this model.


<h3 id="period"><code>period</code></h3>

The seasonal period.




## Methods

<h3 id="batch_shape_tensor"><code>batch_shape_tensor</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/structural_time_series.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/structural_time_series.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/structural_time_series.py">View source</a>

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

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/structural_time_series.py">View source</a>

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



