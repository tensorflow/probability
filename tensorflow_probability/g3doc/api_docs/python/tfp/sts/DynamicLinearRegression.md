<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.DynamicLinearRegression" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="design_matrix"/>
<meta itemprop="property" content="initial_state_prior"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.DynamicLinearRegression

## Class `DynamicLinearRegression`

Formal representation of a dynamic linear regresson model.

Inherits From: [`StructuralTimeSeries`](../../tfp/sts/StructuralTimeSeries.md)



Defined in [`python/sts/dynamic_regression.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/dynamic_regression.py).

<!-- Placeholder for "Used in" -->

The dynamic linear regression model is a special case of a linear Gaussian SSM
and a generalization of typical (static) linear regression. The model
represents regression `weights` with a latent state which evolves via a
Gaussian random walk:

```
weights[t] ~ Normal(weights[t-1], drift_scale)
```

The latent state has dimension `num_features`, while the parameters
`drift_scale` and `observation_noise_scale` are each (a batch of) scalars. The
batch shape of this `Distribution` is the broadcast batch shape of these
parameters, the `initial_state_prior`, and the `design_matrix`. `num_features`
is determined from the last dimension of `design_matrix` (equivalent to the
number of columns in the design matrix in linear regression).

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    design_matrix,
    drift_scale_prior=None,
    initial_weights_prior=None,
    observed_time_series=None,
    name=None
)
```

Specify a dynamic linear regression.


#### Args:


* <b>`design_matrix`</b>: float `Tensor` of shape `concat([batch_shape,
  [num_timesteps, num_features]])`.
* <b>`drift_scale_prior`</b>: instance of `tfd.Distribution` specifying a prior on
  the `drift_scale` parameter. If `None`, a heuristic default prior is
  constructed based on the provided `observed_time_series`.
  Default value: `None`.
* <b>`initial_weights_prior`</b>: instance of `tfd.MultivariateNormal` representing
  the prior distribution on the latent states (the regression weights).
  Must have event shape `[num_features]`. If `None`, a weakly-informative
  Normal(0., 10.) prior is used.
  Default value: `None`.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape `batch_shape + [T, 1]`
  (omitting the trailing unit dimension is also supported when `T > 1`),
  specifying an observed time series. Any priors not explicitly set will
  be given default values according to the scale of the observed time
  series (or batch of time series). May optionally be an instance of
  <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>, which includes a mask `Tensor` to specify
  timesteps with missing observations.
  Default value: `None`.
* <b>`name`</b>: Python `str` for the name of this component.
  Default value: 'DynamicLinearRegression'.



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

Tensor representing the design matrix.


<h3 id="initial_state_prior"><code>initial_state_prior</code></h3>

Prior distribution on the initial latent state (level and scale).


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



