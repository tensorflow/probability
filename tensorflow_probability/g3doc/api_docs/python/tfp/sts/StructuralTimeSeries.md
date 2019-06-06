<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.StructuralTimeSeries" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.StructuralTimeSeries

## Class `StructuralTimeSeries`

Base class for structural time series models.





Defined in [`python/sts/structural_time_series.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/structural_time_series.py).

<!-- Placeholder for "Used in" -->

A StructuralTimeSeries object represents a declarative specification of a
structural time series model, including priors on model parameters.
It implements a joint probability model
  `p(params, y) = p(params) p(y | params)`,
where `params` denotes a list of real-valued parameters specified by the child
class, and `p(y | params)` is a linear Gaussian state space model with
structure determined by the child class.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    parameters,
    latent_size,
    name='StructuralTimeSeries'
)
```

Construct a specification for a structural time series model.


#### Args:


* <b>`parameters`</b>: list of Parameter namedtuples, each specifying the name
  and prior distribution of a model parameter along with a
  bijective transformation from an unconstrained space to the support
  of that parameter. The order of this list determines the canonical
  parameter ordering used by fitting and inference algorithms.
* <b>`latent_size`</b>: Python `int` specifying the dimensionality of the latent
  state space for this model.
* <b>`name`</b>: Python `str` name for this model component.



## Properties

<h3 id="batch_shape"><code>batch_shape</code></h3>

Static batch shape of models represented by this component.


#### Returns:


* <b>`batch_shape`</b>: A `tf.TensorShape` giving the broadcast batch shape of
  all model parameters. This should match the batch shape of
  derived state space models, i.e.,
  `self.make_state_space_model(...).batch_shape`. It may be partially
  defined or unknown.

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



