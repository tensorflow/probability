<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.SparseLinearRegression" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_shape"/>
<meta itemprop="property" content="design_matrix"/>
<meta itemprop="property" content="latent_size"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="weights_prior_scale"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_shape_tensor"/>
<meta itemprop="property" content="joint_log_prob"/>
<meta itemprop="property" content="make_state_space_model"/>
<meta itemprop="property" content="params_to_weights"/>
<meta itemprop="property" content="prior_sample"/>
</div>

# tfp.sts.SparseLinearRegression

## Class `SparseLinearRegression`

Formal representation of a sparse linear regression.

Inherits From: [`StructuralTimeSeries`](../../tfp/sts/StructuralTimeSeries.md)



Defined in [`python/sts/regression.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/regression.py).

<!-- Placeholder for "Used in" -->

This model defines a time series given by a sparse linear combination of
covariate time series provided in a design matrix:

```python
observed_time_series = matmul(design_matrix, weights)
```

This is identical to <a href="../../tfp/sts/LinearRegression.md"><code>tfp.sts.LinearRegression</code></a>, except that
`SparseLinearRegression` uses a parameterization of a Horseshoe
prior [1][2] to encode the assumption that many of the `weights` are zero,
i.e., many of the covariate time series are irrelevant. See the mathematical
details section below for further discussion. The prior parameterization used
by `SparseLinearRegression` is more suitable for inference than that
obtained by simply passing the equivalent `tfd.Horseshoe` prior to
`LinearRegression`; when sparsity is desired, `SparseLinearRegression` will
likely yield better results.

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
regression = tfp.sts.SparseLinearRegression(
  design_matrix=tf.stack([series1, series2], axis=-1),
  weights_prior_scale=0.1)
```

The `weights_prior_scale` determines the level of sparsity; small
scales encourage the weights to be sparse. In some cases, such as when
the likelihood is iid Gaussian with known scale, the prior scale can be
analytically related to the expected number of nonzero weights [2]; however,
this is not the case in general for STS models.

If the design matrix has batch dimensions, by default the model will create a
matching batch of weights. For example, if `design_matrix.shape == [
num_users, num_timesteps, num_features]`, by default the model will fit
separate weights for each user, i.e., it will internally represent
`weights.shape == [num_users, num_features]`. To share weights across some or
all batch dimensions, you can manually specify the batch shape for the
weights:

```python
# design_matrix.shape == [num_users, num_timesteps, num_features]
regression = tfp.sts.SparseLinearRegression(
  design_matrix=design_matrix,
  weights_batch_shape=[])  # weights.shape -> [num_features]
```

#### Mathematical Details

The basic horseshoe prior [1] is defined as a Cauchy-normal scale mixture:

```
scales[i] ~ HalfCauchy(loc=0, scale=1)
weights[i] ~ Normal(loc=0., scale=scales[i] * global_scale)`
```

The Cauchy scale parameters puts substantial mass near zero, encouraging
weights to be sparse, but their heavy tails allow weights far from zero to be
estimated without excessive shrinkage. The horseshoe can be thought of as a
continuous relaxation of a traditional 'spike-and-slab' discrete sparsity
prior, in which the latent Cauchy scale mixes between 'spike'
(`scales[i] ~= 0`) and 'slab' (`scales[i] >> 0`) regimes.

Following the recommendations in [2], `SparseLinearRegression` implements
a horseshoe with the following adaptations:

- The Cauchy prior on `scales[i]` is represented as an InverseGamma-Normal
  compound.
- The `global_scale` parameter is integrated out following a `Cauchy(0.,
  scale=weights_prior_scale)` hyperprior, which is also represented as an
  InverseGamma-Normal compound.
- All compound distributions are implemented using a non-centered
  parameterization.

The compound, non-centered representation defines the same marginal prior as
the original horseshoe (up to integrating out the global scale),
but allows samplers to mix more efficiently through the heavy tails; for
variational inference, the compound representation implicity expands the
representational power of the variational model.

Note that we do not yet implement the regularized ('Finnish') horseshoe,
proposed in [2] for models with weak likelihoods, because the likelihood
in STS models is typically Gaussian, where it's not clear that additional
regularization is appropriate. If you need this functionality, please
email tfprobability@tensorflow.org.

The full prior parameterization implemented in `SparseLinearRegression` is
as follows:

```
# Sample global_scale from Cauchy(0, scale=weights_prior_scale).
global_scale_variance ~ InverseGamma(alpha=0.5, beta=0.5)
global_scale_noncentered ~ HalfNormal(loc=0, scale=1)
global_scale = (global_scale_noncentered *
                sqrt(global_scale_variance) *
                weights_prior_scale)

# Sample local_scales from Cauchy(0, 1).
local_scale_variances[i] ~ InverseGamma(alpha=0.5, beta=0.5)
local_scales_noncentered[i] ~ HalfNormal(loc=0, scale=1)
local_scales[i] = local_scales_noncentered[i] * sqrt(local_scale_variances[i])

weights[i] ~ Normal(loc=0., scale=local_scales[i] * global_scale)
```

#### References

[1]: Carvalho, C., Polson, N. and Scott, J. Handling Sparsity via the
  Horseshoe. AISTATS (2009).
  http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf
[2]: Juho Piironen, Aki Vehtari. Sparsity information and regularization in
  the horseshoe and other shrinkage priors (2017).
  https://arxiv.org/abs/1707.01694

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    design_matrix,
    weights_prior_scale=0.1,
    weights_batch_shape=None,
    name=None
)
```

Specify a sparse linear regression model.


#### Args:


* <b>`design_matrix`</b>: float `Tensor` of shape `concat([batch_shape,
  [num_timesteps, num_features]])`. This may also optionally be
  an instance of `tf.linalg.LinearOperator`.
* <b>`weights_prior_scale`</b>: float `Tensor` defining the scale of the Horseshoe
  prior on regression weights. Small values encourage the weights to be
  sparse. The shape must broadcast with `weights_batch_shape`.
  Default value: `0.1`.
* <b>`weights_batch_shape`</b>: if `None`, defaults to
  `design_matrix.batch_shape_tensor()`. Must broadcast with the batch
  shape of `design_matrix`.
  Default value: `None`.
* <b>`name`</b>: the name of this model component.
  Default value: 'SparseLinearRegression'.



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


<h3 id="weights_prior_scale"><code>weights_prior_scale</code></h3>






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

<h3 id="params_to_weights"><code>params_to_weights</code></h3>

``` python
params_to_weights(
    global_scale_variance,
    global_scale_noncentered,
    local_scale_variances,
    local_scales_noncentered,
    weights_noncentered
)
```

Build regression weights from model parameters.


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



