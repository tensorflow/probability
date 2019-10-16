<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.impute_missing_values" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.impute_missing_values


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/forecast.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Runs posterior inference to impute the missing values in a time series.

``` python
tfp.sts.impute_missing_values(
    model,
    observed_time_series,
    parameter_samples,
    include_observation_noise=False
)
```



<!-- Placeholder for "Used in" -->

This method computes the posterior marginals `p(latent state | observations)`,
given the time series at observed timesteps (a missingness mask should
be specified using <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>). It pushes this posterior back
through the observation model to impute a predictive distribution on the
observed time series. At unobserved steps, this is an imputed value; at other
steps it is interpreted as the model's estimate of the underlying noise-free
series.

#### Args:


* <b>`model`</b>: <a href="../../tfp/sts/Sum.md"><code>tfp.sts.Sum</code></a> instance defining an additive STS model.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape
  `concat([sample_shape, model.batch_shape, [num_timesteps, 1]])` where
  `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
  dimension may (optionally) be omitted if `num_timesteps > 1`. May
  optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a> including a
  mask `Tensor` to encode the locations of missing observations.
* <b>`parameter_samples`</b>: Python `list` of `Tensors` representing posterior
  samples of model parameters, with shapes `[concat([
  [num_posterior_draws], param.prior.batch_shape,
  param.prior.event_shape]) for param in model.parameters]`. This may
  optionally also be a map (Python `dict`) of parameter names to
  `Tensor` values.
* <b>`include_observation_noise`</b>: If `False`, the imputed uncertainties
  represent the model's estimate of the noise-free time series at each
  timestep. If `True`, they represent the model's estimate of the range of
  values that could be *observed* at each timestep, including any i.i.d.
  observation noise.
  Default value: `False`.


#### Returns:


* <b>`imputed_series_dist`</b>: a `tfd.MixtureSameFamily` instance with event shape
  [num_timesteps] and batch shape `concat([sample_shape,
  model.batch_shape])`, with `num_posterior_draws` mixture components.

#### Example

To specify a time series with missing values, use <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>:

```python
time_series_with_nans = [-1., 1., np.nan, 2.4, np.nan, 5]
observed_time_series = tfp.sts.MaskedTimeSeries(
  time_series=time_series_with_nans,
  is_missing=tf.math.is_nan(time_series_with_nans))
```

Masked time series can be passed to <a href="../../tfp/sts.md"><code>tfp.sts</code></a> methods in place of a
`observed_time_series` `Tensor`:

```python
# Build model using observed time series to set heuristic priors.
linear_trend_model = tfp.sts.LocalLinearTrend(
  observed_time_series=observed_time_series)
model = tfp.sts.Sum([linear_trend_model],
                    observed_time_series=observed_time_series)

# Fit model to data
parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)
```

After fitting a model, `impute_missing_values` will return a distribution
```python
# Impute missing values
imputed_series_distribution = tfp.sts.impute_missing_values(
  model, observed_time_series)
print('imputed means and stddevs: ',
      imputed_series_distribution.mean(),
      imputed_series_distribution.stddev())
```