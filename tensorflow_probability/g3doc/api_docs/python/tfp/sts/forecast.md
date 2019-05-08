<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.forecast" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.forecast

Construct predictive distribution over future observations.

``` python
tfp.sts.forecast(
    model,
    observed_time_series,
    parameter_samples,
    num_steps_forecast
)
```



Defined in [`python/sts/forecast.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/forecast.py).

<!-- Placeholder for "Used in" -->

Given samples from the posterior over parameters, return the predictive
distribution over future observations for num_steps_forecast timesteps.

#### Args:

* <b>`model`</b>: An instance of `StructuralTimeSeries` representing a
  time-series model. This represents a joint distribution over
  time-series and their parameters with batch shape `[b1, ..., bN]`.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape
  `concat([sample_shape, model.batch_shape, [num_timesteps, 1]])` where
  `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
  dimension may (optionally) be omitted if `num_timesteps > 1`. May
  optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a> including a
  mask `Tensor` to encode the locations of missing observations.
* <b>`parameter_samples`</b>: Python `list` of `Tensors` representing posterior samples
  of model parameters, with shapes `[concat([[num_posterior_draws],
  param.prior.batch_shape, param.prior.event_shape]) for param in
  model.parameters]`. This may optionally also be a map (Python `dict`) of
  parameter names to `Tensor` values.
* <b>`num_steps_forecast`</b>: scalar `int` `Tensor` number of steps to forecast.


#### Returns:

  forecast_dist: a `tfd.MixtureSameFamily` instance with event shape
    [num_steps_forecast, 1] and batch shape
    `concat([sample_shape, model.batch_shape])`, with `num_posterior_draws`
    mixture components.

#### Examples

Suppose we've built a model and fit it to data using HMC:

```python
  day_of_week = tfp.sts.Seasonal(
      num_seasons=7,
      observed_time_series=observed_time_series,
      name='day_of_week')
  local_linear_trend = tfp.sts.LocalLinearTrend(
      observed_time_series=observed_time_series,
      name='local_linear_trend')
  model = tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                      observed_time_series=observed_time_series)

  samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
```

Passing the posterior samples into `forecast`, we construct a forecast
* <b>`distribution`</b>: 
```python
  forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                   parameter_samples=samples,
                                   num_steps_forecast=50)

  forecast_mean = forecast_dist.mean()[..., 0]  # shape: [50]
  forecast_scale = forecast_dist.stddev()[..., 0]  # shape: [50]
  forecast_samples = forecast_dist.sample(10)[..., 0]  # shape: [10, 50]
```

If using variational inference instead of HMC, we'd construct a forecast using
samples from the variational posterior:

```python
  (variational_loss,
   variational_distributions) = tfp.sts.build_factored_variational_loss(
     model=model, observed_time_series=observed_time_series)

  # OMITTED: take steps to optimize variational loss

  samples = {k: q.sample(30) for (k, q) in variational_distributions.items()}
  forecast_dist = tfp.sts.forecast(model, observed_time_series,
                                       parameter_samples=samples,
                                       num_steps_forecast=50)
```

We can visualize the forecast by plotting:

```python
  from matplotlib import pylab as plt
  def plot_forecast(observed_time_series,
                    forecast_mean,
                    forecast_scale,
                    forecast_samples):
    plt.figure(figsize=(12, 6))

    num_steps = observed_time_series.shape[-1]
    num_steps_forecast = forecast_mean.shape[-1]
    num_steps_train = num_steps - num_steps_forecast

    c1, c2 = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05)
    plt.plot(np.arange(num_steps), observed_time_series,
             lw=2, color=c1, label='ground truth')

    forecast_steps = np.arange(num_steps_train,
                     num_steps_train+num_steps_forecast)
    plt.plot(forecast_steps, forecast_samples.T, lw=1, color=c2, alpha=0.1)
    plt.plot(forecast_steps, forecast_mean, lw=2, ls='--', color=c2,
             label='forecast')
    plt.fill_between(forecast_steps,
                     forecast_mean - 2 * forecast_scale,
                     forecast_mean + 2 * forecast_scale, color=c2, alpha=0.2)

    plt.xlim([0, num_steps])
    plt.legend()

  plot_forecast(observed_time_series,
                forecast_mean=forecast_mean,
                forecast_scale=forecast_scale,
                forecast_samples=forecast_samples)
```