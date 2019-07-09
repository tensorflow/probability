<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.decompose_forecast_by_component" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.decompose_forecast_by_component

Decompose a forecast distribution into contributions from each component.

``` python
tfp.sts.decompose_forecast_by_component(
    model,
    forecast_dist,
    parameter_samples
)
```



Defined in [`python/sts/decomposition.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/decomposition.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`model`</b>: An instance of <a href="../../tfp/sts/Sum.md"><code>tfp.sts.Sum</code></a> representing a structural time series
  model.
* <b>`forecast_dist`</b>: A `Distribution` instance returned by `tfp.sts.forecast()`.
  (specifically, must be a `tfd.MixtureSameFamily` over a
  `tfd.LinearGaussianStateSpaceModel` parameterized by posterior samples).
* <b>`parameter_samples`</b>: Python `list` of `Tensors` representing posterior samples
  of model parameters, with shapes `[concat([[num_posterior_draws],
  param.prior.batch_shape, param.prior.event_shape]) for param in
  model.parameters]`. This may optionally also be a map (Python `dict`) of
  parameter names to `Tensor` values.

#### Returns:


* <b>`component_forecasts`</b>: A `collections.OrderedDict` instance mapping
  component StructuralTimeSeries instances (elements of `model.components`)
  to `tfd.Distribution` instances representing the marginal forecast for
  each component. Each distribution has batch and event shape matching
  `forecast_dist` (specifically, the event shape is
  `[num_steps_forecast]`).

#### Examples

Suppose we've built a model, fit it to data, and constructed a forecast
distribution:

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

  num_steps_forecast = 50
  samples, kernel_results = tfp.sts.fit_with_hmc(model, observed_time_series)
  forecast_dist = tfp.sts.forecast(model, observed_time_series,
                               parameter_samples=samples,
                               num_steps_forecast=num_steps_forecast)
```

To extract the forecast for individual components, pass the forecast
distribution into `decompose_forecast_by_components`:

```python
  component_forecasts = decompose_forecast_by_component(
    model, forecast_dist, samples)

  # Component mean and stddev have shape `[num_steps_forecast]`.
  day_of_week_effect_mean = forecast_components[day_of_week].mean()
  day_of_week_effect_stddev = forecast_components[day_of_week].stddev()
```

Using the component forecasts, we can visualize the uncertainty for each
component:

```
from matplotlib import pylab as plt
num_components = len(component_forecasts)
xs = np.arange(num_steps_forecast)
fig = plt.figure(figsize=(12, 3 * num_components))
for i, (component, component_dist) in enumerate(component_forecasts.items()):

  # If in graph mode, replace `.numpy()` with `.eval()` or `sess.run()`.
  component_mean = component_dist.mean().numpy()
  component_stddev = component_dist.stddev().numpy()

  ax = fig.add_subplot(num_components, 1, 1 + i)
  ax.plot(xs, component_mean, lw=2)
  ax.fill_between(xs,
                  component_mean - 2 * component_stddev,
                  component_mean + 2 * component_stddev,
                  alpha=0.5)
  ax.set_title(component.name)
```