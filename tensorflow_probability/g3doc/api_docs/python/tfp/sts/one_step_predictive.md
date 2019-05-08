<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.one_step_predictive" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.one_step_predictive

Compute one-step-ahead predictive distributions for all timesteps.

``` python
tfp.sts.one_step_predictive(
    model,
    observed_time_series,
    parameter_samples
)
```



Defined in [`python/sts/forecast.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/forecast.py).

<!-- Placeholder for "Used in" -->

Given samples from the posterior over parameters, return the predictive
distribution over observations at each time `T`, given observations up
through time `T-1`.

#### Args:

* <b>`model`</b>: An instance of `StructuralTimeSeries` representing a
  time-series model. This represents a joint distribution over
  time-series and their parameters with batch shape `[b1, ..., bN]`.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape
  `concat([sample_shape, model.batch_shape, [num_timesteps, 1]]) where
  `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
  dimension may (optionally) be omitted if `num_timesteps > 1`. May
  optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a> including a
  mask `Tensor` to encode the locations of missing observations.
* <b>`parameter_samples`</b>: Python `list` of `Tensors` representing posterior samples
  of model parameters, with shapes `[concat([[num_posterior_draws],
  param.prior.batch_shape, param.prior.event_shape]) for param in
  model.parameters]`. This may optionally also be a map (Python `dict`) of
  parameter names to `Tensor` values.


#### Returns:

  forecast_dist: a `tfd.MixtureSameFamily` instance with event shape
    [num_timesteps] and
    batch shape `concat([sample_shape, model.batch_shape])`, with
    `num_posterior_draws` mixture components. The `t`th step represents the
    forecast distribution `p(observed_time_series[t] |
    observed_time_series[0:t-1], parameter_samples)`.

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

Passing the posterior samples into `one_step_predictive`, we construct a
one-step-ahead predictive distribution:

```python
  one_step_predictive_dist = tfp.sts.one_step_predictive(
    model, observed_time_series, parameter_samples=samples)

  predictive_means = one_step_predictive_dist.mean()
  predictive_scales = one_step_predictive_dist.stddev()
```

If using variational inference instead of HMC, we'd construct a forecast using
samples from the variational posterior:

```python
  (variational_loss,
   variational_distributions) = tfp.sts.build_factored_variational_loss(
     model=model, observed_time_series=observed_time_series)

  # OMITTED: take steps to optimize variational loss

  samples = {k: q.sample(30) for (k, q) in variational_distributions.items()}
  one_step_predictive_dist = tfp.sts.one_step_predictive(
    model, observed_time_series, parameter_samples=samples)
```

We can visualize the forecast by plotting:

```python
  from matplotlib import pylab as plt
  def plot_one_step_predictive(observed_time_series,
                               forecast_mean,
                               forecast_scale):
    plt.figure(figsize=(12, 6))
    num_timesteps = forecast_mean.shape[-1]
    c1, c2 = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05)
    plt.plot(observed_time_series, label="observed time series", color=c1)
    plt.plot(forecast_mean, label="one-step prediction", color=c2)
    plt.fill_between(np.arange(num_timesteps),
                     forecast_mean - 2 * forecast_scale,
                     forecast_mean + 2 * forecast_scale,
                     alpha=0.1, color=c2)
    plt.legend()

  plot_one_step_predictive(observed_time_series,
                           forecast_mean=predictive_means,
                           forecast_scale=predictive_scales)
```

To detect anomalous timesteps, we check whether the observed value at each
step is within a 95% predictive interval, i.e., two standard deviations from
the mean:

```python
  z_scores = ((observed_time_series[..., 1:] - predictive_means[..., :-1])
               / predictive_scales[..., :-1])
  anomalous_timesteps = tf.boolean_mask(
      tf.range(1, num_timesteps),
      tf.abs(z_scores) > 2.0)
```