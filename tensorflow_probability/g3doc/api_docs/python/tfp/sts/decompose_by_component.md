<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.decompose_by_component" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.decompose_by_component

Decompose an observed time series into contributions from each component.

``` python
tfp.sts.decompose_by_component(
    model,
    observed_time_series,
    parameter_samples
)
```



Defined in [`python/sts/decomposition.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/decomposition.py).

<!-- Placeholder for "Used in" -->

This method decomposes a time series according to the posterior represention
of a structural time series model. In particular, it:
  - Computes the posterior marginal mean and covariances over the additive
    model's latent space.
  - Decomposes the latent posterior into the marginal blocks for each
    model component.
  - Maps the per-component latent posteriors back through each component's
    observation model, to generate the time series modeled by that component.

#### Args:

* <b>`model`</b>: An instance of <a href="../../tfp/sts/Sum.md"><code>tfp.sts.Sum</code></a> representing a structural time series
  model.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape
  `batch_shape + [num_timesteps, 1]` (omitting the trailing unit dimension
  is also supported when `num_timesteps > 1`), specifying an observed time
  series. May optionally be an instance of <a href="../../tfp/sts/MaskedTimeSeries.md"><code>tfp.sts.MaskedTimeSeries</code></a>, which
  includes a mask `Tensor` to specify timesteps with missing observations.
* <b>`parameter_samples`</b>: Python `list` of `Tensors` representing posterior
  samples of model parameters, with shapes `[concat([
  [num_posterior_draws], param.prior.batch_shape,
  param.prior.event_shape]) for param in model.parameters]`. This may
  optionally also be a map (Python `dict`) of parameter names to
  `Tensor` values.

#### Returns:

  component_dists: A `collections.OrderedDict` instance mapping
    component StructuralTimeSeries instances (elements of `model.components`)
    to `tfd.Distribution` instances representing the posterior marginal
    distributions on the process modeled by each component. Each distribution
    has batch shape matching that of `posterior_means`/`posterior_covs`, and
    event shape of `[num_timesteps]`.

#### Examples

Suppose we've built a model and fit it to data:

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
```

To extract the contributions of individual components, pass the time series
and sampled parameters into `decompose_by_component`:

```python
  component_dists = decompose_by_component(
    model,
    observed_time_series=observed_time_series,
    parameter_samples=samples)

  # Component mean and stddev have shape `[len(observed_time_series)]`.
  day_of_week_effect_mean = component_dists[day_of_week].mean()
  day_of_week_effect_stddev = component_dists[day_of_week].stddev()
```

Using the component distributions, we can visualize the uncertainty for
each component:

```
from matplotlib import pylab as plt
num_components = len(component_dists)
xs = np.arange(len(observed_time_series))
fig = plt.figure(figsize=(12, 3 * num_components))
for i, (component, component_dist) in enumerate(component_dists.items()):

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