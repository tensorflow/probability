<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.MaskedTimeSeries" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="time_series"/>
<meta itemprop="property" content="is_missing"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.sts.MaskedTimeSeries

## Class `MaskedTimeSeries`

Named tuple encoding a time series `Tensor` and optional missingness mask.





Defined in [`python/sts/internal/missing_values_util.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/sts/internal/missing_values_util.py).

<!-- Placeholder for "Used in" -->

Structural time series models handle missing values naturally, following the
rules of conditional probability. Posterior inference can be used to impute
missing values, with uncertainties. Forecasting and posterior decomposition
are also supported for time series with missing values; the missing values
will generally lead to corresponding higher forecast uncertainty.

All methods in the <a href="../../tfp/sts.md"><code>tfp.sts</code></a> API that accept an `observed_time_series`
`Tensor` should optionally also accept a `MaskedTimeSeries` instance.

The time series should be a float `Tensor` of shape `[..., num_timesteps]` or
`[..., num_timesteps, 1]`. The `is_missing` mask must be either a boolean
`Tensor` of shape `[..., num_timesteps]`, or `None`. `True` values in
`is_missing` denote missing (masked) observations; `False` denotes observed
(unmasked) values. Note that these semantics are opposite that of low-level
TensorFlow methods like `tf.boolean_mask`, but consistent with the behavior
of Numpy masked arrays.

The batch dimensions of `is_missing` must broadcast with the batch
dimensions of `time_series`.

A `MaskedTimeSeries` is just a `collections.namedtuple` instance, i.e., a dumb
container. Although the convention for the elements is as described here, it's
left to downstream methods to validate or convert the elements as required.
In particular, most downstream methods will call `tf.convert_to_tensor`
on the components. In order to prevent duplicate `Tensor` creation, you may
(if memory is an issue) wish to ensure that the components are *already*
`Tensors`, as opposed to numpy arrays or similar.

#### Examples

To construct a simple MaskedTimeSeries instance:

```
observed_time_series = tfp.sts.MaskedTimeSeries(
  time_series=tf.random_normal([3, 4, 5]),
  is_missing=[True, False, False, True, False])
```

Note that the mask we specified will broadcast against the batch dimensions of
the time series.

For time series with missing entries specified as NaN 'magic values', you can
generate a mask using `tf.is_nan`:

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

time_series_with_nans = [-1., 1., np.nan, 2.4, np.nan, 5]
observed_time_series = tfp.sts.MaskedTimeSeries(
  time_series=time_series_with_nans,
  is_missing=tf.is_nan(time_series_with_nans))

# Build model using observed time series to set heuristic priors.
linear_trend_model = tfp.sts.LocalLinearTrend(
  observed_time_series=observed_time_series)
model = tfp.sts.Sum([linear_trend_model],
                    observed_time_series=observed_time_series)

# Fit model to data
parameter_samples, _ = tfp.sts.fit_with_hmc(model, observed_time_series)

# Forecast
forecast_dist = tfp.sts.forecast(
  model, observed_time_series, num_steps_forecast=5)
```

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    time_series,
    is_missing
)
```

Create new instance of MaskedTimeSeries(time_series, is_missing)



## Properties

<h3 id="time_series"><code>time_series</code></h3>



<h3 id="is_missing"><code>is_missing</code></h3>





