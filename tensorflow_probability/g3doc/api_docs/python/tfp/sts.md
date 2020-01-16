<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.sts


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Framework for Bayesian structural time series models.

<!-- Placeholder for "Used in" -->


## Classes

[`class AdditiveStateSpaceModel`](../tfp/sts/AdditiveStateSpaceModel.md): A state space model representing a sum of component state space models.

[`class Autoregressive`](../tfp/sts/Autoregressive.md): Formal representation of an autoregressive model.

[`class AutoregressiveStateSpaceModel`](../tfp/sts/AutoregressiveStateSpaceModel.md): State space model for an autoregressive process.

[`class ConstrainedSeasonalStateSpaceModel`](../tfp/sts/ConstrainedSeasonalStateSpaceModel.md): Seasonal state space model with effects constrained to sum to zero.

[`class DynamicLinearRegression`](../tfp/sts/DynamicLinearRegression.md): Formal representation of a dynamic linear regresson model.

[`class DynamicLinearRegressionStateSpaceModel`](../tfp/sts/DynamicLinearRegressionStateSpaceModel.md): State space model for a dynamic linear regression from provided covariates.

[`class LinearRegression`](../tfp/sts/LinearRegression.md): Formal representation of a linear regression from provided covariates.

[`class LocalLevel`](../tfp/sts/LocalLevel.md): Formal representation of a local level model.

[`class LocalLevelStateSpaceModel`](../tfp/sts/LocalLevelStateSpaceModel.md): State space model for a local level.

[`class LocalLinearTrend`](../tfp/sts/LocalLinearTrend.md): Formal representation of a local linear trend model.

[`class LocalLinearTrendStateSpaceModel`](../tfp/sts/LocalLinearTrendStateSpaceModel.md): State space model for a local linear trend.

[`class MaskedTimeSeries`](../tfp/sts/MaskedTimeSeries.md): Named tuple encoding a time series `Tensor` and optional missingness mask.

[`class Seasonal`](../tfp/sts/Seasonal.md): Formal representation of a seasonal effect model.

[`class SeasonalStateSpaceModel`](../tfp/sts/SeasonalStateSpaceModel.md): State space model for a seasonal effect.

[`class SemiLocalLinearTrend`](../tfp/sts/SemiLocalLinearTrend.md): Formal representation of a semi-local linear trend model.

[`class SemiLocalLinearTrendStateSpaceModel`](../tfp/sts/SemiLocalLinearTrendStateSpaceModel.md): State space model for a semi-local linear trend.

[`class SmoothSeasonal`](../tfp/sts/SmoothSeasonal.md): Formal representation of a smooth seasonal effect model.

[`class SmoothSeasonalStateSpaceModel`](../tfp/sts/SmoothSeasonalStateSpaceModel.md): State space model for a smooth seasonal effect.

[`class SparseLinearRegression`](../tfp/sts/SparseLinearRegression.md): Formal representation of a sparse linear regression.

[`class StructuralTimeSeries`](../tfp/sts/StructuralTimeSeries.md): Base class for structural time series models.

[`class Sum`](../tfp/sts/Sum.md): Sum of structural time series components.

## Functions

[`build_factored_surrogate_posterior(...)`](../tfp/sts/build_factored_surrogate_posterior.md): Build a variational posterior that factors over model parameters.

[`build_factored_variational_loss(...)`](../tfp/sts/build_factored_variational_loss.md): Build a loss function for variational inference in STS models. (deprecated)

[`decompose_by_component(...)`](../tfp/sts/decompose_by_component.md): Decompose an observed time series into contributions from each component.

[`decompose_forecast_by_component(...)`](../tfp/sts/decompose_forecast_by_component.md): Decompose a forecast distribution into contributions from each component.

[`fit_with_hmc(...)`](../tfp/sts/fit_with_hmc.md): Draw posterior samples using Hamiltonian Monte Carlo (HMC).

[`forecast(...)`](../tfp/sts/forecast.md): Construct predictive distribution over future observations.

[`impute_missing_values(...)`](../tfp/sts/impute_missing_values.md): Runs posterior inference to impute the missing values in a time series.

[`one_step_predictive(...)`](../tfp/sts/one_step_predictive.md): Compute one-step-ahead predictive distributions for all timesteps.

[`sample_uniform_initial_state(...)`](../tfp/sts/sample_uniform_initial_state.md): Initialize from a uniform [-2, 2] distribution in unconstrained space.

