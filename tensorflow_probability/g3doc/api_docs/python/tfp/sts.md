<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.sts

Framework for Bayesian structural time series models.

## Classes

[`class AdditiveStateSpaceModel`](../tfp/sts/AdditiveStateSpaceModel.md): A state space model representing a sum of component state space models.

[`class LinearRegression`](../tfp/sts/LinearRegression.md): Formal representation of a linear regression from provided covariates.

[`class LocalLinearTrend`](../tfp/sts/LocalLinearTrend.md): Formal representation of a local linear trend model.

[`class LocalLinearTrendStateSpaceModel`](../tfp/sts/LocalLinearTrendStateSpaceModel.md): State space model for a local linear trend.

[`class Seasonal`](../tfp/sts/Seasonal.md): Formal representation of a seasonal effect model.

[`class SeasonalStateSpaceModel`](../tfp/sts/SeasonalStateSpaceModel.md): State space model for a seasonal effect.

[`class StructuralTimeSeries`](../tfp/sts/StructuralTimeSeries.md): Base class for structural time series models.

[`class Sum`](../tfp/sts/Sum.md): Sum of structural time series components.

## Functions

[`build_factored_variational_loss(...)`](../tfp/sts/build_factored_variational_loss.md): Build a loss function for variational inference in STS models.

[`fit_with_hmc(...)`](../tfp/sts/fit_with_hmc.md): Draw posterior samples using Hamiltonian Monte Carlo (HMC).

[`forecast(...)`](../tfp/sts/forecast.md): Construct predictive distribution over future observations.

[`one_step_predictive(...)`](../tfp/sts/one_step_predictive.md): Compute one-step-ahead predictive distributions for all timesteps.

[`sample_uniform_initial_state(...)`](../tfp/sts/sample_uniform_initial_state.md): Initialize from a uniform [-2, 2] distribution in unconstrained space.

