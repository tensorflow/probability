<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.glm

TensorFlow Probability GLM python package.



Defined in [`python/glm/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/glm/__init__.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class Bernoulli`](../tfp/glm/Bernoulli.md): `Bernoulli(probs=mean)` where `mean = sigmoid(matmul(X, weights))`.

[`class BernoulliNormalCDF`](../tfp/glm/BernoulliNormalCDF.md): `Bernoulli(probs=mean)` where

[`class CustomExponentialFamily`](../tfp/glm/CustomExponentialFamily.md): Constucts GLM from arbitrary distribution and inverse link function.

[`class ExponentialFamily`](../tfp/glm/ExponentialFamily.md): Specifies a mean-value parameterized exponential family.

[`class GammaExp`](../tfp/glm/GammaExp.md): `Gamma(concentration=1, rate=1 / mean)` where

[`class GammaSoftplus`](../tfp/glm/GammaSoftplus.md): `Gamma(concentration=1, rate=1 / mean)` where

[`class LogNormal`](../tfp/glm/LogNormal.md): `LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))` where

[`class LogNormalSoftplus`](../tfp/glm/LogNormalSoftplus.md): `LogNormal(loc=log(mean) - log(2) / 2, scale=sqrt(log(2)))`

[`class Normal`](../tfp/glm/Normal.md): `Normal(loc=mean, scale=1)` where `mean = matmul(X, weights)`.

[`class NormalReciprocal`](../tfp/glm/NormalReciprocal.md): `Normal(loc=mean, scale=1)` where `mean = 1 / matmul(X, weights)`.

[`class Poisson`](../tfp/glm/Poisson.md): `Poisson(rate=mean)` where `mean = exp(matmul(X, weights))`.

[`class PoissonSoftplus`](../tfp/glm/PoissonSoftplus.md): `Poisson(rate=mean)` where `mean = softplus(matmul(X, weights))`.

## Functions

[`convergence_criteria_small_relative_norm_weights_change(...)`](../tfp/glm/convergence_criteria_small_relative_norm_weights_change.md): Returns Python `callable` which indicates fitting procedure has converged.

[`fit(...)`](../tfp/glm/fit.md): Runs multiple Fisher scoring steps.

[`fit_one_step(...)`](../tfp/glm/fit_one_step.md): Runs one step of Fisher scoring.

[`fit_sparse(...)`](../tfp/glm/fit_sparse.md): Fits a GLM using coordinate-wise FIM-informed proximal gradient descent.

[`fit_sparse_one_step(...)`](../tfp/glm/fit_sparse_one_step.md): One step of (the outer loop of) the GLM fitting algorithm.

