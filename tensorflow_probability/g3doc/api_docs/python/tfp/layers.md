<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.layers

Probabilistic Layers.

## Classes

[`class CategoricalMixtureOfOneHotCategorical`](../tfp/layers/CategoricalMixtureOfOneHotCategorical.md): A OneHotCategorical mixture Keras layer from `k * (1 + d)` params.

[`class Convolution1DFlipout`](../tfp/layers/Convolution1DFlipout.md): 1D convolution layer (e.g. temporal convolution) with Flipout.

[`class Convolution1DReparameterization`](../tfp/layers/Convolution1DReparameterization.md): 1D convolution layer (e.g. temporal convolution).

[`class Convolution2DFlipout`](../tfp/layers/Convolution2DFlipout.md): 2D convolution layer (e.g. spatial convolution over images) with Flipout.

[`class Convolution2DReparameterization`](../tfp/layers/Convolution2DReparameterization.md): 2D convolution layer (e.g. spatial convolution over images).

[`class Convolution3DFlipout`](../tfp/layers/Convolution3DFlipout.md): 3D convolution layer (e.g. spatial convolution over volumes) with Flipout.

[`class Convolution3DReparameterization`](../tfp/layers/Convolution3DReparameterization.md): 3D convolution layer (e.g. spatial convolution over volumes).

[`class DenseFlipout`](../tfp/layers/DenseFlipout.md): Densely-connected layer class with Flipout estimator.

[`class DenseLocalReparameterization`](../tfp/layers/DenseLocalReparameterization.md): Densely-connected layer class with local reparameterization estimator.

[`class DenseReparameterization`](../tfp/layers/DenseReparameterization.md): Densely-connected layer class with reparameterization estimator.

[`class DistributionLambda`](../tfp/layers/DistributionLambda.md): Keras layer enabling plumbing TFP distributions through Keras models.

[`class IndependentBernoulli`](../tfp/layers/IndependentBernoulli.md): An Independent-Bernoulli Keras layer from `prod(event_shape)` params.

[`class IndependentLogistic`](../tfp/layers/IndependentLogistic.md): An independent logistic Keras layer.

[`class IndependentNormal`](../tfp/layers/IndependentNormal.md): An independent normal Keras layer.

[`class IndependentPoisson`](../tfp/layers/IndependentPoisson.md): An independent Poisson Keras layer.

[`class KLDivergenceAddLoss`](../tfp/layers/KLDivergenceAddLoss.md): Pass-through layer that adds a KL divergence penalty to the model loss.

[`class KLDivergenceRegularizer`](../tfp/layers/KLDivergenceRegularizer.md): Regularizer that adds a KL divergence penalty to the model loss.

[`class MixtureSameFamily`](../tfp/layers/MixtureSameFamily.md): A mixture (same-family) Keras layer.

[`class MultivariateNormalTriL`](../tfp/layers/MultivariateNormalTriL.md): A `d`-variate MVNTriL Keras layer from `d + d * (d + 1) // 2` params.

[`class OneHotCategorical`](../tfp/layers/OneHotCategorical.md): A `d`-variate OneHotCategorical Keras layer from `d` params.

## Functions

[`default_loc_scale_fn(...)`](../tfp/layers/default_loc_scale_fn.md): Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

[`default_mean_field_normal_fn(...)`](../tfp/layers/default_mean_field_normal_fn.md): Creates a function to build Normal distributions with trainable params.

[`default_multivariate_normal_fn(...)`](../tfp/layers/default_multivariate_normal_fn.md): Creates multivariate standard `Normal` distribution.

