<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.layers

Probabilistic Layers.



Defined in [`python/layers/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/__init__.py).

<!-- Placeholder for "Used in" -->


## Modules

[`conv_variational`](../tfp/layers/conv_variational.md) module: Convolutional variational layers.

[`dense_variational`](../tfp/layers/dense_variational.md) module: Dense variational layers.

[`dense_variational_v2`](../tfp/layers/dense_variational_v2.md) module: DenseVariational layer.

[`distribution_layer`](../tfp/layers/distribution_layer.md) module: Layers for combining <a href="../tfp/distributions.md"><code>tfp.distributions</code></a> and `tf.keras`.

[`initializers`](../tfp/layers/initializers.md) module: Keras initializers useful for TFP Keras layers.

[`internal`](../tfp/layers/internal.md) module

[`masked_autoregressive`](../tfp/layers/masked_autoregressive.md) module: Layers for normalizing flows and masked autoregressive density estimation.

[`util`](../tfp/layers/util.md) module: Utilities for probabilistic layers.

[`variable_input`](../tfp/layers/variable_input.md) module: VariableInputLayer.

## Classes

[`class AutoregressiveTransform`](../tfp/layers/AutoregressiveTransform.md): An autoregressive normalizing flow layer, given an `AutoregressiveLayer`.

[`class BlockwiseInitializer`](../tfp/layers/BlockwiseInitializer.md): Initializer which concats other intializers.

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

[`class DenseVariational`](../tfp/layers/DenseVariational.md): Dense layer with random `kernel` and `bias`.

[`class DistributionLambda`](../tfp/layers/DistributionLambda.md): Keras layer enabling plumbing TFP distributions through Keras models.

[`class IndependentBernoulli`](../tfp/layers/IndependentBernoulli.md): An Independent-Bernoulli Keras layer from `prod(event_shape)` params.

[`class IndependentLogistic`](../tfp/layers/IndependentLogistic.md): An independent logistic Keras layer.

[`class IndependentNormal`](../tfp/layers/IndependentNormal.md): An independent normal Keras layer.

[`class IndependentPoisson`](../tfp/layers/IndependentPoisson.md): An independent Poisson Keras layer.

[`class KLDivergenceAddLoss`](../tfp/layers/KLDivergenceAddLoss.md): Pass-through layer that adds a KL divergence penalty to the model loss.

[`class KLDivergenceRegularizer`](../tfp/layers/KLDivergenceRegularizer.md): Regularizer that adds a KL divergence penalty to the model loss.

[`class MixtureLogistic`](../tfp/layers/MixtureLogistic.md): A mixture distribution Keras layer, with independent logistic components.

[`class MixtureNormal`](../tfp/layers/MixtureNormal.md): A mixture distribution Keras layer, with independent normal components.

[`class MixtureSameFamily`](../tfp/layers/MixtureSameFamily.md): A mixture (same-family) Keras layer.

[`class MultivariateNormalTriL`](../tfp/layers/MultivariateNormalTriL.md): A `d`-variate MVNTriL Keras layer from `d + d * (d + 1) // 2` params.

[`class OneHotCategorical`](../tfp/layers/OneHotCategorical.md): A `d`-variate OneHotCategorical Keras layer from `d` params.

[`class VariableLayer`](../tfp/layers/VariableLayer.md): Simply returns a (trainable) variable, regardless of input.

[`class VariationalGaussianProcess`](../tfp/layers/VariationalGaussianProcess.md): A VariationalGaussianProcess Layer.

## Functions

[`default_loc_scale_fn(...)`](../tfp/layers/default_loc_scale_fn.md): Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

[`default_mean_field_normal_fn(...)`](../tfp/layers/default_mean_field_normal_fn.md): Creates a function to build Normal distributions with trainable params.

[`default_multivariate_normal_fn(...)`](../tfp/layers/default_multivariate_normal_fn.md): Creates multivariate standard `Normal` distribution.

