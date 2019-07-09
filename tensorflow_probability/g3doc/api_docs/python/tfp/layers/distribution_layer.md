<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.distribution_layer" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.layers.distribution_layer

Layers for combining <a href="../../tfp/distributions.md"><code>tfp.distributions</code></a> and `tf.keras`.



Defined in [`python/layers/distribution_layer.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/distribution_layer.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class CategoricalMixtureOfOneHotCategorical`](../../tfp/layers/CategoricalMixtureOfOneHotCategorical.md): A OneHotCategorical mixture Keras layer from `k * (1 + d)` params.

[`class DistributionLambda`](../../tfp/layers/DistributionLambda.md): Keras layer enabling plumbing TFP distributions through Keras models.

[`class IndependentBernoulli`](../../tfp/layers/IndependentBernoulli.md): An Independent-Bernoulli Keras layer from `prod(event_shape)` params.

[`class IndependentLogistic`](../../tfp/layers/IndependentLogistic.md): An independent logistic Keras layer.

[`class IndependentNormal`](../../tfp/layers/IndependentNormal.md): An independent normal Keras layer.

[`class IndependentPoisson`](../../tfp/layers/IndependentPoisson.md): An independent Poisson Keras layer.

[`class KLDivergenceAddLoss`](../../tfp/layers/KLDivergenceAddLoss.md): Pass-through layer that adds a KL divergence penalty to the model loss.

[`class KLDivergenceRegularizer`](../../tfp/layers/KLDivergenceRegularizer.md): Regularizer that adds a KL divergence penalty to the model loss.

[`class MixtureLogistic`](../../tfp/layers/MixtureLogistic.md): A mixture distribution Keras layer, with independent logistic components.

[`class MixtureNormal`](../../tfp/layers/MixtureNormal.md): A mixture distribution Keras layer, with independent normal components.

[`class MixtureSameFamily`](../../tfp/layers/MixtureSameFamily.md): A mixture (same-family) Keras layer.

[`class MultivariateNormalTriL`](../../tfp/layers/MultivariateNormalTriL.md): A `d`-variate MVNTriL Keras layer from `d + d * (d + 1) // 2` params.

[`class OneHotCategorical`](../../tfp/layers/OneHotCategorical.md): A `d`-variate OneHotCategorical Keras layer from `d` params.

[`class VariationalGaussianProcess`](../../tfp/layers/VariationalGaussianProcess.md): A VariationalGaussianProcess Layer.

