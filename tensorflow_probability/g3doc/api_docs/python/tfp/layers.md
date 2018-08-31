<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers" />
</div>

# Module: tfp.layers

Probabilistic Layers.

## Classes

[`class Convolution1DFlipout`](../tfp/layers/Convolution1DFlipout.md): 1D convolution layer (e.g. temporal convolution) with Flipout.

[`class Convolution1DReparameterization`](../tfp/layers/Convolution1DReparameterization.md): 1D convolution layer (e.g. temporal convolution).

[`class Convolution2DFlipout`](../tfp/layers/Convolution2DFlipout.md): 2D convolution layer (e.g. spatial convolution over images) with Flipout.

[`class Convolution2DReparameterization`](../tfp/layers/Convolution2DReparameterization.md): 2D convolution layer (e.g. spatial convolution over images).

[`class Convolution3DFlipout`](../tfp/layers/Convolution3DFlipout.md): 3D convolution layer (e.g. spatial convolution over volumes) with Flipout.

[`class Convolution3DReparameterization`](../tfp/layers/Convolution3DReparameterization.md): 3D convolution layer (e.g. spatial convolution over volumes).

[`class DenseFlipout`](../tfp/layers/DenseFlipout.md): Densely-connected layer class with Flipout estimator.

[`class DenseLocalReparameterization`](../tfp/layers/DenseLocalReparameterization.md): Densely-connected layer class with local reparameterization estimator.

[`class DenseReparameterization`](../tfp/layers/DenseReparameterization.md): Densely-connected layer class with reparameterization estimator.

## Functions

[`default_loc_scale_fn(...)`](../tfp/layers/default_loc_scale_fn.md): Makes closure which creates `loc`, `scale` params from `tf.get_variable`.

[`default_mean_field_normal_fn(...)`](../tfp/layers/default_mean_field_normal_fn.md): Creates a function to build Normal distributions with trainable params.

[`default_multivariate_normal_fn(...)`](../tfp/layers/default_multivariate_normal_fn.md): Creates multivariate standard `Normal` distribution.

