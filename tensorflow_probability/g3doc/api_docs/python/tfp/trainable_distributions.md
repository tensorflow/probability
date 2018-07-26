Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.trainable_distributions" />
<meta itemprop="property" content="__all__"/>
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: tfp.trainable_distributions

Trainable distributions.

"Trainable distributions" are instances of `tf.contrib.distributions` which are
parameterized by a transformation of a single input `Tensor`. The
transformations are presumed to use TensorFlow variables and typically need to
be fit, e.g., using `tf.train` optimizers or `tfp.optimizers`.

## Functions

[`bernoulli(...)`](../tfp/trainable_distributions/bernoulli.md): Constructs a trainable `tfd.Bernoulli` distribution.

[`multivariate_normal_tril(...)`](../tfp/trainable_distributions/multivariate_normal_tril.md): Constructs a trainable `tfd.MultivariateNormalTriL` distribution.

[`normal(...)`](../tfp/trainable_distributions/normal.md): Constructs a trainable `tfd.Normal` distribution.

[`poisson(...)`](../tfp/trainable_distributions/poisson.md): Constructs a trainable `tfd.Poisson` distribution.

[`softplus_and_shift(...)`](../tfp/trainable_distributions/softplus_and_shift.md): Converts (batch of) scalars to (batch of) positive valued scalars.

[`tril_with_diag_softplus_and_shift(...)`](../tfp/trainable_distributions/tril_with_diag_softplus_and_shift.md): Converts (batch of) vectors to (batch of) lower-triangular scale matrices.

## Other Members

`__all__`

`absolute_import`

`division`

`print_function`

