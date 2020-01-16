<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.math" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.substrates.jax.math


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/math/__init__.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



JAX math.

<!-- Placeholder for "Used in" -->


## Modules

[`generic`](../../../../tfp/experimental/substrates/jax/math/generic.md) module: Functions for generic calculations.

[`gradient`](../../../../tfp/experimental/substrates/jax/math/gradient.md) module: Functions for computing gradients.

[`linalg`](../../../../tfp/experimental/substrates/jax/math/linalg.md) module: Functions for common linear algebra operations.

[`numeric`](../../../../tfp/experimental/substrates/jax/math/numeric.md) module: Numerically stable variants of common mathematical expressions.

## Functions

[`fill_triangular(...)`](../../../../tfp/experimental/substrates/jax/math/fill_triangular.md): Creates a (batch of) triangular matrix from a vector of inputs.

[`fill_triangular_inverse(...)`](../../../../tfp/experimental/substrates/jax/math/fill_triangular_inverse.md): Creates a vector from a (batch of) triangular matrix.

[`log1psquare(...)`](../../../../tfp/experimental/substrates/jax/math/log1psquare.md): Numerically stable calculation of `log(1 + x**2)` for small or large `|x|`.

[`log_add_exp(...)`](../../../../tfp/experimental/substrates/jax/math/log_add_exp.md): Computes `log(exp(x) + exp(y))` in a numerically stable way.

[`log_combinations(...)`](../../../../tfp/experimental/substrates/jax/math/log_combinations.md): Multinomial coefficient.

[`reduce_weighted_logsumexp(...)`](../../../../tfp/experimental/substrates/jax/math/reduce_weighted_logsumexp.md): Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

[`softplus_inverse(...)`](../../../../tfp/experimental/substrates/jax/math/softplus_inverse.md): Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

[`value_and_gradient(...)`](../../../../tfp/experimental/substrates/jax/math/value_and_gradient.md): Computes `f(*xs)` and its gradients wrt to `*xs`.

