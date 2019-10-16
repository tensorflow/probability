<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.math.generic" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.substrates.numpy.math.generic


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/numpy/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Functions for generic calculations.

<!-- Placeholder for "Used in" -->

Note: Many of these functions will eventually be migrated to core TensorFlow.

## Functions

[`log_add_exp(...)`](../../../../../tfp/experimental/substrates/numpy/math/log_add_exp.md): Computes `log(exp(x) + exp(y))` in a numerically stable way.

[`log_combinations(...)`](../../../../../tfp/experimental/substrates/numpy/math/log_combinations.md): Multinomial coefficient.

[`reduce_logmeanexp(...)`](../../../../../tfp/experimental/substrates/numpy/math/generic/reduce_logmeanexp.md): Computes `log(mean(exp(input_tensor)))`.

[`reduce_weighted_logsumexp(...)`](../../../../../tfp/experimental/substrates/numpy/math/reduce_weighted_logsumexp.md): Computes `log(abs(sum(weight * exp(elements across tensor dimensions))))`.

[`smootherstep(...)`](../../../../../tfp/experimental/substrates/numpy/math/generic/smootherstep.md): Computes a sigmoid-like interpolation function on the unit-interval.

[`soft_threshold(...)`](../../../../../tfp/experimental/substrates/numpy/math/generic/soft_threshold.md): Soft Thresholding operator.

[`softplus_inverse(...)`](../../../../../tfp/experimental/substrates/numpy/math/softplus_inverse.md): Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

