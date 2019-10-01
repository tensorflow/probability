<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.type_inference" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.type_inference


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/type_inference.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Type inference pass on functional control flow graph.

### Aliases:

* Module `tfp.experimental.auto_batching.frontend.ab_type_inference`


<!-- Placeholder for "Used in" -->

Until converged, we propagate type information (dtype and shape) from inputs
toward outputs.

## Functions

[`infer_types(...)`](../../../tfp/experimental/auto_batching/type_inference/infer_types.md): Infers the variable types of a given program.

[`infer_types_from_signature(...)`](../../../tfp/experimental/auto_batching/type_inference/infer_types_from_signature.md): Infers the variable types of a given program.

[`is_inferring(...)`](../../../tfp/experimental/auto_batching/type_inference/is_inferring.md): Returns whether type inference is running.

[`signature(...)`](../../../tfp/experimental/auto_batching/type_inference/signature.md): Computes a type signature for the given `inputs`.

[`type_of_pattern(...)`](../../../tfp/experimental/auto_batching/type_inference/type_of_pattern.md): Returns the <a href="../../../tfp/experimental/auto_batching/Type.md"><code>instructions.Type</code></a> of `val`.

