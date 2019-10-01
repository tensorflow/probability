<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.liveness" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.liveness


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/liveness.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Live variable analysis.

<!-- Placeholder for "Used in" -->

A variable is "dead" at some point if the compiler can find a proof that no
future instruction will read the value before that value is overwritten; "live"
otherwise.

This module implements a liveness analysis for the IR defined in
instructions.py.

## Functions

[`liveness_analysis(...)`](../../../tfp/experimental/auto_batching/liveness/liveness_analysis.md): Computes liveness information for each op in each block.

