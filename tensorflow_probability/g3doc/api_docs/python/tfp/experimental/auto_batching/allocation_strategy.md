<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.allocation_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.allocation_strategy


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/allocation_strategy.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Live variable analysis.

### Aliases:

* Module `tfp.experimental.auto_batching.frontend.allocation_strategy`


<!-- Placeholder for "Used in" -->

A variable is "dead" at some point if the compiler can find a proof that no
future instruction will read the value before that value is overwritten; "live"
otherwise.

This module implements a liveness analysis for the IR defined in
instructions.py.

## Functions

[`optimize(...)`](../../../tfp/experimental/auto_batching/allocation_strategy/optimize.md): Optimizes a `Program`'s variable allocation strategy.

