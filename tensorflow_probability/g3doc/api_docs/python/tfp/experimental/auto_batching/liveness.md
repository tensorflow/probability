<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.liveness" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.liveness

Live variable analysis.



Defined in [`python/internal/auto_batching/liveness.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/liveness.py).

<!-- Placeholder for "Used in" -->

A variable is "dead" at some point if the compiler can find a proof that no
future instruction will read the value before that value is overwritten; "live"
otherwise.

This module implements a liveness analysis for the IR defined in
instructions.py.

## Functions

[`liveness_analysis(...)`](../../../tfp/experimental/auto_batching/liveness/liveness_analysis.md): Computes liveness information for each op in each block.

