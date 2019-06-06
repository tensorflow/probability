<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.allocation_strategy" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.allocation_strategy

Live variable analysis.

### Aliases:

* Module `tfp.experimental.auto_batching.allocation_strategy`
* Module `tfp.experimental.auto_batching.frontend.allocation_strategy`



Defined in [`python/internal/auto_batching/allocation_strategy.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/allocation_strategy.py).

<!-- Placeholder for "Used in" -->

A variable is "dead" at some point if the compiler can find a proof that no
future instruction will read the value before that value is overwritten; "live"
otherwise.

This module implements a liveness analysis for the IR defined in
instructions.py.

## Functions

[`optimize(...)`](../../../tfp/experimental/auto_batching/allocation_strategy/optimize.md): Optimizes a `Program`'s variable allocation strategy.

