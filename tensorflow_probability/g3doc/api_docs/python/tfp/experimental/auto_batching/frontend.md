<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.frontend" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="TF_BACKEND"/>
</div>

# Module: tfp.experimental.auto_batching.frontend

AutoGraph-based auto-batching frontend.



Defined in [`python/internal/auto_batching/frontend.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/frontend.py).

<!-- Placeholder for "Used in" -->


## Modules

[`ab_type_inference`](../../../tfp/experimental/auto_batching/type_inference.md) module: Type inference pass on functional control flow graph.

[`allocation_strategy`](../../../tfp/experimental/auto_batching/allocation_strategy.md) module: Live variable analysis.

[`dsl`](../../../tfp/experimental/auto_batching/dsl.md) module: Python-embedded DSL frontend for authoring autobatchable IR programs.

[`instructions`](../../../tfp/experimental/auto_batching/instructions.md) module: Instruction language for auto-batching virtual machine.

[`lowering`](../../../tfp/experimental/auto_batching/lowering.md) module: Lowering the full IR to stack machine instructions.

[`st`](../../../tfp/experimental/auto_batching/frontend/st.md) module: A stackless auto-batching VM.

[`tf_backend`](../../../tfp/experimental/auto_batching/tf_backend.md) module: TensorFlow (graph) backend for auto-batching VM.

[`vm`](../../../tfp/experimental/auto_batching/virtual_machine.md) module: The auto-batching VM itself.

## Classes

[`class Context`](../../../tfp/experimental/auto_batching/Context.md): Context object for auto-batching multiple Python functions together.

## Functions

[`truthy(...)`](../../../tfp/experimental/auto_batching/truthy.md): Normalizes Tensor ranks for use in `if` conditions.

## Other Members

* `TF_BACKEND` <a id="TF_BACKEND"></a>
