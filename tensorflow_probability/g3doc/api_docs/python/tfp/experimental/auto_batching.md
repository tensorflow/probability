<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching

TensorFlow Probability auto-batching package.



Defined in [`python/internal/auto_batching/__init__.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/__init__.py).

<!-- Placeholder for "Used in" -->


## Modules

[`allocation_strategy`](../../tfp/experimental/auto_batching/allocation_strategy.md) module: Live variable analysis.

[`dsl`](../../tfp/experimental/auto_batching/dsl.md) module: Python-embedded DSL frontend for authoring autobatchable IR programs.

[`frontend`](../../tfp/experimental/auto_batching/frontend.md) module: AutoGraph-based auto-batching frontend.

[`instructions`](../../tfp/experimental/auto_batching/instructions.md) module: Instruction language for auto-batching virtual machine.

[`liveness`](../../tfp/experimental/auto_batching/liveness.md) module: Live variable analysis.

[`lowering`](../../tfp/experimental/auto_batching/lowering.md) module: Lowering the full IR to stack machine instructions.

[`numpy_backend`](../../tfp/experimental/auto_batching/numpy_backend.md) module: Numpy backend for auto-batching VM.

[`tf_backend`](../../tfp/experimental/auto_batching/tf_backend.md) module: TensorFlow (graph) backend for auto-batching VM.

[`type_inference`](../../tfp/experimental/auto_batching/type_inference.md) module: Type inference pass on functional control flow graph.

[`virtual_machine`](../../tfp/experimental/auto_batching/virtual_machine.md) module: The auto-batching VM itself.

[`xla`](../../tfp/experimental/auto_batching/xla.md) module: XLA utilities.

## Classes

[`class Context`](../../tfp/experimental/auto_batching/Context.md): Context object for auto-batching multiple Python functions together.

[`class NumpyBackend`](../../tfp/experimental/auto_batching/NumpyBackend.md): Implements the Numpy backend ops for a PC auto-batching VM.

[`class TensorFlowBackend`](../../tfp/experimental/auto_batching/TensorFlowBackend.md): Implements the TF backend ops for a PC auto-batching VM.

[`class TensorType`](../../tfp/experimental/auto_batching/TensorType.md): TensorType(dtype, shape)

[`class Type`](../../tfp/experimental/auto_batching/Type.md): Type(tensors,)

## Functions

[`truthy(...)`](../../tfp/experimental/auto_batching/truthy.md): Normalizes Tensor ranks for use in `if` conditions.

