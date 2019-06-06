<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.frontend.st" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.frontend.st

A stackless auto-batching VM.



Defined in [`python/internal/auto_batching/stackless.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/stackless.py).

<!-- Placeholder for "Used in" -->

Borrows the stack, and conditional execution, from the host Python; manages only
divergence.

## Modules

[`inst`](../../../../tfp/experimental/auto_batching/instructions.md) module: Instruction language for auto-batching virtual machine.

## Classes

[`class ExecutionQueue`](../../../../tfp/experimental/auto_batching/frontend/st/ExecutionQueue.md): A priority queue of resumption points.

## Functions

[`execute(...)`](../../../../tfp/experimental/auto_batching/frontend/st/execute.md): Executes a given program in stackless auto-batching mode.

[`is_running(...)`](../../../../tfp/experimental/auto_batching/frontend/st/is_running.md): Returns whether the stackless machine is running a computation.

