<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.lowering" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.auto_batching.lowering

Lowering the full IR to stack machine instructions.

### Aliases:

* Module `tfp.experimental.auto_batching.frontend.lowering`
* Module `tfp.experimental.auto_batching.lowering`



Defined in [`python/internal/auto_batching/lowering.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/lowering.py).

<!-- Placeholder for "Used in" -->

At present, only one pass is needed to make the whole instruction
language defined in instructions.py understandable by the virtual
machine defined in virtual_machine.py, namely lowering FunctionCallOp
instructions to sequences of push, pop, and goto.

## Functions

[`lower_function_calls(...)`](../../../tfp/experimental/auto_batching/lowering/lower_function_calls.md): Lowers a `Program` that may have (recursive) FunctionCallOp instructions.

