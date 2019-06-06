<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.IndirectGotoOp" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.instructions.IndirectGotoOp

## Class `IndirectGotoOp`

Jump to the address in the reserved "program counter" variable.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.IndirectGotoOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.IndirectGotoOp`
* Class `tfp.experimental.auto_batching.instructions.IndirectGotoOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

Use to return from a function call.

Also restores the previous saved program counter (under the one
jumped to), enforcing proper nesting with invocations of
`PushGotoOp`.

