<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="pc_var"/>
</div>

# Module: tfp.experimental.auto_batching.instructions

Instruction language for auto-batching virtual machine.

### Aliases:

* Module `tfp.experimental.auto_batching.frontend.instructions`
* Module `tfp.experimental.auto_batching.frontend.st.inst`
* Module `tfp.experimental.auto_batching.instructions`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class Block`](../../../tfp/experimental/auto_batching/instructions/Block.md): A basic block.

[`class BranchOp`](../../../tfp/experimental/auto_batching/instructions/BranchOp.md): A conditional.

[`class ControlFlowGraph`](../../../tfp/experimental/auto_batching/instructions/ControlFlowGraph.md): A control flow graph (CFG).

[`class Function`](../../../tfp/experimental/auto_batching/instructions/Function.md): A function subject to auto-batching, callable with `FunctionCallOp`.

[`class FunctionCallOp`](../../../tfp/experimental/auto_batching/instructions/FunctionCallOp.md): Call a `Function`.

[`class GotoOp`](../../../tfp/experimental/auto_batching/instructions/GotoOp.md): An unconditional jump.

[`class IndirectGotoOp`](../../../tfp/experimental/auto_batching/instructions/IndirectGotoOp.md): Jump to the address in the reserved "program counter" variable.

[`class PopOp`](../../../tfp/experimental/auto_batching/instructions/PopOp.md): Restore the given variables from their stacks.

[`class PrimOp`](../../../tfp/experimental/auto_batching/instructions/PrimOp.md): An arbitrary already-batched computation, a 'primitive operation'.

[`class Program`](../../../tfp/experimental/auto_batching/instructions/Program.md): An auto-batchable program.

[`class PushGotoOp`](../../../tfp/experimental/auto_batching/instructions/PushGotoOp.md): Save an address for `IndirectGotoOp` and unconditionally jump to another.

[`class TensorType`](../../../tfp/experimental/auto_batching/TensorType.md): TensorType(dtype, shape)

[`class Type`](../../../tfp/experimental/auto_batching/Type.md): Type(tensors,)

[`class VariableAllocation`](../../../tfp/experimental/auto_batching/instructions/VariableAllocation.md): A token indicating how to allocate memory for an autobatched variable.

## Functions

[`extract_referenced_variables(...)`](../../../tfp/experimental/auto_batching/instructions/extract_referenced_variables.md): Extracts a set of the variable names referenced by the node in question.

[`halt_op(...)`](../../../tfp/experimental/auto_batching/instructions/halt_op.md): Returns a control transfer `Op` that means "exit this graph".

[`interpret(...)`](../../../tfp/experimental/auto_batching/instructions/interpret.md): Interprets a program in this instruction language and returns the result.

[`push_op(...)`](../../../tfp/experimental/auto_batching/instructions/push_op.md): Returns an `Op` that pushes values from `vars_in` into `vars_out`.

## Other Members

* `pc_var = '__program_counter__'` <a id="pc_var"></a>
