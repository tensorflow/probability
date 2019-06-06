<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.Block" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="label_str"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="assign_instructions"/>
</div>

# tfp.experimental.auto_batching.instructions.Block

## Class `Block`

A basic block.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.Block`
* Class `tfp.experimental.auto_batching.frontend.st.inst.Block`
* Class `tfp.experimental.auto_batching.instructions.Block`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    instructions=None,
    terminator=None,
    name=None
)
```

Initialize a `Block`.

A basic block is a sequence of instructions that are always executed
in order from first to last.  After the last instruction is executed,
the block transfers control as indicated by the control transfer
instruction in the `terminator` field.

Pre-lowering, `FunctionCallOp` is admissible as an internal
instruction in a `Block`, on the grounds that it returns to a fixed
place, and the block is guaranteed to continue executing.

#### Args:


* <b>`instructions`</b>: A list of `PrimOp`, `PopOp`, and `FunctionCallOp`
  instructions to execute in order.  Control transfer instructions (that
  do not return) are not permitted in this list.
* <b>`terminator`</b>: A single `BranchOp`, `GotoOp`, `PushGotoOp` or
  `IndirectGotoOp`, indicating how to transfer control out of this basic
  block.
* <b>`name`</b>: An object serving as the name of this `Block`, for display.



## Properties

<h3 id="label_str"><code>label_str</code></h3>

A string suitable for referring to this `Block` in printed output.




## Methods

<h3 id="assign_instructions"><code>assign_instructions</code></h3>

``` python
assign_instructions(instructions)
```

Assigns the body `instructions` and the `terminator` at once.

This is a convenience method, to set a `Block`'s program content
in one invocation instead of having to assign the `instructions`
and the `terminator` fields separately.

#### Args:


* <b>`instructions`</b>: A non-empty Python list of `Op` objects.  The last one must
  be a `BranchOp`, `GotoOp`, `PushGotoOp`, or `IndirectGotoOp`, and
  becomes the `terminator`.  The others, if any, must be `PrimOp`,
  `PopOp`, or `FunctionCallOp`, and become the `instructions`, in order.



