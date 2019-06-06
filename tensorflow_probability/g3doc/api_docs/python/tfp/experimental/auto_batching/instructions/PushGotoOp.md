<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.PushGotoOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="push_block"/>
<meta itemprop="property" content="goto_block"/>
</div>

# tfp.experimental.auto_batching.instructions.PushGotoOp

## Class `PushGotoOp`

Save an address for `IndirectGotoOp` and unconditionally jump to another.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.PushGotoOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.PushGotoOp`
* Class `tfp.experimental.auto_batching.instructions.PushGotoOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

Use in the function call sequence.  The address is saved on the
reserved "program counter" variable.

#### Args:


* <b>`push_block`</b>: The `Block` that the matching `IndirectGotoOp` will jump to.
* <b>`goto_block`</b>: The `Block` to jump to.

## Properties

<h3 id="push_block"><code>push_block</code></h3>




<h3 id="goto_block"><code>goto_block</code></h3>






