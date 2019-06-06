<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.GotoOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="block"/>
</div>

# tfp.experimental.auto_batching.instructions.GotoOp

## Class `GotoOp`

An unconditional jump.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.GotoOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.GotoOp`
* Class `tfp.experimental.auto_batching.instructions.GotoOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

Use for skipping non-taken `if` branches, and for transferring
control during the function call sequence.

#### Args:


* <b>`block`</b>: The `Block` to jump to.

## Properties

<h3 id="block"><code>block</code></h3>






