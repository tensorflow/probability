<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.PopOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="vars"/>
</div>

# tfp.experimental.auto_batching.instructions.PopOp

## Class `PopOp`

Restore the given variables from their stacks.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.PopOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.PopOp`
* Class `tfp.experimental.auto_batching.instructions.PopOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

The current top value of each popped variable is lost.

#### Args:


* <b>`vars`</b>: list of strings: The names of the VM variables to restore.

## Properties

<h3 id="vars"><code>vars</code></h3>






