<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.BranchOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="cond_var"/>
<meta itemprop="property" content="true_block"/>
<meta itemprop="property" content="false_block"/>
</div>

# tfp.experimental.auto_batching.instructions.BranchOp

## Class `BranchOp`

A conditional.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.BranchOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.BranchOp`
* Class `tfp.experimental.auto_batching.instructions.BranchOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`cond_var`</b>: The string name of the VM variable holding the condition.
  This variable must have boolean dtype and scalar data shape.
* <b>`true_block`</b>: The `Block` where to transfer logical threads whose
  condition is `True`.
* <b>`false_block`</b>: The `Block` where to transfer logical threads whose
  condition is `False`.

## Properties

<h3 id="cond_var"><code>cond_var</code></h3>




<h3 id="true_block"><code>true_block</code></h3>




<h3 id="false_block"><code>false_block</code></h3>






