<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.BranchOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="cond_var"/>
<meta itemprop="property" content="true_block"/>
<meta itemprop="property" content="false_block"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.experimental.auto_batching.instructions.BranchOp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `BranchOp`

A conditional.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.BranchOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.BranchOp`
* Class `tfp.experimental.auto_batching.frontend.stack.inst.BranchOp`
* Class `tfp.experimental.auto_batching.stack_optimization.inst.BranchOp`
* Class `tfp.experimental.auto_batching.stackless.inst.BranchOp`


<!-- Placeholder for "Used in" -->


#### Args:


* <b>`cond_var`</b>: The string name of the VM variable holding the condition.
  This variable must have boolean dtype and scalar data shape.
* <b>`true_block`</b>: The `Block` where to transfer logical threads whose
  condition is `True`.
* <b>`false_block`</b>: The `Block` where to transfer logical threads whose
  condition is `False`.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    cond_var,
    true_block,
    false_block
)
```

Create new instance of BranchOp(cond_var, true_block, false_block)




## Properties

<h3 id="cond_var"><code>cond_var</code></h3>




<h3 id="true_block"><code>true_block</code></h3>




<h3 id="false_block"><code>false_block</code></h3>






