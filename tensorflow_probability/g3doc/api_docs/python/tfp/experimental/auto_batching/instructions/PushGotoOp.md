<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.PushGotoOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="push_block"/>
<meta itemprop="property" content="goto_block"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.experimental.auto_batching.instructions.PushGotoOp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `PushGotoOp`

Save an address for `IndirectGotoOp` and unconditionally jump to another.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.PushGotoOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.PushGotoOp`
* Class `tfp.experimental.auto_batching.frontend.stack.inst.PushGotoOp`
* Class `tfp.experimental.auto_batching.stack_optimization.inst.PushGotoOp`
* Class `tfp.experimental.auto_batching.stackless.inst.PushGotoOp`


<!-- Placeholder for "Used in" -->

Use in the function call sequence.  The address is saved on the
reserved "program counter" variable.

#### Args:


* <b>`push_block`</b>: The `Block` that the matching `IndirectGotoOp` will jump to.
* <b>`goto_block`</b>: The `Block` to jump to.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    push_block,
    goto_block
)
```

Create new instance of PushGotoOp(push_block, goto_block)




## Properties

<h3 id="push_block"><code>push_block</code></h3>




<h3 id="goto_block"><code>goto_block</code></h3>






