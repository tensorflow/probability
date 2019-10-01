<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.GotoOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="block"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.experimental.auto_batching.instructions.GotoOp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `GotoOp`

An unconditional jump.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.GotoOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.GotoOp`
* Class `tfp.experimental.auto_batching.frontend.stack.inst.GotoOp`
* Class `tfp.experimental.auto_batching.stack_optimization.inst.GotoOp`
* Class `tfp.experimental.auto_batching.stackless.inst.GotoOp`


<!-- Placeholder for "Used in" -->

Use for skipping non-taken `if` branches, and for transferring
control during the function call sequence.

#### Args:


* <b>`block`</b>: The `Block` to jump to.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    block
)
```

Create new instance of GotoOp(block,)




## Properties

<h3 id="block"><code>block</code></h3>






