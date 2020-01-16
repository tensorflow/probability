<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.IndirectGotoOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tfp.experimental.auto_batching.instructions.IndirectGotoOp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `IndirectGotoOp`

Jump to the address in the reserved "program counter" variable.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.IndirectGotoOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.IndirectGotoOp`
* Class `tfp.experimental.auto_batching.frontend.stack.inst.IndirectGotoOp`
* Class `tfp.experimental.auto_batching.stack_optimization.inst.IndirectGotoOp`
* Class `tfp.experimental.auto_batching.stackless.inst.IndirectGotoOp`


<!-- Placeholder for "Used in" -->

Use to return from a function call.

Also restores the previous saved program counter (under the one
jumped to), enforcing proper nesting with invocations of
`PushGotoOp`.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(_cls)
```

Create new instance of IndirectGotoOp()




