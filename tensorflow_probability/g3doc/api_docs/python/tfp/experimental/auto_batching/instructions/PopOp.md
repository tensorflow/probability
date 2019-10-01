<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.PopOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="vars"/>
<meta itemprop="property" content="__new__"/>
</div>

# tfp.experimental.auto_batching.instructions.PopOp


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `PopOp`

Restore the given variables from their stacks.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.PopOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.PopOp`
* Class `tfp.experimental.auto_batching.frontend.stack.inst.PopOp`
* Class `tfp.experimental.auto_batching.stack_optimization.inst.PopOp`
* Class `tfp.experimental.auto_batching.stackless.inst.PopOp`


<!-- Placeholder for "Used in" -->

The current top value of each popped variable is lost.

#### Args:


* <b>`vars`</b>: list of strings: The names of the VM variables to restore.

<h2 id="__new__"><code>__new__</code></h2>

``` python
__new__(
    _cls,
    vars
)
```

Create new instance of PopOp(vars,)




## Properties

<h3 id="vars"><code>vars</code></h3>






