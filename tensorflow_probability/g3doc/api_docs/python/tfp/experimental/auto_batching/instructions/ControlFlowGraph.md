<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.ControlFlowGraph" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="blocks"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="block"/>
<meta itemprop="property" content="block_index"/>
<meta itemprop="property" content="enter_block"/>
<meta itemprop="property" content="exit_index"/>
</div>

# tfp.experimental.auto_batching.instructions.ControlFlowGraph


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `ControlFlowGraph`

A control flow graph (CFG).



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.ControlFlowGraph`
* Class `tfp.experimental.auto_batching.frontend.st.inst.ControlFlowGraph`
* Class `tfp.experimental.auto_batching.frontend.stack.inst.ControlFlowGraph`
* Class `tfp.experimental.auto_batching.stack_optimization.inst.ControlFlowGraph`
* Class `tfp.experimental.auto_batching.stackless.inst.ControlFlowGraph`


<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">View source</a>

``` python
__init__(blocks)
```

A control flow graph (CFG).

A CFG is a set of basic `Block`s available in a program.  In this
system, the `Block`s are ordered and indexed to support VM
instruction selection, so the CFG also keeps the reverse map from
`Block`s to their indexes.

#### Args:


* <b>`blocks`</b>: Python list of `Block` objects, the content of the CFG.
  Any terminator instructions of said `Block` objects should
  refer to other `Block`s in the same CFG.  Otherwise,
  downstream passes or staging may fail.



## Properties

<h3 id="blocks"><code>blocks</code></h3>






## Methods

<h3 id="block"><code>block</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">View source</a>

``` python
block(index)
```

Returns the `Block` given by the input `index`.


#### Args:


* <b>`index`</b>: A Python `int`.


#### Returns:


* <b>`block`</b>: The `Block` at that location in the block list.

<h3 id="block_index"><code>block_index</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">View source</a>

``` python
block_index(block)
```

Returns the `int` index of the given `Block`.


#### Args:


* <b>`block`</b>: The block to look up. If `None`, returns the exit index.


#### Returns:


* <b>`index`</b>: Python `int`, the index of the requested block.

<h3 id="enter_block"><code>enter_block</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">View source</a>

``` python
enter_block()
```

Returns the entry `Block`.


<h3 id="exit_index"><code>exit_index</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/auto_batching/instructions.py">View source</a>

``` python
exit_index()
```

Returns the `int` index denoting "exit this CFG".




