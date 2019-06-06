<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.VariableAllocation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="FULL"/>
<meta itemprop="property" content="NULL"/>
<meta itemprop="property" content="REGISTER"/>
<meta itemprop="property" content="TEMPORARY"/>
</div>

# tfp.experimental.auto_batching.instructions.VariableAllocation

## Class `VariableAllocation`

A token indicating how to allocate memory for an autobatched variable.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.VariableAllocation`
* Class `tfp.experimental.auto_batching.frontend.st.inst.VariableAllocation`
* Class `tfp.experimental.auto_batching.instructions.VariableAllocation`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

In general, a variable holding data with a longer or more complex lifetime
will need a more expensive storage strategy.

Specifically, the four variable allocation strategies are:
- `NULL`: Holds nothing.  Drops writes, raises on reads.  Useful for
  representing dummy variables that the user program never reads.
- `TEMPORARY`: Holds one value per thread, but not across basic block
  boundaries.  Only usable for temporaries that live in a single basic block,
  and thus never experience joins (or vm execution loop crossings).  For such
  a variable, `push` just overwrites the whole Tensor; `pop` nulls the whole
  Tensor out.
- `REGISTER`: Holds one value per thread, with no associated stack.  Useful
  for representing temporaries that do not cross (recursive) function calls,
  but do span multiple basic blocks.  For such a variable, `push` amounts to a
  `where`, with an optional runtime safety check for overwriting a defined
  value.
- `FULL`: Holds a complete stack for each thread.  Used as a last resort, when
  a stack is unavoidable.

The difference between `register` and `temporary` is that `register` is a
`[batch_size] + event_shape` Tensor in the loop state of the toplevel
`while_loop`, whereas `temporary` is represented as an empty tuple in the loop
state, and only holds a Tensor during the execution of the
`virtual_machine._run_block` call that uses it.  Consequently, `register`
updating involves a `where`, but writing to a `temporary` produces 0 TF ops.
Also, in the (envisioned) gather-scatter batching mode, the `temporary` Tensor
will automatically only hold data for the live threads, whereas reading and
writing a `register` will still require the gathers and scatters.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(name)
```

Initialize self.  See help(type(self)) for accurate signature.




## Class Members

* `FULL` <a id="FULL"></a>
* `NULL` <a id="NULL"></a>
* `REGISTER` <a id="REGISTER"></a>
* `TEMPORARY` <a id="TEMPORARY"></a>
