<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.virtual_machine.execute" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.virtual_machine.execute

Executes or stages a complete auto-batching VM program.

### Aliases:

* `tfp.experimental.auto_batching.frontend.vm.execute`
* `tfp.experimental.auto_batching.virtual_machine.execute`

``` python
tfp.experimental.auto_batching.virtual_machine.execute(
    program,
    args,
    max_stack_depth,
    backend,
    block_code_cache=None
)
```



Defined in [`python/internal/auto_batching/virtual_machine.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/virtual_machine.py).

<!-- Placeholder for "Used in" -->

Whether this executes or stages computation depends on whether the backend has
an eager or deferred computation model.

The dimensions of the inputs and internal variables are split into
one top batch dimension and an arbitrary number (here `E`) event
dimensions.  The event rank may be different for different inputs,
outputs, and internal variables.

#### Args:


* <b>`program`</b>: A `instructions.Program` to execute or stage.
* <b>`args`</b>: Input values, a list of arrays, each of shape `[batch_size,
  e1, ..., eE]`.  The batch size must be the same for all inputs.
  The other dimensions must agree with the declared shapes of the
  variables they will be stored in, but need not in general be the
  same as one another.
* <b>`max_stack_depth`</b>: Python `int`. Maximum depth of stack to allocate.
* <b>`backend`</b>: Object implementing required backend operations.
* <b>`block_code_cache`</b>: Dict (allows cache to live across calls to `vm.execute`,
  or `None` (in which case a dict is created and used per call).


#### Returns:


* <b>`results`</b>: A list of the output values. Each returned value is an
  array of shape `[batch_size, e1, ..., eE]`.  The results are
  returned in the same order as the variables appear in
  `program.out_vars`.