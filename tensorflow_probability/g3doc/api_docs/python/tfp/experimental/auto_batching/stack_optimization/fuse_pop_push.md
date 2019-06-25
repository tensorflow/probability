<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.stack_optimization.fuse_pop_push" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.stack_optimization.fuse_pop_push

Fuses pop+push sequences in the given `Program`.

### Aliases:

* `tfp.experimental.auto_batching.frontend.stack.fuse_pop_push`
* `tfp.experimental.auto_batching.stack_optimization.fuse_pop_push`

``` python
tfp.experimental.auto_batching.stack_optimization.fuse_pop_push(program)
```



Defined in [`python/internal/auto_batching/stack_optimization.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/stack_optimization.py).

<!-- Placeholder for "Used in" -->

A stack pop followed by a stack push (with no intervening read) is equivalent
to just updating the top of the stack.  The latter is more efficient for FULL
variables, because it just updates the cache for the top, and avoids gathering
from and scattering to the backing stack Tensor.

This pass mutates the `ControlFlowGraph` of the input `Program` to convert
pop+push sequences into updates.  The pass will work despite intervening
instructions that interact with other Variables, but will not cross basic
block boundaries.  As a side-effect, the pass moves non-optimized pops to the
last place in their basic block where they are still sound.  This has no
effect on the runtime behavior of the program.

#### Args:


* <b>`program`</b>: A lowered `Program` whose pop+push sequences to fuse.  `Block`s in
  the program may be mutated.


#### Returns:


* <b>`fused`</b>: A `Program` with statically redundant pop+push sequences eliminated
  in favor of `PrimOp`s with non-trivial `skip_push_mask` fields.


#### Raises:


* <b>`ValueError`</b>: If the input `Program` has not been lowered (i.e., contains
  `FunctionCallOp`), or is ill-formed (e.g., contains invalid instructions).