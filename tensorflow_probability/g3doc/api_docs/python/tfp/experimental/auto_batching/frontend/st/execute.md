<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.frontend.st.execute" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.frontend.st.execute

Executes a given program in stackless auto-batching mode.

``` python
tfp.experimental.auto_batching.frontend.st.execute(
    program,
    backend,
    block_code_cache,
    *inputs
)
```



Defined in [`python/internal/auto_batching/stackless.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/stackless.py).

<!-- Placeholder for "Used in" -->

Compare `auto_batching.virtual_machine.execute`, which executes the program in
full auto-batching mode.

#### Advantages:


- Stackless mode is much simpler than full mode.
- Stackless mode incurs much less overhead, especially in TensorFlow.

#### Disadvantages:


- Stackless mode is only compatible with TensorFlow Eager.
- Stackless mode cannot batch execution across function call boundaries.
  This is only relevant for recursive functions, and then only if
  they might converge at different stack depths.

#### Algorithm:



- Each function call is executed by its own (recursive) call to `_interpret`,
  with a current "active threads" mask.
  - This amounts to borrowing the stack from Python so we don't have to
    implement it.
  - This is why it is not possible to converge across function call
    boundaries.

- The variable environment only has registers (and temporaries and nulls); no
  expensive full variables, since we are reusing the Python stack.

- To execute one control flow graph:

  - Maintain (i) a current program counter, (ii) a current active threads
    mask, and (iii) a priority queue of waiting points, each with a mask
    giving the logical threads that are waiting to run from there.  All these
    masks should be disjoint.

  - At each time step, execute the basic block indicated by the current
    program counter, updating only the active threads.

    - Execute FunctionCallOp by recursively invoking `_interpret` (with only
      the threads that were active on entry to the FunctionCallOp).

  - If a block ends in a goto, enqueue the target, with the current active
    thread mask -- all these threads are waiting to resume there.

  - If a block ends in a branch, split the active threads according to the
    condition value and enqueue two entries, one for the true branch and one
    for the false branch.  This is how divergence happens.

  - At the end of each block (after enqueueing successors), dequeue the
    smallest program counter, and make active all the threads that were
    waiting there.  This is how re-convergence happens.

  - If the smallest remaining program counter is off the end of the graph,
    return.

- Notes: (i) To avoid infinite regress, it's important to avoid actually
  enqueueing any blocks with an empty waiting thread mask; and (ii) In order
  to actually take advantage of re-convergence, we take care to coalesce
  queued entries waiting for the same block (by computing the or of their
  masks).

This is a reimplementation in TensorFlow Eager of [1].

[1] James Bradbury and Chunli Fu, "Automatic Batching as a Compiler Pass in
PyTorch", Workshop on Systems for ML and Open Source Software at NeurIPS 2018.

#### Args:


* <b>`program`</b>: A `instructions.Program` to execute.
* <b>`backend`</b>: Object implementing required backend operations.
* <b>`block_code_cache`</b>: Dict used to enable caching of defun+XLA across multiple
  calls to `execute`. If `None` is provided, we use a new dict per call to
  `execute` which can still achieve caching across depths of the call stack.
  This caching has no real effect unless calls to
  `backend.wrap_straightline_callable` have some effect.
* <b>`*inputs`</b>: Input arrays, each of shape `[batch_size, e1, ..., eE]`.  The batch
  size must be the same for all inputs.  The other dimensions must agree
  with the declared shapes of the variables they will be stored in, but need
  not in general be the same as one another.


#### Returns:


* <b>`results`</b>: A list of the output values. Each returned value is an
  array of shape `[batch_size, e1, ..., eE]`.  The results are
  returned in the same order as the variables appear in
  `program.out_vars`.