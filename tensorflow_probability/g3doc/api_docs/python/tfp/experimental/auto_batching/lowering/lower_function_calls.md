<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.lowering.lower_function_calls" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.lowering.lower_function_calls

Lowers a `Program` that may have (recursive) FunctionCallOp instructions.

### Aliases:

* `tfp.experimental.auto_batching.frontend.lowering.lower_function_calls`
* `tfp.experimental.auto_batching.lowering.lower_function_calls`

``` python
tfp.experimental.auto_batching.lowering.lower_function_calls(program)
```



Defined in [`python/internal/auto_batching/lowering.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/lowering.py).

<!-- Placeholder for "Used in" -->

Mutates the `ControlFlowGraph` of the input program in place.  After
lowering, the result CFG

- Has no `FunctionCallOp` instructions

- Obeys a stack discipline

What is the stack discipline?  Every function body becomes a CFG
subset that:

- Never transfers control in except to the first block
  (corresponding to being called), or to a block stored with
  `PushGotoOp` (corresponding to a subroutine returning)

- Never transfers control out except with `IndirectGotoOp`
  (corresponding to returning), or with a `PushGotoOp`
  (corresponding to calling a subroutine)

- Every path through the graph has the following effect on the
  variable stacks:

  - The formal parameters receive exactly one net pop

  - The return variables receive exactly one net push

  - All other variable stacks are left as they are

  - No data is read except the top frames of the formal parameter
    stacks

Why mutate in place?  Because tying the knot in the result seemed
too hard without an explicit indirection between `Block`s and
references thereto in various `Op`s.  Specifically, when building a
new CFG to replicate the structure of an existing one, it is
necessary to allocate `Block`s to serve as the targets of all
`BranchOp`, `GotoOp` (and `FunctionCallOp`) before building those
`Op`s, and then correctly reuse those `Block`s when processing said
targets.  With an explicit indirection, however, it would have been
possible to reuse the same `Label`s, simply creating a new mapping
from them to `Block`s.

Note that the semantics assumed by this transformation is that the
CFGs being transformed do not use variable stacks internally, but
they will only be used to implement the function sequence when
function calls are lowered.  This semantics licenses placing
`PopOp`s to enforce a stack discipline for `FunctionCallOp`s.

#### Args:


* <b>`program`</b>: A `Program` whose function calls to lower.  `Block`s in
  the program may be mutated.


#### Returns:


* <b>`lowered`</b>: A `Program` that defines no `Function`s and does not use the
  `FunctionCallOp` instruction.  May share structure with the input
  `program`.


#### Raises:


* <b>`ValueError`</b>: If an invalid instruction is encountered, if a live
  variable is undefined, if different paths into a `Block` cause
  different sets of variables to be defined, or if trying to lower
  function calls in a program that already has loops (within a
  `Function` body) or `IndirectGotoOp` instructions (they confuse
  the liveness analysis).