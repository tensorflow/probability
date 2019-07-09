<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.liveness.liveness_analysis" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.liveness.liveness_analysis

Computes liveness information for each op in each block.

``` python
tfp.experimental.auto_batching.liveness.liveness_analysis(
    graph,
    live_out
)
```



Defined in [`python/internal/auto_batching/liveness.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/liveness.py).

<!-- Placeholder for "Used in" -->

A "live" variable is one that this analysis cannot prove will not be
read later in the execution of the program:
https://en.wikipedia.org/wiki/Live_variable_analysis.

Note that the semantics assumed by this analysis is that the control
flow graph being analyzed does not use variable stacks internally,
but they will only be used to implement the function sequence when
function calls are lowered.  For this reason, a write (as from
`PrimOp` and `FunctionCallOp`) is treated as killing a variable,
even though the downstream virtual machine pushes the outputs of
`PrimOp`.  This semantics is also why `lower_function_calls` can
place `PopOp`s.

The algorithm is to traverse the blocks in the CFG in reverse order,
computing the in-block liveness map and the set of variables that
are live into the entry of each block.

- Within a block, proceed down the instructions in reverse order,
  updating the live variable set as each instruction is stepped
  over, and recording (a copy of) it.  The set of variables that are
  live out of the last instruction of a block is the union of the
  variables that are live into the blocks to which control may be
  transferred (plus anything the terminator instruction itself may
  read).

- As coded, this assumes that all control transfers go later in the
  CFG.  Supporting loops will require equation solving.

- As coded, crashes in the presence of `IndirectGotoOp`.

The latter two limitations amount to requiring this liveness
analysis only be used before lowering function calls, and that the
body of every `Function` analyzed be loop-free.  Ergo, all recursive
computations must cross `FunctionCallOp` boundaries in any program
to which this is applied.

#### Args:


* <b>`graph`</b>: The `ControlFlowGraph` on which to perform liveness analysis.
* <b>`live_out`</b>: A Python list of `str`.  The set of variables that are
  live on exit from this `graph`.


#### Returns:


* <b>`liveness_map`</b>: Python `dict` mapping each `Block` in `graph` to a
  `LivenessInfo` tuple.  Each of these has three fields:
  `live_into_block` gives the `frozenset` of `str` variable names
  live into the block; `live_out_instructions` gives a list
  parallel to the `Block`'s instructions list, of variables live
  out of that instruction; and `live_out_of_block` gives the
  `frozenset` of `str` variable names live out of the block.


#### Raises:


* <b>`ValueError`</b>: If an invalid instruction is encountered, or if trying
  to do liveness analysis in the presence of IndirectGotoOp or of
  backward control transfers.