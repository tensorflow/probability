<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.allocation_strategy.optimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.allocation_strategy.optimize

Optimizes a `Program`'s variable allocation strategy.

### Aliases:

* `tfp.experimental.auto_batching.allocation_strategy.optimize`
* `tfp.experimental.auto_batching.frontend.allocation_strategy.optimize`

``` python
tfp.experimental.auto_batching.allocation_strategy.optimize(program)
```



Defined in [`python/internal/auto_batching/allocation_strategy.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/allocation_strategy.py).

<!-- Placeholder for "Used in" -->

The variable allocation strategies determine how much memory the `Program`
consumes, and how costly its memory access operations are (see
`instructions.VariableAllocation`).  In general, a variable holding data with
a longer or more complex lifetime will need a more expensive storage strategy.
This analysis examines variables' liveness and opportunistically selects
inexpensive sound allocation strategies.

Specifically, the algorithm is to:
- Run liveness analysis to determine the lifespan of each variable.
- Assume optimistically that no variable needs to be stored at all
  (`instructions.VariableAllocation.NULL`).
- Traverse the instructions and pattern-match conditions that require
  some storage:
  - If a variable is read by an instruction, it must be at least
    `instructions.VariableAllocation.TEMPORARY`.
  - If a variable is live out of some block (i.e., crosses a block boundary),
    it must be at least `instructions.VariableAllocation.REGISTER`.  This is
    because temporaries do not appear in the loop state in `execute`.
  - If a variable is alive across a call to an autobatched `Function`, it must
    be `instructions.VariableAllocation.FULL`, because that `Function` may
    push values to it that must not overwrite the value present at the call
    point.  (This can be improved by examining the call graph to see whether
    the callee really does push values to this variable, but that's future
    work.)

#### Args:


* <b>`program`</b>: `Program` to optimize.


#### Returns:


* <b>`program`</b>: A newly allocated `Program` with the same semantics but possibly
  different allocation strategies for some (or all) variables.  Each new
  strategy may be more efficient than the input `Program`'s allocation
  strategy for that variable (if the analysis can prove it safe), but will
  not be less efficient.