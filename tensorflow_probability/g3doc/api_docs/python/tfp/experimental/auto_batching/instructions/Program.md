<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.Program" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="main_function"/>
<meta itemprop="property" content="replace"/>
</div>

# tfp.experimental.auto_batching.instructions.Program

## Class `Program`

An auto-batchable program.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.Program`
* Class `tfp.experimental.auto_batching.frontend.st.inst.Program`
* Class `tfp.experimental.auto_batching.instructions.Program`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

The primary ingredient of a Program is the control flow graph of
operations to perform.  The operation language is a union that
serves two purposes: one subset is designed to be convenient to run
in Single Instruction Multiple Thread style, and the other to
generate from an upstream Python-embedded DSL.

As such, there are operations for explicit control transfers and
stack management, as well as for interpreted function calls (pending
lowering to explicit control transfers).  The primitive computations
are encapsulated from the interpreter as Python functions.  It is
not expected that one would author programs in this operation
language directly, but rather generate them with an appropriate
compiler.

Lowering consists of eliminating `FunctionCallOp` in favor of a
specific sequence of lower-level instructions.  A few choices for
the lowered language may register as somewhat nonstandard:
- The variable name space is global; the upstream compiler is
  expected to generate unique variable names.
- There is no one stack; instead, every variable has its own stack.
  This reduces push traffic, because only the variables that are
  actually written to are ever pushed.
- Return addresses for pending lowered function calls are stored in
  a reserved variable, that is otherwise in the same environment as
  the user variables.
- There are no designated registers for function arguments or return
  values.  This is because all runtime storage is in Tensors, which
  need to have fixed types.  Instead, it is the (post-lowering)
  caller's responsibility to write the arguments into the formal
  parameters and to retrieve the returned value(s) from the
  variable(s) in which the callee left them.

The post-lowering function call sequence is
- Push the arguments to the formal parameters;
- Pop any argument variables that are no longer used;
- Store the desired return address and jump to the beginning of the function's
  body (with a single `PushGotoOp`);
- When the function returns by executing `IndirectGotoOp`, assign the
  returned values to the variables that should receive them; and
- Pop the variables holding the returned values.

Note that this sequence requires that all calls in the source
language be to statically known functions, and for every function to
leave its results in the same variable(s) on every call (regardless
of internal control flow).

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    graph,
    functions,
    var_defs,
    vars_in,
    vars_out,
    var_alloc=None
)
```

Initialize a new `Program`.


#### Args:


* <b>`graph`</b>: A `ControlFlowGraph`.  This is the graph of basic blocks
  to execute.
* <b>`functions`</b>: A list of `Function`s giving the definitions of all
  the auto-batchable functions this `Program` may (recursively)
  call.
* <b>`var_defs`</b>: A dict mapping variable names to `Type` objects
  giving their pattern of held Tensors.  Each leaf of the pattern
  is a `TensorType` object giving the dtype and shape of that leaf.
  The shape excludes the batch dimension.
* <b>`vars_in`</b>: A list of the names of the variables in which to store
  the inputs when starting.
* <b>`vars_out`</b>: A pattern of the names of the variables from which to
  read the outputs when finished.
* <b>`var_alloc`</b>: A dict mapping variable names to allocation strategies (see
  `VariableAllocation`).  The semantics of an entry are "A proof has been
  found that this strategy suffices for this variable."



## Methods

<h3 id="main_function"><code>main_function</code></h3>

``` python
main_function()
```

Return a representation of the main program as a `Function`.


<h3 id="replace"><code>replace</code></h3>

``` python
replace(
    var_defs=None,
    var_alloc=None
)
```

Return a copy of `self` with `var_defs` and/or `var_alloc` replaced.




