<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.FunctionCallOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="function"/>
<meta itemprop="property" content="vars_in"/>
<meta itemprop="property" content="vars_out"/>
<meta itemprop="property" content="replace"/>
</div>

# tfp.experimental.auto_batching.instructions.FunctionCallOp

## Class `FunctionCallOp`

Call a `Function`.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.FunctionCallOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.FunctionCallOp`
* Class `tfp.experimental.auto_batching.instructions.FunctionCallOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

This is a higher-level instruction, what in LLVM jargon is called an
"intrinsic".  An upstream compiler may construct such instructions;
there is a pass that lowers these to sequences of instructions the
downstream VM can stage directly.

This differs from `PrimOp` in that the function being called is
itself implemented in this instruction language, and is subject to
auto-batching by the downstream VM.

A `FunctionCallOp` is required to statically know the identity of the
`Function` being called.  This is because we want to copy the return
values to their destinations at the caller side of the return
sequence.  Why do we want that?  Because threads may diverge at
function returns, thus needing to write the returned values to
different caller variables.  Doing that on the callee side would
require per-thread information about where to write the variables,
which, in this design, is encoded in the program counter stack.
Why, in turn, may threads diverge at function returns?  Because part
of the point is to allow them to converge when calling the same
function, even if from different points.

#### Args:


* <b>`function`</b>: A `Function` object describing the function to call.
  This requires all call targets to be known statically.
* <b>`vars_in`</b>: list of strings.  The names of the VM variables whose
  current values to pass to the `function`.
* <b>`vars_out`</b>: pattern of strings.  The names of the VM variables
  where to save the results returned from `function`.

## Properties

<h3 id="function"><code>function</code></h3>




<h3 id="vars_in"><code>vars_in</code></h3>




<h3 id="vars_out"><code>vars_out</code></h3>






## Methods

<h3 id="replace"><code>replace</code></h3>

``` python
replace(vars_out=None)
```

Return a copy of `self` with `vars_out` replaced.




