<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.PrimOp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="vars_in"/>
<meta itemprop="property" content="vars_out"/>
<meta itemprop="property" content="function"/>
<meta itemprop="property" content="replace"/>
</div>

# tfp.experimental.auto_batching.instructions.PrimOp

## Class `PrimOp`

An arbitrary already-batched computation, a 'primitive operation'.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.PrimOp`
* Class `tfp.experimental.auto_batching.frontend.st.inst.PrimOp`
* Class `tfp.experimental.auto_batching.instructions.PrimOp`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

These are the items of work on which auto-batching is applied.  The
`function` must accept and produce Tensors with a batch dimension,
and is free to stage any (batched) computation it wants.
Restriction: the `function` must use the same computation substrate
as the VM backend.  That is, if the VM is staging to XLA, the
`function` will see XLA Tensor handles; if the VM is staging to
graph-mode TensorFlow, the `function` will see TensorFlow Tensors;
etc.

The current values of the `vars_out` are saved on their respective
stacks, and the results written to the new top.

The exact contract for `function` is as follows:
- It will be invoked with a list of positional (only) arguments,
  parallel to `vars_in`.
- Each argument will be a pattern of Tensors (meaning, either one
  Tensor or a (potentially nested) list or tuple of Tensors),
  corresponding to the `Type` of that variable.
- Each Tensor in the argument will have the `dtype` and `shape`
  given in the corresponding `TensorType`, and an additional leading
  batch dimension.
- Some indices in the batch dimension may contain junk data, if the
  corresponding threads are not executing this instruction [this is
  subject to change based on the batch execution strategy].
- The `function` must return a pattern of Tensors, or objects
  convertible to Tensors.
- The returned pattern must be compatible with the `Type`s of
  `vars_out`.
- The Tensors in the returned pattern must have `dtype` and `shape`
  compatible with the corresponding `TensorType`s of `vars_out`.
- The returned Tensors will be broadcast into their respective
  positions if necessary.  The broadcasting _includes the batch
  dimension_: Thus, a returned Tensor of insufficient rank (e.g., a
  constant) will be broadcast across batch members.  In particular,
  a Tensor that carries the indended batch size but whose sub-batch
  shape is too low rank will broadcast incorrectly, and will result
  in an error.
- If the `function` raises an exception, it will propagate and abort
  the entire computation.
- Even in the TensorFlow backend, the `function` will be staged
  several times: at least twice during type inference (to ascertain
  the shapes of the Tensors it likes to return, as a function of the
  shapes of the Tensors it is given), and exactly once during
  executable graph construction.

#### Args:


* <b>`vars_in`</b>: list of strings.  The names of the VM variables whose
  current values to pass to the `function`.
* <b>`vars_out`</b>: Pattern of strings.  The names of the VM variables
  where to save the results returned from `function`.
* <b>`function`</b>: Python callable implementing the computation.

## Properties

<h3 id="vars_in"><code>vars_in</code></h3>




<h3 id="vars_out"><code>vars_out</code></h3>




<h3 id="function"><code>function</code></h3>






## Methods

<h3 id="replace"><code>replace</code></h3>

``` python
replace(vars_out=None)
```

Return a copy of `self` with `vars_out` replaced.




