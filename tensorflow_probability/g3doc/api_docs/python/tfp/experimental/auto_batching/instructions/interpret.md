<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.interpret" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.auto_batching.instructions.interpret

Interprets a program in this instruction language and returns the result.

### Aliases:

* `tfp.experimental.auto_batching.frontend.instructions.interpret`
* `tfp.experimental.auto_batching.frontend.st.inst.interpret`
* `tfp.experimental.auto_batching.instructions.interpret`

``` python
tfp.experimental.auto_batching.instructions.interpret(
    program,
    *inputs
)
```



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->

This is a definitional interpreter; its purpose is to define the
semantics of the instruction language.  As such, it does no
auto-batching, and generally strives to be as simple as possible.
It also does not stage graph computations, so will only work in
Eager mode TensorFlow.

#### Args:


* <b>`program`</b>: The Program tuple to interpret.
* <b>`*inputs`</b>: Values to pass to the program.  The length of `inputs` must be
  the same as the length of `program.vars_in`.


#### Returns:


* <b>`results`</b>: A tuple of results, which are the values of the variables listed
  in `program.out_vars` at termination.


#### Raises:


* <b>`ValueError`</b>: If an internal invariant is violated, or an error is
  detected in the program being interpreted.