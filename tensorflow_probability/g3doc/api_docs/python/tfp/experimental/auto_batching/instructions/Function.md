<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.instructions.Function" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__init__"/>
</div>

# tfp.experimental.auto_batching.instructions.Function

## Class `Function`

A function subject to auto-batching, callable with `FunctionCallOp`.



### Aliases:

* Class `tfp.experimental.auto_batching.frontend.instructions.Function`
* Class `tfp.experimental.auto_batching.frontend.st.inst.Function`
* Class `tfp.experimental.auto_batching.instructions.Function`



Defined in [`python/internal/auto_batching/instructions.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/instructions.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    graph,
    vars_in,
    vars_out,
    type_inference,
    name=None
)
```

A `Function` is a control flow graph with input and output variables.


#### Args:


* <b>`graph`</b>: A `ControlFlowGraph` comprising the function's body.
* <b>`vars_in`</b>: List of `string` giving the names of the formal parameters
  of the function.
* <b>`vars_out`</b>: Pattern of `string` giving the name(s) of the variables
  the function returns.  Ergo, functions must be canonicalized to
  place the return value(s) in the same-named variable(s) along
  every path to the exit.
* <b>`type_inference`</b>: A callable which takes a list of patterns of `TensorType`s
  corresponding to the data types of `vars_in`.  This callable must
  return a pattern of `TensorType`s corresponding to the structure
  assembled by the `return_vars`.
* <b>`name`</b>: Optional string denoting this `Function` in printed output.



## Properties

<h3 id="name"><code>name</code></h3>






