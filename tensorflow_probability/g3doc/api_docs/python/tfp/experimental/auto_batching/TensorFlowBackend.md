<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.TensorFlowBackend" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="variable_class"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="any"/>
<meta itemprop="property" content="assert_matching_dtype"/>
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="broadcast_to_shape_of"/>
<meta itemprop="property" content="cond"/>
<meta itemprop="property" content="create_variable"/>
<meta itemprop="property" content="equal"/>
<meta itemprop="property" content="full_mask"/>
<meta itemprop="property" content="initial_program_counter"/>
<meta itemprop="property" content="merge_dtypes"/>
<meta itemprop="property" content="merge_shapes"/>
<meta itemprop="property" content="not_equal"/>
<meta itemprop="property" content="prepare_for_cond"/>
<meta itemprop="property" content="reduce_min"/>
<meta itemprop="property" content="run_on_dummies"/>
<meta itemprop="property" content="static_value"/>
<meta itemprop="property" content="switch_case"/>
<meta itemprop="property" content="type_of"/>
<meta itemprop="property" content="where"/>
<meta itemprop="property" content="while_loop"/>
<meta itemprop="property" content="wrap_straightline_callable"/>
</div>

# tfp.experimental.auto_batching.TensorFlowBackend

## Class `TensorFlowBackend`

Implements the TF backend ops for a PC auto-batching VM.



### Aliases:

* Class `tfp.experimental.auto_batching.TensorFlowBackend`
* Class `tfp.experimental.auto_batching.frontend.tf_backend.TensorFlowBackend`
* Class `tfp.experimental.auto_batching.tf_backend.TensorFlowBackend`



Defined in [`python/internal/auto_batching/tf_backend.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/tf_backend.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    safety_checks=True,
    while_parallel_iterations=10,
    while_maximum_iterations=None,
    basic_block_xla_device=None
)
```

Construct a new backend instance.


#### Args:


* <b>`safety_checks`</b>: Python `bool` indicating whether we should use runtime
  assertions to detect stack overflow/underflow.
* <b>`while_parallel_iterations`</b>: Python `int`, the argument to pass along to
  `tf.while_loop(..., parallel_iterations=while_parallel_iterations)`
* <b>`while_maximum_iterations`</b>: Python `int` or None, the argument to pass along
  to `tf.while_loop(..., maximum_iterations=while_maximum_iterations)`
* <b>`basic_block_xla_device`</b>: Python `str` indicating the device to which basic
  blocks should be targeted (i.e. 'CPU:0' or 'GPU:0'); if not None.



## Properties

<h3 id="variable_class"><code>variable_class</code></h3>






## Methods

<h3 id="any"><code>any</code></h3>

``` python
any(
    t,
    name=None
)
```




<h3 id="assert_matching_dtype"><code>assert_matching_dtype</code></h3>

``` python
assert_matching_dtype(
    expected_dtype,
    value,
    message=''
)
```

Asserts that the dtype of `value` matches `expected_dtype`.


#### Args:


* <b>`expected_dtype`</b>: A numpy dtype
* <b>`value`</b>: `Tensor` or convertible.
* <b>`message`</b>: Optional diagnostic message.


#### Raises:


* <b>`ValueError`</b>: If dtype does not match.

<h3 id="batch_size"><code>batch_size</code></h3>

``` python
batch_size(
    value,
    name=None
)
```

Returns the first (batch) dimension of `value`.


<h3 id="broadcast_to_shape_of"><code>broadcast_to_shape_of</code></h3>

``` python
broadcast_to_shape_of(
    val,
    target,
    name=None
)
```

Broadcasts val to the shape of target.

Attempts to match the dtype of `broadcast_val` to the dtype of `target`, if
`val` is not a `Tensor` and `target` has a dtype.

#### Args:


* <b>`val`</b>: The value to be broadcast. Must be broadcast-compatible with
  `target`.
* <b>`target`</b>: `Tensor` whose shape we will broadcast `val` to match.
* <b>`name`</b>: Optional name for the op.


#### Returns:


* <b>`broadcast_val`</b>: A `Tensor` with shape matching `val + target`. Provided
  that `val`'s dimension sizes are all smaller or equal to `target`'s, the
  returned value will be the shape of `target`.

<h3 id="cond"><code>cond</code></h3>

``` python
cond(
    pred,
    true_fn,
    false_fn,
    name=None
)
```

Implements a conditional operation for the backend.


#### Args:


* <b>`pred`</b>: A boolean scalar `Tensor` indicating the condition.
* <b>`true_fn`</b>: A callable accepting and returning nests of `Tensor`s having
  the same structure as `state`, to be executed when `pred` is True.
* <b>`false_fn`</b>: A callable accepting and returning nests of `Tensor`s having
  the same structure as `state`, to be executed when `pred` is False.
* <b>`name`</b>: Optional name for the op.


#### Returns:


* <b>`state`</b>: Output state, matching nest structure of input argument `state`.

<h3 id="create_variable"><code>create_variable</code></h3>

``` python
create_variable(
    name,
    alloc,
    type_,
    max_stack_depth,
    batch_size
)
```

Returns an intialized Variable.


#### Args:


* <b>`name`</b>: Name for the variable.
* <b>`alloc`</b>: `VariableAllocation` for the variable.
* <b>`type_`</b>: `instructions.TensorType` describing the sub-batch shape and dtype
  of the variable being created.
* <b>`max_stack_depth`</b>: Scalar `int` `Tensor`, the maximum stack depth allocated.
* <b>`batch_size`</b>: Scalar `int` `Tensor`, the number of parallel threads being
  executed.


#### Returns:


* <b>`var`</b>: A new, initialized Variable object.

<h3 id="equal"><code>equal</code></h3>

``` python
equal(
    t1,
    t2,
    name=None
)
```

Implements equality comparison for TF backend.


<h3 id="full_mask"><code>full_mask</code></h3>

``` python
full_mask(
    size,
    name=None
)
```

Returns an all-True mask `Tensor` with shape `[size]`.


<h3 id="initial_program_counter"><code>initial_program_counter</code></h3>

``` python
initial_program_counter(
    size,
    dtype,
    name=None
)
```

Returns a 0-value initializer for the Program Counter variable.


#### Args:


* <b>`size`</b>: Scalar int `Tensor` specifying the number of VM threads.
* <b>`dtype`</b>: Type representing the program counter.
* <b>`name`</b>: Optional name for the op.


#### Returns:

`Tensor` zeroes with shape `[size]` and dtype `dtype`.


<h3 id="merge_dtypes"><code>merge_dtypes</code></h3>

``` python
merge_dtypes(
    dt1,
    dt2
)
```

Merges two dtypes, returning a compatible dtype.

In practice, TF implementation asserts that the two dtypes are identical.

#### Args:


* <b>`dt1`</b>: A numpy dtype, or None.
* <b>`dt2`</b>: A numpy dtype, or None.


#### Returns:


* <b>`dtype`</b>: The common numpy dtype.


#### Raises:


* <b>`ValueError`</b>: If dt1 and dt2 are not equal and both are non-`None`.

<h3 id="merge_shapes"><code>merge_shapes</code></h3>

``` python
merge_shapes(
    s1,
    s2
)
```

Merges two shapes, returning a broadcasted shape.


#### Args:


* <b>`s1`</b>: A `list` of Python `int` or None.
* <b>`s2`</b>: A `list` of Python `int` or None.


#### Returns:


* <b>`shape`</b>: A `list` of Python `int` or None.


#### Raises:


* <b>`ValueError`</b>: If `s1` and `s2` are not broadcast compatible.

<h3 id="not_equal"><code>not_equal</code></h3>

``` python
not_equal(
    t1,
    t2,
    name=None
)
```

Implements inequality comparison for TF backend.


<h3 id="prepare_for_cond"><code>prepare_for_cond</code></h3>

``` python
prepare_for_cond(state)
```

Backend hook for preparing Tensors for `cond`.

The TensorFlow backend uses this hook to apply `tf.convert_to_tensor` before
entering the cond tree generated by `virtual_machine._staged_apply`.  One
could do this inside `cond`, but when this API element was defined there
seemed to be a performance reason (for Eager mode) to do it once per cond
tree rather than once per cond.

#### Args:


* <b>`state`</b>: A state to be prepared for use in conditionals.


#### Returns:


* <b>`state`</b>: The prepared state.

<h3 id="reduce_min"><code>reduce_min</code></h3>

``` python
reduce_min(
    t,
    name=None
)
```

Implements reduce_min for TF backend.


<h3 id="run_on_dummies"><code>run_on_dummies</code></h3>

``` python
run_on_dummies(
    primitive_callable,
    input_types
)
```

Runs the given `primitive_callable` with dummy input.

This is useful for examining the outputs for the purpose of type inference.

#### Args:


* <b>`primitive_callable`</b>: A python callable.
* <b>`input_types`</b>: `list` of `instructions.Type` type of each argument to the
  callable.  Note that the contained `TensorType` objects must match the
  dimensions with which the primitive is to be invoked at runtime, even
  though type inference conventionally does not store the batch dimension
  in the `TensorType`s.


#### Returns:


* <b>`outputs`</b>: pattern of backend-specific objects whose types may be
  analyzed by the caller with `type_of`.

<h3 id="static_value"><code>static_value</code></h3>

``` python
static_value(t)
```

Gets the eager/immediate value of `t`, or `None` if `t` is a Tensor.


<h3 id="switch_case"><code>switch_case</code></h3>

``` python
switch_case(
    branch_selector,
    branch_callables,
    name=None
)
```

Implements a switch (branch_selector) { case ... } construct.


<h3 id="type_of"><code>type_of</code></h3>

``` python
type_of(
    t,
    preferred_dtype=None
)
```

Returns the `instructions.Type` of `t`.


#### Args:


* <b>`t`</b>: `tf.Tensor` or a Python or numpy constant.
* <b>`preferred_dtype`</b>: dtype to prefer, if `t` is a constant.


#### Returns:


* <b>`vm_type`</b>: `instructions.TensorType` describing `t`.

<h3 id="where"><code>where</code></h3>

``` python
where(
    condition,
    x,
    y,
    name=None
)
```

Implements a where selector for the TF backend.

Attempts to match the dtypes of the value operands, if they are not yet both
`Tensor`s.

#### Args:


* <b>`condition`</b>: A boolean `Tensor`, either a vector having length
  `(x + y).shape[0]` or matching the full shape of `x + y`.
* <b>`x`</b>: `Tensor` of values to take when `condition` is `True`. Shape must match
  that of `y`.
* <b>`y`</b>: `Tensor` of values to take when `condition` is `False`. Shape must
  match that of `x`.
* <b>`name`</b>: Optional name for the op.


#### Returns:


* <b>`masked`</b>: A broadcast-shaped `Tensor` where elements corresponding to `True`
  values of `condition` come from `x`, and others come from `y`.

<h3 id="while_loop"><code>while_loop</code></h3>

``` python
while_loop(
    cond,
    body,
    loop_vars,
    name=None
)
```

Implements while loops for TF backend.


<h3 id="wrap_straightline_callable"><code>wrap_straightline_callable</code></h3>

``` python
wrap_straightline_callable(f)
```

Method exists solely to be stubbed, i.e. for defun + XLA compile.




