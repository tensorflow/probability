<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.Context" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch"/>
<meta itemprop="property" content="batch_uncurried"/>
<meta itemprop="property" content="function_names"/>
<meta itemprop="property" content="module"/>
<meta itemprop="property" content="program"/>
<meta itemprop="property" content="program_compiled"/>
<meta itemprop="property" content="program_lowered"/>
</div>

# tfp.experimental.auto_batching.Context

## Class `Context`

Context object for auto-batching multiple Python functions together.



### Aliases:

* Class `tfp.experimental.auto_batching.Context`
* Class `tfp.experimental.auto_batching.frontend.Context`



Defined in [`python/internal/auto_batching/frontend.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/frontend.py).

<!-- Placeholder for "Used in" -->

Warning: This is alpha-grade software.  The exact subset of Python that can be
successfully auto-batched is ill-specified and subject to change.
Furthermore, the errors elicited by stepping outside this subset are likely to
be confusing and misleading.  Use at your own risk.

#### Usage:


```python
ctx = frontend.Context()

@ctx.batch(type_inference=lambda ...)
def my_single_example_function_1(args):
  ...

@ctx.batch(type_inference=lambda ...)
def my_single_example_function_2(args):
  ...

# etc
```

Then calling any of the decorated functions will execute a batch computation.
The decorated functions may call each other, including mutually recursively.
See also the `batch` method.

#### Limitations:


- You must explicitly decorate every function to be auto-batched.
- All calls to them must call them by name (no higher-order auto-batching).
- Auto-batched functions must be defined with `def`, not `lambda`.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__()
```

Initializes a `Context` object.




## Methods

<h3 id="batch"><code>batch</code></h3>

``` python
batch(type_inference)
```

Decorates one function to auto-batch.

The decorated function will run in batch.  It accepts all the same
arguments, except:
- All arguments must have an additional leading dimension for the batch.
  (By special dispensation, scalar inputs are promoted to shape [1], which
  then leads to broadcasting.)
- All the arguments' sizes in the batch dimension must be the same, or 1.
  The latter are broadcast.
- The returned value will also have a leading batch dimension, and
  will have the same size.
- The batched function accepts an additional `bool` keyword argument
  `dry_run`.  If present and `True`, just calls the unbatched version,
  circumventing the auto-batching system.  This can be useful for
  debugging the program subject to auto-batching.
- The batched function accepts an additional `bool` keyword argument
  `stackless`.  If present and `True`, invokes the stackless version of the
  auto-batching system.  This can be useful for avoiding stack maintenance
  overhead; but in general, it will recover less batching, and not work in
  graph-mode TensorFlow.
- The batched function accepts an additional `int` keyword argument
  `max_stack_depth` specifying the maximum stack depth (default 15).
  Ignored in stackless execution.
- The batched function accepts an additional keyword argument `backend`
  specifying the backend to use.  Must be an instance of
  `auto_batching.TensorFlowBackend` (default) or
  `auto_batching.NumpyBackend`.
- The batched function accepts an additional keyword argument
  `block_code_cache`, a dict which allows the caching of basic block
  rewrites (i.e. `tf.function` + XLA) to live across calls to the
  autobatched function. The default value of `None` results in caching only
  within a given call to the batched function. Currently, stackless
  autobatching ignores the cache completely.

Note: All functions called by the decorated function that need auto-batching
must be decorated in the same `Context` before invoking any of them.

#### Args:


* <b>`type_inference`</b>: A Python callable giving the type signature of the
  function being auto-batched.  The callable will be invoked with a single
  argument giving the list of `instructions.Type` objects describing the
  arguments at a particular call site, and must return a list of
  `instructions.Type` objects describing the values that call site will
  return.


#### Returns:


* <b>`dec`</b>: A decorator that may be applied to a function with the given
  type signature to auto-batch it.


#### Raises:


* <b>`ValueError`</b>: If the decorated function predictably cannot be auto-batched,
  e.g., name-clashing with another function already decorated in this
  `Context`.

<h3 id="batch_uncurried"><code>batch_uncurried</code></h3>

``` python
batch_uncurried(
    function,
    type_inference
)
```

A non-decorator version of `batch`, which see.


<h3 id="function_names"><code>function_names</code></h3>

``` python
function_names()
```




<h3 id="module"><code>module</code></h3>

``` python
module()
```

Constructs an `instructions.Module` for this `Context`.


#### Returns:


* <b>`module`</b>: An `instructions.Module` representing the batched computation
  defined by all the functions decorated with `batch` in this `Context` so
  far.

<h3 id="program"><code>program</code></h3>

``` python
program(main)
```

Constructs an `instructions.Program` for this `Context`.

This is a helper method, equivalent to `self.module().program(main)`.

#### Args:


* <b>`main`</b>: Python string name of the function that should be the entry point.


#### Returns:


* <b>`prog`</b>: An `instructions.Program` representing the batched computation
  defined by all the functions decorated with `batch` in this `Context` so
  far.  Suitable for downstream compilation with other passes in
  `auto_batching`.


#### Raises:


* <b>`ValueError`</b>: If the intended `main` function was not decorated with
  `batch`.

<h3 id="program_compiled"><code>program_compiled</code></h3>

``` python
program_compiled(
    main,
    sig,
    backend
)
```

Constructs a compiled `instructions.Program` for this `Context`.

This constructs the program with `self.program(main)`, and the performs type
inference, optimization, and lowering, to emit a result that can be executed
(or staged) by the auto-batching VM.

The point of having this as a method in its own right is that it caches the
compilation on the types of the arguments.

#### Args:


* <b>`main`</b>: Python string name of the function that should be the entry point.
* <b>`sig`</b>: A `list` of (patterns of) `instructions.TensorType` aligned with
  the formal parameters to `main`.
* <b>`backend`</b>: Backend implementation.


#### Returns:


* <b>`prog`</b>: An `instructions.Program` representing the batched computation
  defined by all the functions decorated with `batch` in this `Context` so
  far.  Suitable for execution or staging on real data by the
  auto-batching VM.

<h3 id="program_lowered"><code>program_lowered</code></h3>

``` python
program_lowered(
    main,
    sig,
    backend
)
```

Constructs a lowered `instructions.Program` for this `Context`.

This constructs the program with `self.program(main)`, and the performs type
inference, optimization, and lowering, to emit a result that can be executed
(or staged) by the auto-batching VM.

The point of having this as a method in its own right is that it caches the
compilation on the types of the arguments.

#### Args:


* <b>`main`</b>: Python string name of the function that should be the entry point.
* <b>`sig`</b>: A `list` of (patterns of) `instructions.TensorType` aligned with
  the formal parameters to `main`.
* <b>`backend`</b>: Backend implementation.


#### Returns:


* <b>`prog`</b>: An `instructions.Program` representing the batched computation
  defined by all the functions decorated with `batch` in this `Context` so
  far.  Suitable for execution or staging on real data by the
  auto-batching VM.



