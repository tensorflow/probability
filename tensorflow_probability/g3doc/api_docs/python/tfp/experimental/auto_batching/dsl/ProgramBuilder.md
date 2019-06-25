<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.auto_batching.dsl.ProgramBuilder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="var"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="const"/>
<meta itemprop="property" content="declare_function"/>
<meta itemprop="property" content="define_function"/>
<meta itemprop="property" content="else_"/>
<meta itemprop="property" content="function"/>
<meta itemprop="property" content="if_"/>
<meta itemprop="property" content="local"/>
<meta itemprop="property" content="locals_"/>
<meta itemprop="property" content="module"/>
<meta itemprop="property" content="param"/>
<meta itemprop="property" content="primop"/>
<meta itemprop="property" content="program"/>
<meta itemprop="property" content="return_"/>
</div>

# tfp.experimental.auto_batching.dsl.ProgramBuilder

## Class `ProgramBuilder`

An auto-batching DSL context.



### Aliases:

* Class `tfp.experimental.auto_batching.dsl.ProgramBuilder`
* Class `tfp.experimental.auto_batching.frontend.dsl.ProgramBuilder`



Defined in [`python/internal/auto_batching/dsl.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/internal/auto_batching/dsl.py).

<!-- Placeholder for "Used in" -->

Auto-batching DSL operations are methods on the `ProgramBuilder` object.  It's
used like this:

```python
ab = dsl.ProgramBuilder()

def fib_type(arg_types):
  return arg_types[0]

with ab.function(type_inference=fib_type) as fibonacci:
  n = ab.param('n')
  ab.var.cond = ab.primop(lambda n: n > 1)
  with ab.if_(ab.var.cond):
    ab.var.nm1 = ab.primop(lambda n: n - 1)
    ab.var.fibm1 = ab.call(fibonacci, [ab.var.nm1])
    ab.var.nm2 = ab.primop(lambda n: n - 2)
    ab.var.fibm2 = ab.call(fibonacci, [ab.var.nm2])
    ab.var.ans = ab.primop(lambda fibm1, fibm2: fibm1 + fibm2)
  with ab.else_():
    ab.var.ans = ab.const(1)
  ab.return_(ab.var.ans)

prog = ab.program(main=fibonacci)
# Now `prog` is a well-formed `instructions.Program`, and the context
# `ab` is no longer needed.
```

Note the sequence of method calls on `ProgramBuilder` corresponds to the
source code of the `Program` being defined, not its runtime behavior.  This is
because (a) functions are defined with a context manager (rather than a Python
function) which executes its body immediately and exactly once; and (b)
function call instructions (and primitive operations) are just recorded, not
entered recursively.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__()
```

Creates an empty ProgramBuilder.




## Properties

<h3 id="var"><code>var</code></h3>

Auto-batching variables visible in the current scope.

Overrides `setattr` and `getattr` to provide a smooth interface
to reading and defining variables:

- `ProgramBuilder.var.foo = ProgramBuilder.{call,primop}`
  records an assignment to the auto-batched variable `foo`, possibly
  binding it, and

- `ProgramBuilder.var.foo` reads from the auto-batched variable `foo`
  (if it is bound).

#### Example:



```python
ab = dsl.ProgramBuilder()

ab.var.seven = ab.const(7)
```

#### Returns:


* <b>`vars`</b>: A `_MagicVars` instance representing the local scope as above.



## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(pattern)
```

Prepares a multi-value return.


#### Example:


```python
ab = dsl.ProgramBuilder()

ab((ab.var.two, ab.var.four)).pattern = ab.const((2, 4))
```

The protocol is to create a magic pattern object by invoking the
`ProgramBuilder` as a callable, passing the pattern to bind; then assigning
the `pattern` attribute of the returned value to the operation whose values
to accept.

This is like this to work around limitations of embedding a DSL into Python:
the assignment syntax `=` can be overridden _only_ for fields of objects,
not for function calls.  It would have been nicer to implement
`ab.pattern(...) = ...` but that's syntactically invalid Python.  Hence,
putting the `pattern` token at the end of the phrase rather than the
beginning.

#### Args:


* <b>`pattern`</b>: A pattern of variables (e.g., from `ab.var.name`) to bind.


#### Returns:


* <b>`pat_object`</b>: A `_MagicPattern` instance representing the putative binding.
  Invoke the `pattern =` attribute setter on that instance to actually
  bind this pattern as the output of a `primop`, `const`, or `call`.

<h3 id="call"><code>call</code></h3>

``` python
call(
    function,
    vars_in,
    vars_out=None
)
```

Registers a function call instruction.


#### Example:


```
ab = dsl.ProgramBuilder()

# Define a function
with ab.function(...) as func:
  ...
  # Call it (recursively)
  ab.var.thing = ab.call(func, ...)
  ...
```

#### Args:


* <b>`function`</b>: The `instructions.Function` object representing the function to
  call.
* <b>`vars_in`</b>: Python strings giving the variables to pass in as inputs.
* <b>`vars_out`</b>: A pattern of Python strings, giving the auto-batched variable(s)
  to which to write the result of the call.  Defaults to the empty list.


#### Raises:


* <b>`ValueError`</b>: If the call references undefined auto-batched variables.


#### Returns:


* <b>`op`</b>: An `instructions.FunctionCallOp` representing the call.  If one
  subsequently assigns this to a local, via `ProgramBuilder.var.foo = op`,
  that local gets added to the list of output variables.

<h3 id="const"><code>const</code></h3>

``` python
const(
    value,
    vars_out=None
)
```

Records a constant or set of constants.

Like `primop`, the output variables can be specified explicitly via the
`vars_out` argument or implicitly by assigning the return value to
some `ProgramBuilder.var.foo`.

#### Args:


* <b>`value`</b>: A Python list of the constants to record.
* <b>`vars_out`</b>: A pattern of Python strings, giving the auto-batched variable(s)
  to which to write the result of the callable.  Defaults to the empty
  list.


#### Returns:


* <b>`op`</b>: An `instructions.PrimOp` instance representing this operation.  If one
  subsequently assigns this to a local, via `ProgramBuilder.var.foo = op`,
  that local gets added to the list of output variables.

<h3 id="declare_function"><code>declare_function</code></h3>

``` python
declare_function(
    name=None,
    type_inference=None
)
```

Forward-declares a function to be defined later with `define_function`.

This useful for defining mutually recursive functions:
```python
ab = dsl.ProgramBuilder()

foo = ab.declare_function(...)

with ab.function(...) as bar:
  ...
  ab.call(foo)

with ab.define_function(foo):
  ...
  ab.call(bar)
```

It is an error to call but never define a declared function.

#### Args:


* <b>`name`</b>: Optional string naming this function when the program is printed.
* <b>`type_inference`</b>: A Python callable giving the type signature of the
  function being defined.  See `function`.


#### Returns:


* <b>`function`</b>: An `instructions.Function` object representing the function
  being declared.  It can be passed to `call` to call it, and to
  `define_function` to define it.

<h3 id="define_function"><code>define_function</code></h3>

``` python
define_function(
    *args,
    **kwds
)
```

Registers a definition for a previously declared function.

Usually, one would use the `function` method to declare and define a
function at the same time.  Explicit use of `define_function` is only useful
for mutual recursion or controlling code order separately from the call
graph.

#### Example:


```python
ab = dsl.ProgramBuilder()

foo = ab.declare_function(...)

with ab.function(...) as bar:
  ...
  ab.call(foo)

with ab.define_function(foo):
  ...
  ab.call(bar)
```

Function bodies appear in the compiled `instructions.Program` in order of
definition, not declaration.

#### Note:


- The formal parameters are given by calling `ab.param` inside the
  `with` block.
- The body of the `with` block registers the body of the function being
  defined.
- The last statement registered in the `with` block must be a `ab.return_`,
  or the `Function` will be malformed.

#### Args:


* <b>`function`</b>: The function (from `declare_function`) to define.


#### Yields:


* <b>`function`</b>: The `function` being defined, by symmetry with the
  `context.function` method.


#### Raises:


* <b>`ValueError`</b>: If invoked while defining a function, if the `function`
  argument has already been defined, or if the function definition does
  not end in a `return_`.

<h3 id="else_"><code>else_</code></h3>

``` python
else_(
    *args,
    **kwds
)
```

Records the `false` branch of a conditional operation.

The `true` branch must be recorded (by `if_`, above) as the immediately
preceding operation at the same nesting depth.

#### Example:


```python
ab = dsl.ProgramBuilder()

ab.var.false = ab.const(False)
with ab.if_(ab.var.false):
  ...
with ab.else_():
  ...  # The body of the `with` statement gives the `false` branch
```

#### Args:


* <b>`else_name`</b>: Optional Python string naming the false branch when the program
  is printed.  Overrides the `else_name`, if any, given in the
  corresponding `if_`.
* <b>`continue_name`</b>: Optional Python string naming the continuation after the if
  when the program is printed.  Overrides the `continue_name`, if any,
  given in the corresponding `if_`.


#### Raises:


* <b>`ValueError`</b>: If not immediately preceded by an `if_`.


#### Yields:

Nothing.


<h3 id="function"><code>function</code></h3>

``` python
function(
    *args,
    **kwds
)
```

Registers a definition of an auto-batchable function.


#### Example:


```python
ab = dsl.ProgramBuilder()

with ab.function(...) as f:
  ab.param('n')
  ...
  ab.return_(...)
```

#### Note:


- The `as` clause (here `f`) binds an `instructions.Function` object
  representing the function being defined (see Yields).
- The formal parameters are given by calling `param` inside the
  `with` block.
- The body of the `with` block registers the body of the function being
  defined.
- The last statement registered in the `with` block must be a call to
  `return_`, or the `Function` will be malformed.

The `function` method is a shorthand of `declare_function` followed by
`define_function`.  The example is equivalent to:

```python
ab = dsl.ProgramBuilder()

f = ab.declare_function(...)
with ab.define_function(f):
  ab.param('n')
  ...
  ab.return_(...)
```

#### Args:


* <b>`name`</b>: Optional string naming this function when the program is printed.
* <b>`type_inference`</b>: A Python callable giving the type signature of the
  function being defined.  The callable will be invoked with a single
  argument giving the list of `instruction.Type` objects describing
  the arguments at a particular call site, and must return a list
  of `instruction.Type` objects describing the values that call site
  will return.


#### Raises:


* <b>`ValueError`</b>: If invoked while defining a function, or if the function
  definition does not end in a `return_`.


#### Yields:


* <b>`function`</b>: An `instructions.Function` object representing the function
  being defined.  It can be passed to `call` to call it (including
  recursively).  Note that Python scopes `as` bindings to the definition
  enclosing the `with`, so a function thus bound can be referred to after
  its body as well.

<h3 id="if_"><code>if_</code></h3>

``` python
if_(
    *args,
    **kwds
)
```

Records a conditional operation and `true` first branch.

The `false` branch, if present, must be guarded by a call to `else_`, below.

#### Example:


```python
ab = dsl.ProgramBuilder()

ab.var.true = ab.const(True)
with ab.if_(ab.var.true):
  ...  # The body of the `with` statement gives the `true` branch
with ab.else_():  # The else_ clause is optional
  ...
```

#### Args:


* <b>`condition`</b>: Python string giving the boolean variable that holds the branch
  condition.
* <b>`then_name`</b>: Optional Python string naming the true branch when the program
  is printed.
* <b>`else_name`</b>: Optional Python string naming the false branch when the program
  is printed.
* <b>`continue_name`</b>: Optional Python string naming the continuation after the if
  when the program is printed.


#### Yields:

Nothing.



#### Raises:


* <b>`ValueError`</b>: If trying to condition on a variable that has not been
  written to.

<h3 id="local"><code>local</code></h3>

``` python
local(
    name=None,
    define=True
)
```

Declares a local variable in the current scope.

This should typically not be needed, because `ProgramBuilder.var.foo =` can
bind variables; however, may be helpful for a multivalue return (see
`primop` or `call`).

#### Args:


* <b>`name`</b>: Optional Python string to serve a mnemonic name in later compiler
  stages.  Variable names are automatically uniqued.  This variable can
  later be referred to with `ProgramBuilder.var.name`, as well as through
  any Python binding of the returned value.
* <b>`define`</b>: Boolean giving whether to mark this variable defined on creation.
  Default `True`.  Setting `False` is useful for speculatively uniquing a
  variable on its first appearance, before knowning whether said
  appearance is a write (in which case the variable becomes defined) or a
  read (which raises an error).


#### Returns:


* <b>`var`</b>: A Python string representing this variable.  Suitable for passing
  to `primop`, `call`, `if_`, and `return_`.

<h3 id="locals_"><code>locals_</code></h3>

``` python
locals_(
    count,
    name=None
)
```

Declares several variables at once.

This is a convenience method standing for several invocations of `local`.

#### Args:


* <b>`count`</b>: Python int.  The number of distinct variables to return.
* <b>`name`</b>: Optional Python string to serve a mnemonic name in later compiler
  stages.  Variable names are automatically uniqued.


#### Returns:


* <b>`vars`</b>: A list of `count` Python strings representing these variables.
  Suitable for passing to `primop`, `call`, `if_`, and `return_`.

<h3 id="module"><code>module</code></h3>

``` python
module()
```

Returns the registered function definitions as an `instructions.Module`.


#### Example:


```python
ab = dsl.ProgramBuilder()

with ab.function(...) as foo:
  ...  # Do stuff

module = ab.module()
```

#### Raises:


* <b>`ValueError`</b>: If invoked inside a function definition.


#### Returns:


* <b>`module`</b>: The `instructions.Module` corresponding to all the definitions
  accumulated in this `ProgramBuilder`.

<h3 id="param"><code>param</code></h3>

``` python
param(name=None)
```

Declares a parameter of the function currently being defined.

This make a local variable like `local`, but also makes it an input of the
nearest enclosing function (created by `with ProgramBuilder.function()`).
This is a separate method from `function` because the DSL wants to create
Python bindings for the function name itself and all of its input
parameters, and there is no way to convince the `with` syntax to do that.

#### Args:


* <b>`name`</b>: Optional Python string to serve a mnemonic name in later compiler
  stages.  Variable names are automatically uniqued.


#### Returns:


* <b>`var`</b>: A Python string representing this variable.  Suitable for passing
  to `primop`, `call`, `if_`, and `return_`.

<h3 id="primop"><code>primop</code></h3>

``` python
primop(
    f,
    vars_in=None,
    vars_out=None
)
```

Records a primitive operation.


#### Example:



```
ab = dsl.ProgramBuilder()

ab.var.five = ab.const(5)
# Implicit output binding
ab.var.ten = ab.primop(lambda five: five + five)
# Explicit output binding
ab.primop(lambda: (5, 10), vars_out=[ab.var.five, ab.var.ten])
```

#### Args:


* <b>`f`</b>: A Python callable, the primitive operation to perform.  Can be
  an inline lambda expression in simple cases.  Must return a list or
  tuple of results, one for each intended output variable.
* <b>`vars_in`</b>: A list of Python strings, giving the auto-batched variables
  to pass into the callable when invoking it.  If absent, `primop`
  will try to infer it by inspecting the argument list of the callable
  and matching against variables bound in the local scope.
* <b>`vars_out`</b>: A pattern of Python strings, giving the auto-batched variable(s)
  to which to write the result of the callable.  Defaults to the empty
  list.


#### Raises:


* <b>`ValueError`</b>: If the definition is invalid, if the primop references
  undefined auto-batched variables, or if auto-detection of input
  variables fails.


#### Returns:


* <b>`op`</b>: An `instructions.PrimOp` instance representing this operation.  If one
  subsequently assigns this to a local, via `ProgramBuilder.var.foo = op`,
  that local becomes the output pattern.

<h3 id="program"><code>program</code></h3>

``` python
program(main)
```

Returns the registered program as an `instructions.Program`.

This is a helper method, equivalent to `self.module().program(main)`.

#### Example:


```python
ab = dsl.ProgramBuilder()

with ab.function(...) as main:
  ...  # Do the stuff

program = ab.program(main)
```

#### Args:


* <b>`main`</b>: An `instructions.Function` object representing the main entry point.


#### Raises:


* <b>`ValueError`</b>: If invoked inside a function definition, of if the intended
  `main` function was not defined.


#### Returns:


* <b>`program`</b>: The `instructions.Program` corresponding to all the definitions
  accumulated in this `ProgramBuilder`.

<h3 id="return_"><code>return_</code></h3>

``` python
return_(vars_out)
```

Records a function return instruction.


#### Example:


```python
ab = dsl.ProgramBuilder()

with ab.function(...) as f:
  ...
  ab.var.result = ...
  ab.return_(ab.var.result)
```

A `return_` command must occur at the top level of the function definition
(not inside any `if_`s), and must be the last statement therein.  You can
always achieve this by assigning to a dedicated variable for the answer
where you would otherwise return (and massaging your control flow).

#### Args:


* <b>`vars_out`</b>: Pattern of Python strings giving the auto-batched variables to
  return.


#### Raises:


* <b>`ValueError`</b>: If invoked more than once in a function body, or if trying to
  return variables that have not been written to.



