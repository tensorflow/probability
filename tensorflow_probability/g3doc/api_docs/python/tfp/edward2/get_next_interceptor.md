<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.get_next_interceptor" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.get_next_interceptor

Yields the top-most interceptor on the thread-local interceptor stack.

``` python
tfp.edward2.get_next_interceptor(
    *args,
    **kwds
)
```

<!-- Placeholder for "Used in" -->

Operations may be intercepted by multiple nested interceptors. Once reached,
an operation can be forwarded through nested interceptors until resolved.
To allow for nesting, implement interceptors by re-wrapping their first
argument (`f`) as an `interceptable`. To avoid nesting, manipulate the
computation without using `interceptable`.

This function allows for nesting by manipulating the thread-local interceptor
stack, so that operations are intercepted in the order of interceptor nesting.

#### Examples

```python
from tensorflow_probability import edward2 as ed

def model():
  x = ed.Normal(loc=0., scale=1., name="x")
  y = ed.Normal(loc=x, scale=1., name="y")
  return x + y

def double(f, *args, **kwargs):
  return 2. * interceptable(f)(*args, **kwargs)

def set_y(f, *args, **kwargs):
  if kwargs.get("name") == "y":
    kwargs["value"] = 0.42
  return interceptable(f)(*args, **kwargs)

with interception(double):
  with interception(set_y):
    z = model()
```

This will firstly put `double` on the stack, and then `set_y`,
resulting in the stack:
(TOP) set_y -> double -> apply (BOTTOM)

The execution of `model` is then (top lines are current stack state):
1) (TOP) set_y -> double -> apply (BOTTOM);
`ed.Normal(0., 1., "x")` is intercepted by `set_y`, and as the name is not "y"
the operation is simply forwarded to the next interceptor on the stack.

2) (TOP) double -> apply (BOTTOM);
`ed.Normal(0., 1., "x")` is intercepted by `double`, to produce
`2*ed.Normal(0., 1., "x")`, with the operation being forwarded down the stack.

3) (TOP) apply (BOTTOM);
`ed.Normal(0., 1., "x")` is intercepted by `apply`, which simply calls the
constructor.

(At this point, the nested calls to `get_next_interceptor()`, produced by
forwarding operations, exit, and the current stack is again:
(TOP) set_y -> double -> apply (BOTTOM))

4) (TOP) set_y -> double -> apply (BOTTOM);
`ed.Normal(0., 1., "y")` is intercepted by `set_y`,
the value of `y` is set to 0.42 and the operation is forwarded down the stack.

5) (TOP) double -> apply (BOTTOM);
`ed.Normal(0., 1., "y")` is intercepted by `double`, to produce
`2*ed.Normal(0., 1., "y")`, with the operation being forwarded down the stack.

6) (TOP) apply (BOTTOM);
`ed.Normal(0., 1., "y")` is intercepted by `apply`, which simply calls the
constructor.

The final values for `x` and `y` inside of `model()` are tensors where `x` is
a random draw from Normal(0., 1.) doubled, and `y` is a constant 0.84, thus
z = 2 * Normal(0., 1.) + 0.84.