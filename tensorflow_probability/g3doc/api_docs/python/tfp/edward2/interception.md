<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.interception" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.interception

Python context manager for interception.

``` python
tfp.edward2.interception(
    *args,
    **kwds
)
```

<!-- Placeholder for "Used in" -->

Upon entry, an interception context manager pushes an interceptor onto a
thread-local stack. Upon exiting, it pops the interceptor from the stack.

#### Args:

* <b>`interceptor`</b>: Function which takes a callable `f` and inputs `*args`,
  `**kwargs`.


#### Yields:

  None.

#### Examples

Interception controls the execution of Edward programs. Below we illustrate
how to set the value of a specific random variable within a program.

```python
from tensorflow_probability import edward2 as ed

def model():
  return ed.Poisson(rate=1.5, name="y")

def interceptor(f, *args, **kwargs):
  if kwargs.get("name") == "y":
    kwargs["value"] = 42
  return interceptable(f)(*args, **kwargs)

with ed.interception(interceptor):
  y = model()

with tf.Session() as sess:
  assert sess.run(y.value) == 42
```

Wrapping `f` as `interceptable` allows interceptors down the stack to
additionally modify this operation. Since the operation `f()` is not wrapped
by default, we could have called it directly. Refer also to the example in
`get_next_interceptor()` for more details on nested interceptors.