<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.minimize" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.minimize


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/math/minimize.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Minimize a loss function using a provided optimizer.

``` python
tfp.math.minimize(
    loss_fn,
    num_steps,
    optimizer,
    trainable_variables=None,
    trace_fn=_trace_loss,
    name='minimize'
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`loss_fn`</b>: Python callable with signature `loss = loss_fn()`, where `loss`
  is a `Tensor` loss to be minimized.
* <b>`num_steps`</b>: Python `int` number of steps to run the optimizer.
* <b>`optimizer`</b>: Optimizer instance to use. This may be a TF1-style
  `tf.train.Optimizer`, TF2-style `tf.optimizers.Optimizer`, or any Python
  object that implements `optimizer.apply_gradients(grads_and_vars)`.
* <b>`trainable_variables`</b>: list of `tf.Variable` instances to optimize with
  respect to. If `None`, defaults to the set of all variables accessed
  during the execution of `loss_fn()`.
  Default value: `None`.
* <b>`trace_fn`</b>: Python callable with signature `state = trace_fn(
  loss, grads, variables)`, where `state` may be a `Tensor` or nested
  structure of `Tensor`s. The state values are accumulated (by `tf.scan`)
  and returned. The default `trace_fn` simply returns the loss, but in
  general can depend on the gradients and variables (if
  `trainable_variables` is not `None` then `variables==trainable_variables`;
  otherwise it is the list of all variables accessed during execution of
  `loss_fn()`), as well as any other quantities captured in the closure of
  `trace_fn`, for example, statistics of a variational distribution.
  Default value: `lambda loss, grads, variables: loss`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: 'minimize'.


#### Returns:


* <b>`trace`</b>: `Tensor` or nested structure of `Tensor`s, according to the
  return type of `trace_fn`. Each `Tensor` has an added leading dimension
  of size `num_steps`, packing the trajectory of the result over the course
  of the optimization.

### Examples

To minimize the scalar function `(x - 5)**2`:

```python
x = tf.Variable(0.)
loss_fn = lambda: (x - 5.)**2
losses = tfp.math.minimize(loss_fn,
                           num_steps=100,
                           optimizer=tf.optimizers.Adam(learning_rate=0.1))

# In TF2/eager mode, the optimization runs immediately.
print("optimized value is {} with loss {}".format(x, losses[-1]))
```

In graph mode (e.g., inside of `tf.function` wrapping), retrieving any Tensor
that depends on the minimization op will trigger the optimization:

```python
with tf.control_dependencies([losses]):
  optimized_x = tf.identity(x)  # Use a dummy op to attach the dependency.
```

In some cases, we may want to track additional context inside the
optimization. We can do this by defining a custom `trace_fn`. Note that
the `trace_fn` is passed the loss and gradients, but it may also report the
values of trainable variables or other derived quantities by capturing them in
its closure. For example, we can capture `x` and track its value over the
optimization:

```python
# `x` is the tf.Variable instance defined above.
trace_fn = lambda loss, grads, variables: {'loss': loss, 'x': x}
trace = tfp.math.minimize(loss_fn, num_steps=100,
                        optimizer=tf.optimizers.Adam(0.1),
                        trace_fn=trace_fn)
print(trace['loss'].shape,   # => [100]
      trace['x'].shape)      # => [100]
```
