<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.value_and_gradient" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.value_and_gradient

Computes `f(*xs)` and its gradients wrt to `*xs`.

``` python
tfp.math.value_and_gradient(
    f,
    xs,
    use_gradient_tape=False,
    name=None
)
```



Defined in [`python/math/gradient.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/gradient.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`f`</b>: Python `callable` to be differentiated. If `f` returns a scalar, this
  scalar will be differentiated. If `f` returns a tensor or list of tensors,
  by default a scalar will be computed by adding all their values to produce
  a single scalar. If desired, the tensors can be elementwise multiplied by
  the tensors passed as the `dy` keyword argument to the returned gradient
  function.
* <b>`xs`</b>: Python list of parameters of f for which to differentiate. (Can also
  be single `Tensor`.)
* <b>`use_gradient_tape`</b>: Python `bool` indicating that `tf.GradientTape`
  should be used regardless of `tf.executing_eagerly()` status.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` (i.e., `'value_and_gradient'`).


#### Returns:

* <b>`y`</b>: `y = f(*xs)`.
* <b>`dydx`</b>: Gradient of `y` wrt each of `xs`.