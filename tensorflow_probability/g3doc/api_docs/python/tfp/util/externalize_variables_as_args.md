<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.util.externalize_variables_as_args" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.util.externalize_variables_as_args

``` python
tfp.util.externalize_variables_as_args(
    fn,
    fn_args=(),
    ancestor_variables=None,
    possible_ancestor_vars=None,
    assert_variable_override=False,
    name=None
)
```

"Converts variables within a callable into explicit args. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2019-02-28.
Instructions for updating:
`externalize_variables_as_args` will not be supported with TF 2.0

Makes a new callable from `fn` which has arguments `list(fn_args) +
list(ancestor_variables)`. If `ancestor_variables` is not specified, it is
inferred by checking which of `possible_ancestor_vars` actually influences the
return value of `fn` (concretely, gradient of `fn(*fn_args)` is not `None`).
By default `possible_ancestor_vars` is `tf.trainable_variables() +
tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)`.

#### Examples

```python
num_samples = 2
num_dims = 1
dtype = np.float32

def foo(x):
  x = tf.convert_to_tensor(x, dtype=dtype, name="x")
  s = x.shape.as_list()
  y = tf.get_variable(
      name="y",
      dtype=dtype,
      initializer=np.arange(np.prod(s)).reshape(s).astype(dtype))
  return x + y

x = tf.constant(dtype([0.1, 0.2]))

wrapped_foo, discovered_ancestor_variables = (
    externalize_variables_as_args(foo, [x]))

new_x = dtype([[1.], [2.]])
new_y = dtype([[3.], [4.]])
new_result = wrapped_foo(new_x, new_y)
# ==> [[4.], [6.]]

discovered_ancestor_variables == [tf.get_variable("y", dtype)]
# ==> [True]
```

#### Args:

* <b>`fn`</b>: Python callable which returns a `Tensor` and accepts `*fn_args`.
* <b>`fn_args`</b>: Python list of args to `fn`. Represents dummy arguments passed to
    `fn` to trace its execution; actual values are unimportant. These args are
    only used to construct the output of `fn` and to resolve the ancestor
    `tf.Variable`s.
    Default value: `()` (i.e., `fn` takes no args).
* <b>`ancestor_variables`</b>: Python list of `tf.Variable`s. When `None` the list is
    expanded to non-`None` gradients of `fn(*fn_args)`. By directly providing
    the `ancestor_variables` the internal call to `fn` is avoided.
    Default value: `None` (i.e., `tf.Variable` dependencies are discovered).
* <b>`possible_ancestor_vars`</b>: Python list of possible `tf.Variable`s which might
    be a dependency of computing `fn(*fn_args)`.
    Default value: `None` (i.e., expanded as described above).
* <b>`assert_variable_override`</b>: Python `bool` indicating that not finding a
    `tf.Variable` in the override list is an exception.
    Default value: `False` (i.e., missing a `Variable` triggers a `warning`).
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., "externalize_variables_as_args").


#### Returns:

* <b>`wrapped_fn`</b>: Python callable taking arguments like
    `*(list(fn_args) + discovered_ancestor_variables)`.
* <b>`discovered_ancestor_variables`</b>: Python list of `tf.Variable`s known to be a
    dependency of `fn(*fn_args)`.


#### Raises:

* <b>`ValueError`</b>: if `assert_variable_override` is `True` and `Variable` is
    requested but not overridden.