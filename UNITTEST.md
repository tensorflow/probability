# TFP Unit-Test Best Practices

## Recipe to easily test static and dynamic shape.

tl;dr: [See this example.](https://github.com/tensorflow/probability/commit/748af84a032cde7cb256457dba59985f2f483923)

Most of TFP code has two code-paths to handle shape, one prior to graph
execution and one during. The following example illustrates a pattern for making
this easier. The "trick" is to always make the input a placeholder (even when
testing static shape).

```python
import tensorflow as tf
tfe = tf.contrib.eager

class _DistributionTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testSomething(self):
    input_ = ...  # Using `self.dtype`.
    input_ph = tf.placeholder_with_default(
        input=input_,
        shape=input_.shape if self.use_static_shape else None)
    ...
    [...] = self.evaluate([...])
    ...

class DistributionTest_StaticShape(_DistributionTest):
  dtype = np.float32
  use_static_shape = True

class DistributionTest_DynamicShape(_DistributionTest):
  dtype = np.float32
  use_static_shape = False

del _DistributionTest  # Don't run tests for the base class.
```


Notice that we use `tf.placeholder_with_default` rather than `tf.placeholder`.
This allows convenient debugging of the executing code yet still lets us
programmatically hide shape hints.

When implementing this pattern across multiple tests, or for tests that define
multiple input `Tensors`, factoring out the placeholder creation can help
avoid boilerplate and improve readability. For example,

```python
class _DistributionTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testSomething(self):
    input1 = self._build_tensor([0., 1., 2.])
    input2 = self._build_tensor(np.random.randn(5, 4))

    ...
    [...] = self.evaluate([...])
    ...

  def _build_tensor(self, ndarray, dtype=None):
    # Enforce parameterized dtype and static/dynamic testing.
    ndarray = np.asarray(ndarray).astype(
        dtype if dtype is not None else self.dtype)
    return tf.placeholder_with_default(
        input=ndarray, shape=ndarray.shape if self.use_static_shape else None)

class DistributionTest_StaticShape(_DistributionTest):
  dtype = np.float32
  use_static_shape = True

class DistributionTest_DynamicShape(_DistributionTest):
  dtype = np.float32
  use_static_shape = False

del _DistributionTest  # Don't run tests for the base class.
```

These ideas can be extended as appropriate. For example, in the [`Reshape`
bijector](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/reshape_test.py)
we tested error-checking paths that, given a static-shape input, will raise
exceptions at graph construction time, but op errors at runtime in the dynamic
case. To handle these with unified code we can have the static and dynamic
subclasses implement separate versions of assertRaisesError that do the
respectively appropriate check, i.e.,

```python
import tensorflow as tf
tfe = tf.contrib.eager

class _DistributionTest(tf.test.TestCase):

  @tfe.run_test_in_graph_and_eager_modes
  def testSomething(self):
    input_ = ...
    …
    with self.assertRaisesError(
        "Some error message"):
      [...] = self.evaluate(something_that_might_throw_exception([...]))
    ...

class DistributionTest_StaticShape(_DistributionTest):
  ...
  def assertRaisesError(self, msg):
    return self.assertRaisesRegexp(Exception, msg)

class DistributionTest_DynamicShape(_DistributionTest):
  ...
  def assertRaisesError(self, msg):
    if tf.executing_eagerly():
      return self.assertRaisesRegexp(Exception, msg)
    return self.assertRaisesOpError(msg)

del _DistributionTest  # Don't run tests for the base class.
```


## Testing ℝd-Variate Distributions

Helper class to test vector-event distributions.

[VectorDistributionTestHelpers](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distributions/python/ops/test_util.py#L193) ([Example Use](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/mixture_same_family_test.py#L91))

## Testing Discrete, Scalar Distributions

Helper class to test scalar variate distributions over integers (or Booleans).

[DiscreteScalarDistributionTestHelpers](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/internal/test_util.py#L32) ([Example Use](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/distributions/poisson_lognormal_test.py#L34))

## Testing notebooks

Some notebooks involve expensive computation, like running a bunch of
inference steps. To test them, we prefer to short-circuit the expensive parts.
This can be done using Colab's template functionality:

1. Use a templated param to control the expensive computation, e.g., the number
   of inference steps.
2. Set the param in the notebook to whatever large value actually generates the
   desired results.
3. In the BUILD file, add a test rule that uses the `template_params` argument
   to set the value to something much smaller and more tractable.

For example, in the notebook you might write

```python
# Allow the number of optimization steps to be overridden by a test harness.
num_steps = 1000  # @param { isTemplate: true}
num_steps = int(num_steps)  # Enforce correct type when overridden.

for step in range(num_steps):
  # do expensive inference step
```

and then add a BUILD test rule that looks like this:

```
colab_notebook_test(
    name = "your_colab_test",
    colab_binary = ":lite_notebook",
    default_cell_diff = "ignore",
    default_cell_timeout = "600s",
    ipynb = "YourColab.ipynb",
    tags = [
        "cuda",
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    template_params = ["num_steps=2"],  # Avoid long optimization.
)
```

The test still tests that the code executes correctly, but executes 500x faster!

An unfortunate gotcha with this approach is that `template_params` appears to
pass all its arguments as strings, so you have to manually convert them to the
correct type in the Python code. (as demonstrated above).
