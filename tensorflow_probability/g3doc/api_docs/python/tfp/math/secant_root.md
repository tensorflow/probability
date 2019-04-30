<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.secant_root" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.secant_root

``` python
tfp.math.secant_root(
    objective_fn,
    initial_position,
    next_position=None,
    value_at_position=None,
    position_tolerance=1e-08,
    value_tolerance=1e-08,
    max_iterations=50,
    stopping_policy_fn=tf.reduce_all,
    validate_args=False,
    name=None
)
```

Finds root(s) of a function of single variable using the secant method.

The [secant method](https://en.wikipedia.org/wiki/Secant_method) is a
root-finding algorithm that uses a succession of roots of secant lines to
better approximate a root of a function. The secant method can be thought of
as a finite-difference approximation of Newton's method.

#### Args:

* <b>`objective_fn`</b>: Python callable for which roots are searched. It must be a
    callable of a single variable. `objective_fn` must return a `Tensor` of
    the same shape and dtype as `initial_position`.
* <b>`initial_position`</b>: `Tensor` or Python float representing the starting
    position. The function will search for roots in the neighborhood of each
    point. The shape of `initial_position` should match that of the input to
    `objective_fn`.
* <b>`next_position`</b>: Optional `Tensor` representing the next position in the
    search. If specified, this argument must broadcast with the shape of
    `initial_position` and have the same dtype. It will be used to compute the
    first step to take when searching for roots. If not specified, a default
    value will be used instead.
    Default value: `initial_position * (1 + 1e-4) + sign(initial_position) *
      1e-4`.
* <b>`value_at_position`</b>: Optional `Tensor` or Pyhon float representing the value
    of `objective_fn` at `initial_position`. If specified, this argument must
    have the same shape and dtype as `initial_position`. If not specified, the
    value will be evaluated during the search.
    Default value: None.
* <b>`position_tolerance`</b>: Optional `Tensor` representing the tolerance for the
    estimated roots. If specified, this argument must broadcast with the shape
    of `initial_position` and have the same dtype.
    Default value: `1e-8`.
* <b>`value_tolerance`</b>: Optional `Tensor` representing the tolerance used to check
    for roots. If the absolute value of `objective_fn` is smaller than
    `value_tolerance` at a given position, then that position is considered a
    root for the function. If specified, this argument must broadcast with the
    shape of `initial_position` and have the same dtype.
    Default value: `1e-8`.
* <b>`max_iterations`</b>: Optional `Tensor` or Python integer specifying the maximum
    number of steps to perform for each initial position. Must broadcast with
    the shape of `initial_position`.
    Default value: `50`.
* <b>`stopping_policy_fn`</b>: Python `callable` controlling the algorithm termination.
    It must be a callable accepting a `Tensor` of booleans with the shape of
    `initial_position` (each denoting whether the search is finished for each
    starting point), and returning a scalar boolean `Tensor` (indicating
    whether the overall search should stop). Typical values are
    `tf.reduce_all` (which returns only when the search is finished for all
    points), and `tf.reduce_any` (which returns as soon as the search is
    finished for any point).
    Default value: `tf.reduce_all` (returns only when the search is finished
      for all points).
* <b>`validate_args`</b>: Python `bool` indicating whether to validate arguments such
    as `position_tolerance`, `value_tolerance`, and `max_iterations`.
    Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.


#### Returns:

* <b>`root_search_results`</b>: A Python `namedtuple` containing the following items:
* <b>`estimated_root`</b>: `Tensor` containing the last position explored. If the
      search was successful within the specified tolerance, this position is
      a root of the objective function.
* <b>`objective_at_estimated_root`</b>: `Tensor` containing the value of the
      objective function at `position`. If the search was successful within
      the specified tolerance, then this is close to 0.
* <b>`num_iterations`</b>: The number of iterations performed.


#### Raises:

* <b>`ValueError`</b>: if a non-callable `stopping_policy_fn` is passed.

#### Examples

```python
import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()

# Example 1: Roots of a single function from two different starting points.

f = lambda x: (63 * x**5 - 70 * x**3 + 15 * x) / 8.
x = tf.constant([-1, 10], dtype=tf.float64)

tfp.math.secant_root(objective_fn=f, initial_position=x))
# ==> RootSearchResults(
    estimated_root=array([-0.90617985, 0.90617985]),
    objective_at_estimated_root=array([-4.81727769e-10, 7.44957651e-10]),
    num_iterations=array([ 7, 24], dtype=int32))

tfp.math.secant_root(objective_fn=f,
                     initial_position=x,
                     stopping_policy_fn=tf.reduce_any)
# ==> RootSearchResults(
    estimated_root=array([-0.90617985, 3.27379206]),
    objective_at_estimated_root=array([-4.81727769e-10, 2.66058312e+03]),
    num_iterations=array([7, 8], dtype=int32))

# Example 2: Roots of a multiplex function from a single starting point.

def f(x):
  return tf.constant([0., 63. / 8], dtype=tf.float64) * x**5 \
      + tf.constant([5. / 2, -70. / 8], dtype=tf.float64) * x**3 \
      + tf.constant([-3. / 2, 15. / 8], dtype=tf.float64) * x

x = tf.constant([-1, -1], dtype=tf.float64)

tfp.math.secant_root(objective_fn=f, initial_position=x)
# ==> RootSearchResults(
    estimated_root=array([-0.77459667, -0.90617985]),
    objective_at_estimated_root=array([-7.81339438e-11, -4.81727769e-10]),
    num_iterations=array([7, 7], dtype=int32))

# Example 3: Roots of a multiplex function from two starting points.

def f(x):
  return tf.constant([0., 63. / 8], dtype=tf.float64) * x**5 \
      + tf.constant([5. / 2, -70. / 8], dtype=tf.float64) * x**3 \
      + tf.constant([-3. / 2, 15. / 8], dtype=tf.float64) * x

x = tf.constant([[-1, -1], [10, 10]], dtype=tf.float64)

tfp.math.secant_root(objective_fn=f, initial_position=x)
# ==> RootSearchResults(
    estimated_root=array([
        [-0.77459667, -0.90617985],
        [ 0.77459667, 0.90617985]]),
    objective_at_estimated_root=array([
        [-7.81339438e-11, -4.81727769e-10],
        [6.66025013e-11, 7.44957651e-10]]),
    num_iterations=array([
        [7, 7],
        [16, 24]], dtype=int32))
```