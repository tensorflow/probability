<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.batch_interp_regular_1d_grid" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.batch_interp_regular_1d_grid

``` python
tfp.math.batch_interp_regular_1d_grid(
    x,
    x_ref_min,
    x_ref_max,
    y_ref,
    axis=-1,
    fill_value='constant_extension',
    fill_value_below=None,
    fill_value_above=None,
    grid_regularizing_transform=None,
    name=None
)
```

Linear `1-D` interpolation on a regular (constant spacing) grid.

Given [batch of] reference values, this function computes a piecewise linear
interpolant and evaluates it on a [batch of] of new `x` values.

The interpolant is built from `C` reference values indexed by one dimension
of `y_ref` (specified by the `axis` kwarg).

If `y_ref` is a vector, then each value `y_ref[i]` is considered to be equal
to `f(x_ref[i])`, for `C` (implicitly defined) reference values between
`x_ref_min` and `x_ref_max`:

```none
x_ref[i] = x_ref_min + i * (x_ref_max - x_ref_min) / (C - 1),
i = 0, ..., C - 1.
```

In the general case, dimensions to the left of `axis` in `y_ref` are broadcast
with leading dimensions in `x`, `x_ref_min`, `x_ref_max`.

#### Args:

* <b>`x`</b>: Numeric `Tensor` The x-coordinates of the interpolated output values
    for each batch.  Shape broadcasts with `[A1, ..., AN, D]`, `N >= 0`.
* <b>`x_ref_min`</b>:  `Tensor` of same `dtype` as `x`.  The minimum value of the
    each batch of the (implicitly defined) reference `x_ref`.
    Shape broadcasts with `[A1, ..., AN]`, `N >= 0`.
* <b>`x_ref_max`</b>:  `Tensor` of same `dtype` as `x`.  The maximum value of the
    each batch of the (implicitly defined) reference `x_ref`.
    Shape broadcasts with `[A1, ..., AN]`, `N >= 0`.
* <b>`y_ref`</b>:  `Tensor` of same `dtype` as `x`.  The reference output values.
    `y_ref.shape[:axis]` broadcasts with the batch shape `[A1, ..., AN]`, and
    `y_ref.shape[axis:]` is `[C, B1, ..., BM]`, so the trailing dimensions
    index `C` reference values of a rank `M` `Tensor` (`M >= 0`).
* <b>`axis`</b>:  Scalar `Tensor` designating the dimension of `y_ref` that indexes
    values of the interpolation variable.
    Default value: `-1`, the rightmost axis.
* <b>`fill_value`</b>:  Determines what values output should take for `x` values that
    are below `x_ref_min` or above `x_ref_max`. `Tensor` or one of the strings
    "constant_extension" ==> Extend as constant function. "extrapolate" ==>
    Extrapolate in a linear fashion.
    Default value: `"constant_extension"`
* <b>`fill_value_below`</b>:  Optional override of `fill_value` for `x < x_ref_min`.
* <b>`fill_value_above`</b>:  Optional override of `fill_value` for `x > x_ref_max`.
* <b>`grid_regularizing_transform`</b>:  Optional transformation `g` which regularizes
    the implied spacing of the x reference points.  In other words, if
    provided, we assume `g(x_ref_i)` is a regular grid between `g(x_ref_min)`
    and `g(x_ref_max)`.
* <b>`name`</b>:  A name to prepend to created ops.
    Default value: `"batch_interp_regular_1d_grid"`.


#### Returns:

* <b>`y_interp`</b>:  Interpolation between members of `y_ref`, at points `x`.
    `Tensor` of same `dtype` as `x`, and shape `[A1, ..., AN, D, B1, ..., BM]`


#### Raises:

* <b>`ValueError`</b>:  If `fill_value` is not an allowed string.
* <b>`ValueError`</b>:  If `axis` is not a scalar.

#### Examples

Interpolate a function of one variable:

```python
y_ref = tf.exp(tf.linspace(start=0., stop=10., 20))

batch_interp_regular_1d_grid(
    x=[6.0, 0.5, 3.3], x_ref_min=0., x_ref_max=1., y_ref=y_ref)
==> approx [exp(6.0), exp(0.5), exp(3.3)]
```

Interpolate a batch of functions of one variable.

```python
# First batch member is an exponential function, second is a log.
implied_x_ref = [tf.linspace(-3., 3.2, 200), tf.linspace(0.5, 3., 200)]
y_ref = tf.stack(  # Shape [2, 200], 2 batches, 200 reference values per batch
    [tf.exp(implied_x_ref[0]), tf.log(implied_x_ref[1])], axis=0)

x = [[-1., 1., 0.],  # Shape [2, 3], 2 batches, 3 values per batch.
     [1., 2., 3.]]

y = tfp.math.batch_interp_regular_1d_grid(  # Shape [2, 3]
    x,
    x_ref_min=[-3., 0.5],
    x_ref_max=[3.2, 3.],
    y_ref=y_ref,
    axis=-1)

# y[0] approx tf.exp(x[0])
# y[1] approx tf.log(x[1])
```

Interpolate a function of one variable on a log-spaced grid:

```python
x_ref = tf.exp(tf.linspace(tf.log(1.), tf.log(100000.), num_pts))
y_ref = tf.log(x_ref + x_ref**2)

batch_interp_regular_1d_grid(x=[1.1, 2.2], x_ref_min=1., x_ref_max=100000.,
    y_ref, grid_regularizing_transform=tf.log)
==> [tf.log(1.1 + 1.1**2), tf.log(2.2 + 2.2**2)]
```