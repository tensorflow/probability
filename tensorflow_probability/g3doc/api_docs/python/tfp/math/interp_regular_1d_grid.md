<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.interp_regular_1d_grid" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.interp_regular_1d_grid

Linear `1-D` interpolation on a regular (constant spacing) grid.

``` python
tfp.math.interp_regular_1d_grid(
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



Defined in [`python/math/interpolation.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/interpolation.py).

<!-- Placeholder for "Used in" -->

Given reference values, this function computes a piecewise linear interpolant
and evaluates it on a new set of `x` values.

The interpolant is built from `C` reference values indexed by one dimension
of `y_ref` (specified by the `axis` kwarg).

If `y_ref` is a vector, then each value `y_ref[i]` is considered to be equal
to `f(x_ref[i])`, for `C` (implicitly defined) reference values between
`x_ref_min` and `x_ref_max`:

```none
x_ref[i] = x_ref_min + i * (x_ref_max - x_ref_min) / (C - 1),
i = 0, ..., C - 1.
```

If `rank(y_ref) > 1`, then dimension `axis` indexes `C` reference values of
a shape `y_ref.shape[:axis] + y_ref.shape[axis + 1:]` `Tensor`.

If `rank(x) > 1`, then the output is obtained by effectively flattening `x`,
interpolating along `axis`, then expanding the result to shape
`y_ref.shape[:axis] + x.shape + y_ref.shape[axis + 1:]`.

These shape semantics are equivalent to `scipy.interpolate.interp1d`.

#### Args:

* <b>`x`</b>: Numeric `Tensor` The x-coordinates of the interpolated output values.
* <b>`x_ref_min`</b>:  Scalar `Tensor` of same `dtype` as `x`.  The minimum value of
  the (implicitly defined) reference `x_ref`.
* <b>`x_ref_max`</b>:  Scalar `Tensor` of same `dtype` as `x`.  The maximum value of
  the (implicitly defined) reference `x_ref`.
* <b>`y_ref`</b>:  `N-D` `Tensor` (`N > 0`) of same `dtype` as `x`. The reference
  output values.
* <b>`axis`</b>:  Scalar `Tensor` designating the dimension of `y_ref` that indexes
  values of the interpolation table.
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
  Default value: `"interp_regular_1d_grid"`.


#### Returns:

* <b>`y_interp`</b>:  Interpolation between members of `y_ref`, at points `x`.
  `Tensor` of same `dtype` as `x`, and shape
  `y.shape[:axis] + x.shape + y.shape[axis + 1:]`


#### Raises:

  ValueError:  If `fill_value` is not an allowed string.
  ValueError:  If `axis` is not a scalar.

#### Examples

Interpolate a function of one variable:

```python
y_ref = tf.exp(tf.linspace(start=0., stop=10., num=200))

interp_regular_1d_grid(
    x=[6.0, 0.5, 3.3], x_ref_min=0., x_ref_max=1., y_ref=y_ref)
==> approx [exp(6.0), exp(0.5), exp(3.3)]
```

Interpolate a matrix-valued function of one variable:

```python
mat_0 = [[1., 0.], [0., 1.]]
mat_1 = [[0., -1], [1, 0]]
y_ref = [mat_0, mat_1]

# Get three output matrices at once.
tfp.math.interp_regular_1d_grid(
    x=[0., 0.5, 1.], x_ref_min=0., x_ref_max=1., y_ref=y_ref, axis=0)
==> [mat_0, 0.5 * mat_0 + 0.5 * mat_1, mat_1]
```

Interpolate a scalar valued function, and get a matrix of results:

```python
y_ref = tf.exp(tf.linspace(start=0., stop=10., num=200))
x = [[1.1, 1.2], [2.1, 2.2]]
tfp.math.interp_regular_1d_grid(x, x_ref_min=0., x_ref_max=10., y_ref=y_ref)
==> tf.exp(x)
```

Interpolate a function of one variable on a log-spaced grid:

```python
x_ref = tf.exp(tf.linspace(tf.log(1.), tf.log(100000.), num_pts))
y_ref = tf.log(x_ref + x_ref**2)

interp_regular_1d_grid(x=[1.1, 2.2], x_ref_min=1., x_ref_max=100000., y_ref,
    grid_regularizing_transform=tf.log)
==> [tf.log(1.1 + 1.1**2), tf.log(2.2 + 2.2**2)]
```