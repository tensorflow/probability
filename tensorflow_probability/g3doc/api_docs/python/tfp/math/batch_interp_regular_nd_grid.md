<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.batch_interp_regular_nd_grid" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.batch_interp_regular_nd_grid

Multi-linear interpolation on a regular (constant spacing) grid.

``` python
tfp.math.batch_interp_regular_nd_grid(
    x,
    x_ref_min,
    x_ref_max,
    y_ref,
    axis,
    fill_value='constant_extension',
    name=None
)
```



Defined in [`python/math/interpolation.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/interpolation.py).

<!-- Placeholder for "Used in" -->

Given [a batch of] reference values, this function computes a multi-linear
interpolant and evaluates it on [a batch of] of new `x` values.

The interpolant is built from reference values indexed by `nd` dimensions
of `y_ref`, starting at `axis`.

For example, take the case of a `2-D` scalar valued function and no leading
batch dimensions.  In this case, `y_ref.shape = [C1, C2]` and `y_ref[i, j]`
is the reference value corresponding to grid point

```
[x_ref_min[0] + i * (x_ref_max[0] - x_ref_min[0]) / (C1 - 1),
 x_ref_min[1] + j * (x_ref_max[1] - x_ref_min[1]) / (C2 - 1)]
```

In the general case, dimensions to the left of `axis` in `y_ref` are broadcast
with leading dimensions in `x`, `x_ref_min`, `x_ref_max`.

#### Args:

* <b>`x`</b>: Numeric `Tensor` The x-coordinates of the interpolated output values for
  each batch.  Shape `[..., D, nd]`, designating [a batch of] `D`
  coordinates in `nd` space.  `D` must be `>= 1` and is not a batch dim.
* <b>`x_ref_min`</b>:  `Tensor` of same `dtype` as `x`.  The minimum values of the
  (implicitly defined) reference `x_ref`.  Shape `[..., nd]`.
* <b>`x_ref_max`</b>:  `Tensor` of same `dtype` as `x`.  The maximum values of the
  (implicitly defined) reference `x_ref`.  Shape `[..., nd]`.
* <b>`y_ref`</b>:  `Tensor` of same `dtype` as `x`.  The reference output values. Shape
  `[..., C1, ..., Cnd, B1,...,BM]`, designating [a batch of] reference
  values indexed by `nd` dimensions, of a shape `[B1,...,BM]` valued
  function (for `M >= 0`).
* <b>`axis`</b>:  Scalar integer `Tensor`.  Dimensions `[axis, axis + nd)` of `y_ref`
  index the interpolation table.  E.g. `3-D` interpolation of a scalar
  valued function requires `axis=-3` and a `3-D` matrix valued function
  requires `axis=-5`.
* <b>`fill_value`</b>:  Determines what values output should take for `x` values that
  are below `x_ref_min` or above `x_ref_max`. Scalar `Tensor` or
  "constant_extension" ==> Extend as constant function.
  Default value: `"constant_extension"`
* <b>`name`</b>:  A name to prepend to created ops.
  Default value: `"batch_interp_regular_nd_grid"`.


#### Returns:

* <b>`y_interp`</b>:  Interpolation between members of `y_ref`, at points `x`.
  `Tensor` of same `dtype` as `x`, and shape `[..., D, B1, ..., BM].`


#### Raises:

  ValueError:  If `rank(x) < 2` is determined statically.
  ValueError:  If `axis` is not a scalar is determined statically.
  ValueError:  If `axis + nd > rank(y_ref)` is determined statically.

#### Examples

Interpolate a function of one variable.

```python
y_ref = tf.exp(tf.linspace(start=0., stop=10., 20))

tfp.math.batch_interp_regular_nd_grid(
    # x.shape = [3, 1], x_ref_min/max.shape = [1].  Trailing `1` for `1-D`.
    x=[[6.0], [0.5], [3.3]], x_ref_min=[0.], x_ref_max=[1.], y_ref=y_ref)
==> approx [exp(6.0), exp(0.5), exp(3.3)]
```

Interpolate a scalar function of two variables.

```python
x_ref_min = [0., 2 * np.pi]
x_ref_max = [0., 2 * np.pi]

# Build y_ref.
x0s, x1s = tf.meshgrid(
    tf.linspace(x_ref_min[0], x_ref_max[0], num=100),
    tf.linspace(x_ref_min[1], x_ref_max[1], num=100),
    indexing='ij')

def func(x0, x1):
  return tf.sin(x0) * tf.cos(x1)

y_ref = func(x0s, x1s)

x = np.pi * tf.random_uniform(shape=(10, 2))

tfp.math.batch_interp_regular_nd_grid(x, x_ref_min, x_ref_max, y_ref, axis=-2)
==> tf.sin(x[:, 0]) * tf.cos(x[:, 1])
```