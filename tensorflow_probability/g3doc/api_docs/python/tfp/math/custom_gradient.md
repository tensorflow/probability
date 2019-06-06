<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.custom_gradient" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.custom_gradient

Embeds a custom gradient into a `Tensor`.

``` python
tfp.math.custom_gradient(
    fx,
    gx,
    x,
    fx_gx_manually_stopped=False,
    name=None
)
```



Defined in [`python/math/custom_gradient.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/custom_gradient.py).

<!-- Placeholder for "Used in" -->

This function works by clever application of `stop_gradient`. I.e., observe
that:

```none
h(x) = stop_gradient(f(x)) + stop_gradient(g(x)) * (x - stop_gradient(x))
```

is such that `h(x) == stop_gradient(f(x))` and
`grad[h(x), x] == stop_gradient(g(x)).`

In addition to scalar-domain/scalar-range functions, this function also
supports tensor-domain/scalar-range functions.

Partial Custom Gradient:

Suppose `h(x) = htilde(x, y)`. Note that `dh/dx = stop(g(x))` but `dh/dy =
None`. This is because a `Tensor` cannot have only a portion of its gradient
stopped. To circumvent this issue, one must manually `stop_gradient` the
relevant portions of `f`, `g`. For example see the unit-test,
`test_works_correctly_fx_gx_manually_stopped`.

#### Args:


* <b>`fx`</b>: `Tensor`. Output of function evaluated at `x`.
* <b>`gx`</b>: `Tensor` or list of `Tensor`s. Gradient of function at (each) `x`.
* <b>`x`</b>: `Tensor` or list of `Tensor`s. Args of evaluation for `f`.
* <b>`fx_gx_manually_stopped`</b>: Python `bool` indicating that `fx`, `gx` manually
  have `stop_gradient` applied.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`fx`</b>: Floating-type `Tensor` equal to `f(x)` but which has gradient
  `stop_gradient(g(x))`.