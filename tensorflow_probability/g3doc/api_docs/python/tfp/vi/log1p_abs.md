<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.log1p_abs" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.log1p_abs

The log1p-abs Csiszar-function in log-space.

``` python
tfp.vi.log1p_abs(
    logu,
    name=None
)
```



Defined in [`python/vi/csiszar_divergence.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi/csiszar_divergence.py).

<!-- Placeholder for "Used in" -->

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The Log1p-Abs Csiszar-function is:

```none
f(u) = u**(sign(u-1)) - 1
```

This function is so-named because it was invented from the following recipe.
Choose a convex function g such that g(0)=0 and solve for f:

```none
log(1 + f(u)) = g(log(u)).
  <=>
f(u) = exp(g(log(u))) - 1
```

That is, the graph is identically `g` when y-axis is `log1p`-domain and x-axis
is `log`-domain.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`log1p_abs_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated
  at `u = exp(logu)`.