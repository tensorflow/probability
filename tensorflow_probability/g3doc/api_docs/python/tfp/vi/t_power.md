<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.t_power" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.t_power

``` python
tfp.vi.t_power(
    logu,
    t,
    self_normalized=False,
    name=None
)
```

The T-Power Csiszar-function in log-space.

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

When `self_normalized = True` the T-Power Csiszar-function is:

```none
f(u) = s [ u**t - 1 - t(u - 1) ]
s = { -1   0 < t < 1
    { +1   otherwise
```

When `self_normalized = False` the `- t(u - 1)` term is omitted.

This is similar to the `amari_alpha` Csiszar-function, with the associated
divergence being the same up to factors depending only on `t`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`t`</b>:  `Tensor` of same `dtype` as `logu` and broadcastable shape.
* <b>`self_normalized`</b>: Python `bool` indicating whether `f'(u=1)=0`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`t_power_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated
    at `u = exp(logu)`.