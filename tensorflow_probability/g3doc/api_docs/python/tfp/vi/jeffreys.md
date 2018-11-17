<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.jeffreys" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.jeffreys

``` python
tfp.vi.jeffreys(
    logu,
    name=None
)
```

The Jeffreys Csiszar-function in log-space.

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The Jeffreys Csiszar-function is:

```none
f(u) = 0.5 ( u log(u) - log(u) )
     = 0.5 kl_forward + 0.5 kl_reverse
     = symmetrized_csiszar_function(kl_reverse)
     = symmetrized_csiszar_function(kl_forward)
```

This Csiszar-function induces a symmetric f-Divergence, i.e.,
`D_f[p, q] = D_f[q, p]`.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`jeffreys_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated
    at `u = exp(logu)`.