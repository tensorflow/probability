Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.triangular" />
</div>

# tfp.vi.triangular

``` python
tfp.vi.triangular(
    logu,
    name=None
)
```

The Triangular Csiszar-function in log-space.

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The Triangular Csiszar-function is:

```none
f(u) = (u - 1)**2 / (1 + u)
```

This Csiszar-function induces a symmetric f-Divergence, i.e.,
`D_f[p, q] = D_f[q, p]`.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`triangular_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated
    at `u = exp(logu)`.