Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.total_variation" />
</div>

# tfp.vi.total_variation

``` python
tfp.vi.total_variation(
    logu,
    name=None
)
```

The Total Variation Csiszar-function in log-space.

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The Total-Variation Csiszar-function is:

```none
f(u) = 0.5 |u - 1|
```

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`total_variation_of_u`</b>: `float`-like `Tensor` of the Csiszar-function
    evaluated at `u = exp(logu)`.