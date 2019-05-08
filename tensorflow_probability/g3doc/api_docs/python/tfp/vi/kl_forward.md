<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.kl_forward" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.kl_forward

The forward Kullback-Leibler Csiszar-function in log-space.

``` python
tfp.vi.kl_forward(
    logu,
    self_normalized=False,
    name=None
)
```



Defined in [`python/vi/csiszar_divergence.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi/csiszar_divergence.py).

<!-- Placeholder for "Used in" -->

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

When `self_normalized = True`, the KL-forward Csiszar-function is:

```none
f(u) = u log(u) - (u - 1)
```

When `self_normalized = False` the `(u - 1)` term is omitted.

Observe that as an f-Divergence, this Csiszar-function implies:

```none
D_f[p, q] = KL[p, q]
```

The KL is "forward" because in maximum likelihood we think of minimizing `q`
as in `KL[p, q]`.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`self_normalized`</b>: Python `bool` indicating whether `f'(u=1)=0`. When
  `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
  when `p, q` are unnormalized measures.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`kl_forward_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated at
  `u = exp(logu)`.


#### Raises:

* <b>`TypeError`</b>: if `self_normalized` is `None` or a `Tensor`.