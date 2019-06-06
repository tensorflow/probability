<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.symmetrized_csiszar_function" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.symmetrized_csiszar_function

Symmetrizes a Csiszar-function in log-space.

``` python
tfp.vi.symmetrized_csiszar_function(
    logu,
    csiszar_function,
    name=None
)
```



Defined in [`python/vi/csiszar_divergence.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/vi/csiszar_divergence.py).

<!-- Placeholder for "Used in" -->

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The symmetrized Csiszar-function is defined as:

```none
f_g(u) = 0.5 g(u) + 0.5 u g (1 / u)
```

where `g` is some other Csiszar-function.

We say the function is "symmetrized" because:

```none
D_{f_g}[p, q] = D_{f_g}[q, p]
```

for all `p << >> q` (i.e., `support(p) = support(q)`).

There exists alternatives for symmetrizing a Csiszar-function. For example,

```none
f_g(u) = max(f(u), f^*(u)),
```

where `f^*` is the dual Csiszar-function, also implies a symmetric
f-Divergence.

#### Example:



When either of the following functions are symmetrized, we obtain the
Jensen-Shannon Csiszar-function, i.e.,

```none
g(u) = -log(u) - (1 + u) log((1 + u) / 2) + u - 1
h(u) = log(4) + 2 u log(u / (1 + u))
```

implies,

```none
f_g(u) = f_h(u) = u log(u) - (1 + u) log((1 + u) / 2)
       = jensen_shannon(log(u)).
```

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`csiszar_function`</b>: Python `callable` representing a Csiszar-function over
  log-domain.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`symmetrized_g_of_u`</b>: `float`-like `Tensor` of the result of applying the
  symmetrization of `g` evaluated at `u = exp(logu)`.