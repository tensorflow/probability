<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.jensen_shannon" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.jensen_shannon

The Jensen-Shannon Csiszar-function in log-space.

``` python
tfp.vi.jensen_shannon(
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

When `self_normalized = True`, the Jensen-Shannon Csiszar-function is:

```none
f(u) = u log(u) - (1 + u) log(1 + u) + (u + 1) log(2)
```

When `self_normalized = False` the `(u + 1) log(2)` term is omitted.

Observe that as an f-Divergence, this Csiszar-function implies:

```none
D_f[p, q] = KL[p, m] + KL[q, m]
m(x) = 0.5 p(x) + 0.5 q(x)
```

In a sense, this divergence is the "reverse" of the Arithmetic-Geometric
f-Divergence.

This Csiszar-function induces a symmetric f-Divergence, i.e.,
`D_f[p, q] = D_f[q, p]`.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

For more information, see:
  Lin, J. "Divergence measures based on the Shannon entropy." IEEE Trans.
  Inf. Th., 37, 145-151, 1991.

#### Args:

* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`self_normalized`</b>: Python `bool` indicating whether `f'(u=1)=0`. When
  `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
  when `p, q` are unnormalized measures.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:

* <b>`jensen_shannon_of_u`</b>: `float`-like `Tensor` of the Csiszar-function
  evaluated at `u = exp(logu)`.