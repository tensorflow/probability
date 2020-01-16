<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.arithmetic_geometric" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.arithmetic_geometric


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/csiszar_divergence.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



The Arithmetic-Geometric Csiszar-function in log-space.

``` python
tfp.vi.arithmetic_geometric(
    logu,
    self_normalized=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

When `self_normalized = True` the Arithmetic-Geometric Csiszar-function is:

```none
f(u) = (1 + u) log( (1 + u) / sqrt(u) ) - (1 + u) log(2)
```

When `self_normalized = False` the `(1 + u) log(2)` term is omitted.

Observe that as an f-Divergence, this Csiszar-function implies:

```none
D_f[p, q] = KL[m, p] + KL[m, q]
m(x) = 0.5 p(x) + 0.5 q(x)
```

In a sense, this divergence is the "reverse" of the Jensen-Shannon
f-Divergence.

This Csiszar-function induces a symmetric f-Divergence, i.e.,
`D_f[p, q] = D_f[q, p]`.

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`self_normalized`</b>: Python `bool` indicating whether `f'(u=1)=0`. When
  `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
  when `p, q` are unnormalized measures.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`arithmetic_geometric_of_u`</b>: `float`-like `Tensor` of the
  Csiszar-function evaluated at `u = exp(logu)`.