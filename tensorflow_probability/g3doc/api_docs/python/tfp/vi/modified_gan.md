<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.modified_gan" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.modified_gan

The Modified-GAN Csiszar-function in log-space.

``` python
tfp.vi.modified_gan(
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

When `self_normalized = True` the modified-GAN (Generative/Adversarial
Network) Csiszar-function is:

```none
f(u) = log(1 + u) - log(u) + 0.5 (u - 1)
```

When `self_normalized = False` the `0.5 (u - 1)` is omitted.

The unmodified GAN Csiszar-function is identical to Jensen-Shannon (with
`self_normalized = False`).

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`self_normalized`</b>: Python `bool` indicating whether `f'(u=1)=0`. When
  `f'(u=1)=0` the implied Csiszar f-Divergence remains non-negative even
  when `p, q` are unnormalized measures.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`chi_square_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated
  at `u = exp(logu)`.