<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.chi_square" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.chi_square


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/csiszar_divergence.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



The chi-Square Csiszar-function in log-space.

``` python
tfp.vi.chi_square(
    logu,
    name=None
)
```



<!-- Placeholder for "Used in" -->

A Csiszar-function is a member of,

```none
F = { f:R_+ to R : f convex }.
```

The Chi-square Csiszar-function is:

```none
f(u) = u**2 - 1
```

Warning: this function makes non-log-space calculations and may therefore be
numerically unstable for `|logu| >> 0`.

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` representing `log(u)` from above.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.


#### Returns:


* <b>`chi_square_of_u`</b>: `float`-like `Tensor` of the Csiszar-function evaluated
  at `u = exp(logu)`.