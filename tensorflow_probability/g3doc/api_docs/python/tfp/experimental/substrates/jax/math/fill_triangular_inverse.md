<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.math.fill_triangular_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.jax.math.fill_triangular_inverse


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/math/linalg.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Creates a vector from a (batch of) triangular matrix.

### Aliases:

* `tfp.experimental.substrates.jax.math.linalg.fill_triangular_inverse`


``` python
tfp.experimental.substrates.jax.math.fill_triangular_inverse(
    x,
    upper=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The vector is created from the lower-triangular or upper-triangular portion
depending on the value of the parameter `upper`.

If `x.shape` is `[b1, b2, ..., bB, n, n]` then the output shape is
`[b1, b2, ..., bB, d]` where `d = n (n + 1) / 2`.

#### Example:



```python
fill_triangular_inverse(
  [[4, 0, 0],
   [6, 5, 0],
   [3, 2, 1]])

# ==> [1, 2, 3, 4, 5, 6]

fill_triangular_inverse(
  [[1, 2, 3],
   [0, 5, 6],
   [0, 0, 4]], upper=True)

# ==> [1, 2, 3, 4, 5, 6]
```

#### Args:


* <b>`x`</b>: `Tensor` representing lower (or upper) triangular elements.
* <b>`upper`</b>: Python `bool` representing whether output matrix should be upper
  triangular (`True`) or lower triangular (`False`, default).
* <b>`name`</b>: Python `str`. The name to give this op.


#### Returns:


* <b>`flat_tril`</b>: (Batch of) vector-shaped `Tensor` representing vectorized lower
  (or upper) triangular elements from `x`.