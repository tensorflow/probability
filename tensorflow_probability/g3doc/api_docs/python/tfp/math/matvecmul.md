<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.matvecmul" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.matvecmul

``` python
tfp.math.matvecmul(
    a,
    b,
    transpose_a=False,
    validate_args=False,
    name=None
)
```

Multiply a matrix by a vector.

Note that similarly to `tf.matmul`, this function does not broadcast its
arguments.

#### Args:

* <b>`a`</b>: (Batch of) matrix-shaped `Tensor`(s).
* <b>`b`</b>: (Batch of) vector-shaped `Tensor`(s).
* <b>`transpose_a`</b>: If `True`, `a` is transposed before multiplication.
* <b>`validate_args`</b>: When `True`, additional assertions might be embedded in the
    graph.
    Default value: `False` (i.e., no graph assertions are added).
* <b>`name`</b>: Python `str` prefixed to ops created by this function.
    Default value: "matvecmul".


#### Returns:

A vector-shaped `Tensor` containing the result of the a*b.


#### Raises:

* <b>`ValueError`</b>: if the dimensions or dtypes don't match up.

#### Examples

```python
import tensorflow as tf
import tensorflow_probability as tfp

a = tf.constant([[1., .4, .5],
                 [.4, .2, .25]])
b = tf.constant([.3, .7, .5])
tfp.matvecmul(a, b)
# ==> array([0.83, 0.385], dtype=float32)

y = tf.constant([.3, .7])
tfp.matvecmul(a, y, transpose_a=True)
# ==> array([0.58 , 0.26 , 0.325], dtype=float32)
```