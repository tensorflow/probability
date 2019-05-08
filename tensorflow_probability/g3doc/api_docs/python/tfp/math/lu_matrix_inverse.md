<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.lu_matrix_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.lu_matrix_inverse

Computes a matrix inverse given the matrix's LU decomposition.

``` python
tfp.math.lu_matrix_inverse(
    lower_upper,
    perm,
    validate_args=False,
    name=None
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->

This op is conceptually identical to,

````python
inv_X = tf.lu_matrix_inverse(*tf.linalg.lu(X))
tf.assert_near(tf.matrix_inverse(X), inv_X)
# ==> True
```

Note: this function does not verify the implied matrix is actually invertible
nor is this condition checked even when `validate_args=True`.

#### Args:

* <b>`lower_upper`</b>: `lu` as returned by `tf.linalg.lu`, i.e., if
  `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
* <b>`perm`</b>: `p` as returned by `tf.linag.lu`, i.e., if
  `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
* <b>`validate_args`</b>: Python `bool` indicating whether arguments should be checked
  for correctness. Note: this function does not verify the implied matrix is
  actually invertible, even when `validate_args=True`.
  Default value: `False` (i.e., don't validate arguments).
* <b>`name`</b>: Python `str` name given to ops managed by this object.
  Default value: `None` (i.e., "lu_matrix_inverse").


#### Returns:

  inv_x: The matrix_inv, i.e.,
    `tf.matrix_inverse(tfp.math.lu_reconstruct(lu, perm))`.

#### Examples

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

x = [[[3., 4], [1, 2]],
     [[7., 8], [3, 4]]]
inv_x = tfp.math.lu_matrix_inverse(*tf.linalg.lu(x))
tf.assert_near(tf.matrix_inverse(x), inv_x)
# ==> True
```