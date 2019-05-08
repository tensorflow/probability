<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.lu_reconstruct" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.lu_reconstruct

The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

``` python
tfp.math.lu_reconstruct(
    lower_upper,
    perm,
    validate_args=False,
    name=None
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`lower_upper`</b>: `lu` as returned by `tf.linalg.lu`, i.e., if
  `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
* <b>`perm`</b>: `p` as returned by `tf.linag.lu`, i.e., if
  `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
* <b>`validate_args`</b>: Python `bool` indicating whether arguments should be checked
  for correctness.
  Default value: `False` (i.e., don't validate arguments).
* <b>`name`</b>: Python `str` name given to ops managed by this object.
  Default value: `None` (i.e., "lu_reconstruct").


#### Returns:

  x: The original input to `tf.linalg.lu`, i.e., `x` as in,
    `lu_reconstruct(*tf.linalg.lu(x))`.

#### Examples

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

x = [[[3., 4], [1, 2]],
     [[7., 8], [3, 4]]]
x_reconstructed = tfp.math.lu_reconstruct(*tf.linalg.lu(x))
tf.assert_near(x, x_reconstructed)
# ==> True
```