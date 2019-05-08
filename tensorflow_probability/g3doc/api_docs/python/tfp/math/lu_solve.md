<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.lu_solve" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.lu_solve

Solves systems of linear eqns `A X = RHS`, given LU factorizations.

``` python
tfp.math.lu_solve(
    lower_upper,
    perm,
    rhs,
    validate_args=False,
    name=None
)
```



Defined in [`python/math/linalg.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/linalg.py).

<!-- Placeholder for "Used in" -->

Note: this function does not verify the implied matrix is actually invertible
nor is this condition checked even when `validate_args=True`.

#### Args:

* <b>`lower_upper`</b>: `lu` as returned by `tf.linalg.lu`, i.e., if
  `matmul(P, matmul(L, U)) = X` then `lower_upper = L + U - eye`.
* <b>`perm`</b>: `p` as returned by `tf.linag.lu`, i.e., if
  `matmul(P, matmul(L, U)) = X` then `perm = argmax(P)`.
* <b>`rhs`</b>: Matrix-shaped float `Tensor` representing targets for which to solve;
  `A X = RHS`. To handle vector cases, use:
  `lu_solve(..., rhs[..., tf.newaxis])[..., 0]`.
* <b>`validate_args`</b>: Python `bool` indicating whether arguments should be checked
  for correctness. Note: this function does not verify the implied matrix is
  actually invertible, even when `validate_args=True`.
  Default value: `False` (i.e., don't validate arguments).
* <b>`name`</b>: Python `str` name given to ops managed by this object.
  Default value: `None` (i.e., "lu_solve").


#### Returns:

  x: The `X` in `A @ X = RHS`.

#### Examples

```python
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

x = [[[1., 2],
      [3, 4]],
     [[7, 8],
      [3, 4]]]
inv_x = tfp.math.lu_solve(*tf.linalg.lu(x), rhs=tf.eye(2))
tf.assert_near(tf.matrix_inverse(x), inv_x)
# ==> True
```