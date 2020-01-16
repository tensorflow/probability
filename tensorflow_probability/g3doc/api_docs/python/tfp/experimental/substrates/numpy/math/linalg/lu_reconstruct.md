<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.math.linalg.lu_reconstruct" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.numpy.math.linalg.lu_reconstruct


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/numpy/math/linalg.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



The inverse LU decomposition, `X == lu_reconstruct(*tf.linalg.lu(X))`.

``` python
tfp.experimental.substrates.numpy.math.linalg.lu_reconstruct(
    lower_upper,
    perm,
    validate_args=False,
    name=None
)
```



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
  Default value: `None` (i.e., 'lu_reconstruct').


#### Returns:


* <b>`x`</b>: The original input to `tf.linalg.lu`, i.e., `x` as in,
  `lu_reconstruct(*tf.linalg.lu(x))`.

#### Examples

```python
import numpy as np
from tensorflow_probability.python.internal.backend import numpy as tf
import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.numpy

x = [[[3., 4], [1, 2]],
     [[7., 8], [3, 4]]]
x_reconstructed = tfp.math.lu_reconstruct(*tf.linalg.lu(x))
tf.assert_near(x, x_reconstructed)
# ==> True
```