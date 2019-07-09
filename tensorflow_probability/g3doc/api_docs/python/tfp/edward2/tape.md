<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.tape" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.tape

Context manager for recording interceptable executions onto a tape.

``` python
tfp.edward2.tape(
    *args,
    **kwds
)
```

<!-- Placeholder for "Used in" -->

Similar to `tf.GradientTape`, operations are recorded if they are executed
within this context manager. In addition, the operation must be registered
(wrapped) as `ed.interceptable`.

#### Yields:


* <b>`tape`</b>: OrderedDict where operations are recorded in sequence. Keys are
  the `name` keyword argument to the operation (typically, a random
  variable's `name`) and values are the corresponding output of the
  operation. If the operation has no name, it is not recorded.

#### Examples

```python
from tensorflow_probability import edward2 as ed

def probabilistic_matrix_factorization():
  users = ed.Normal(0., 1., sample_shape=[5000, 128], name="users")
  items = ed.Normal(0., 1., sample_shape=[7500, 128], name="items")
  ratings = ed.Normal(loc=tf.matmul(users, items, transpose_b=True),
                      scale=0.1,
                      name="ratings")
  return ratings

with ed.tape() as model_tape:
  ratings = probabilistic_matrix_factorization()

assert model_tape["users"].shape == (5000, 128)
assert model_tape["items"].shape == (7500, 128)
assert model_tape["ratings"] == ratings
```