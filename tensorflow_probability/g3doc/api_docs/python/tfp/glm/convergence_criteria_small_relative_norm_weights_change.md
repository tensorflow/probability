<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm.convergence_criteria_small_relative_norm_weights_change" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.glm.convergence_criteria_small_relative_norm_weights_change

Returns Python `callable` which indicates fitting procedure has converged.

``` python
tfp.glm.convergence_criteria_small_relative_norm_weights_change(
    tolerance=1e-05,
    norm_order=2
)
```



Defined in [`python/glm/fisher_scoring.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/glm/fisher_scoring.py).

<!-- Placeholder for "Used in" -->

Writing old, new `model_coefficients` as `w0`, `w1`, this function
defines convergence as,

```python
relative_euclidean_norm = (tf.norm(w0 - w1, ord=2, axis=-1) /
                           (1. + tf.norm(w0, ord=2, axis=-1)))
reduce_all(relative_euclidean_norm < tolerance)
```

where `tf.norm(x, ord=2)` denotes the [Euclidean norm](
https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of `x`.

#### Args:

* <b>`tolerance`</b>: `float`-like `Tensor` indicating convergence, i.e., when
  max relative Euclidean norm weights difference < tolerance`.
  Default value: `1e-5`.
* <b>`norm_order`</b>: Order of the norm. Default value: `2` (i.e., "Euclidean norm".)


#### Returns:

* <b>`convergence_criteria_fn`</b>: Python `callable` which returns `bool` `Tensor`
  indicated fitting procedure has converged. (See inner function
  specification for argument signature.)
  Default value: `1e-5`.