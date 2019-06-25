<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.math.log_combinations" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.math.log_combinations

Multinomial coefficient.

``` python
tfp.math.log_combinations(
    n,
    counts,
    name='log_combinations'
)
```



Defined in [`python/math/generic.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/math/generic.py).

<!-- Placeholder for "Used in" -->

Given `n` and `counts`, where `counts` has last dimension `k`, we compute
the multinomial coefficient as:

```n! / sum_i n_i!```

where `i` runs over all `k` classes.

#### Args:


* <b>`n`</b>: Floating-point `Tensor` broadcastable with `counts`. This represents `n`
  outcomes.
* <b>`counts`</b>: Floating-point `Tensor` broadcastable with `n`. This represents
  counts in `k` classes, where `k` is the last dimension of the tensor.
* <b>`name`</b>: A name for this operation (optional).


#### Returns:


* <b>`log_combinations`</b>: `Tensor` representing the multinomial coefficient between
  `n` and `counts`.