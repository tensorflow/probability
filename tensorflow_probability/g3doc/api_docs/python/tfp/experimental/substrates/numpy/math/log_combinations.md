<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.numpy.math.log_combinations" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.numpy.math.log_combinations


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/numpy/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Multinomial coefficient.

### Aliases:

* `tfp.experimental.substrates.numpy.math.generic.log_combinations`


``` python
tfp.experimental.substrates.numpy.math.log_combinations(
    n,
    counts,
    name='log_combinations'
)
```



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