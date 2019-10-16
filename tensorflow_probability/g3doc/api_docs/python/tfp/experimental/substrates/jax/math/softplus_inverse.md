<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.substrates.jax.math.softplus_inverse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.substrates.jax.math.softplus_inverse


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/substrates/jax/math/generic.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

### Aliases:

* `tfp.experimental.substrates.jax.math.generic.softplus_inverse`


``` python
tfp.experimental.substrates.jax.math.softplus_inverse(
    x,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Mathematically this op is equivalent to:

```none
softplus_inverse = log(exp(x) - 1.)
```

#### Args:


* <b>`x`</b>: `Tensor`. Non-negative (not enforced), floating-point.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

`Tensor`. Has the same type/shape as input `x`.
