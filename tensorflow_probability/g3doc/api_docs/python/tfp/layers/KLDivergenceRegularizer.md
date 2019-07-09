<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.layers.KLDivergenceRegularizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
</div>

# tfp.layers.KLDivergenceRegularizer

## Class `KLDivergenceRegularizer`

Regularizer that adds a KL divergence penalty to the model loss.



### Aliases:

* Class `tfp.layers.KLDivergenceRegularizer`
* Class `tfp.layers.distribution_layer.KLDivergenceRegularizer`



Defined in [`python/layers/distribution_layer.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/layers/distribution_layer.py).

<!-- Placeholder for "Used in" -->

When using Monte Carlo approximation (e.g., `use_exact=False`), it is presumed
that the input distribution's concretization (i.e.,
`tf.convert_to_tensor(distribution)`) corresponds to a random sample. To
override this behavior, set `test_points_fn`.

#### Example

```python
tfd = tfp.distributions
tfpl = tfp.layers
tfk = tf.keras
tfkl = tf.keras.layers

# Create a variational encoder and add a KL Divergence penalty to the
# loss that encourages marginal coherence with a unit-MVN (the "prior").
input_shape = [28, 28, 1]
encoded_size = 2
variational_encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Flatten(),
    tfkl.Dense(10, activation='relu'),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size)),
    tfpl.MultivariateNormalTriL(
        encoded_size,
        lambda s: s.sample(10),
        activity_regularizer=tfpl.KLDivergenceRegularizer(
           tfd.MultivariateNormalDiag(loc=tf.zeros(encoded_size)),
           weight=num_train_samples)),
])
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    distribution_b,
    use_exact_kl=False,
    test_points_reduce_axis=(),
    test_points_fn=tf.convert_to_tensor,
    weight=None
)
```

Initialize the `KLDivergenceRegularizer` regularizer.


#### Args:


* <b>`distribution_b`</b>: distribution instance corresponding to `b` as in
  `KL[a, b]`. The previous layer's output is presumed to be a
  `Distribution` instance and is `a`).
* <b>`use_exact_kl`</b>: Python `bool` indicating if KL divergence should be
  calculated exactly via <a href="../../tfp/distributions/kl_divergence.md"><code>tfp.distributions.kl_divergence</code></a> or via Monte
  Carlo approximation.
  Default value: `False`.
* <b>`test_points_reduce_axis`</b>: `int` vector or scalar representing dimensions
  over which to `reduce_mean` while calculating the Monte Carlo
  approximation of the KL divergence.  As is with all `tf.reduce_*` ops,
  `None` means reduce over all dimensions; `()` means reduce over none of
  them.
  Default value: `()` (i.e., no reduction).
* <b>`test_points_fn`</b>: Python `callable` taking a `Distribution` instance and
  returning a `Tensor` used for random test points to approximate the KL
  divergence.
  Default value: `tf.convert_to_tensor`.
* <b>`weight`</b>: Multiplier applied to the calculated KL divergence for each Keras
  batch member.
  Default value: `None` (i.e., do not weight each batch member).



## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(distribution_a)
```

Call self as a function.


<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config
)
```






