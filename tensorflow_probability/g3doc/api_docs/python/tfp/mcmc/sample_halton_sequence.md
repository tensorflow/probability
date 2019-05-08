<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.sample_halton_sequence" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.sample_halton_sequence

Returns a sample from the `dim` dimensional Halton sequence.

``` python
tfp.mcmc.sample_halton_sequence(
    dim,
    num_results=None,
    sequence_indices=None,
    dtype=tf.float32,
    randomized=True,
    seed=None,
    name=None
)
```



Defined in [`python/mcmc/sample_halton_sequence.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/sample_halton_sequence.py).

<!-- Placeholder for "Used in" -->

Warning: The sequence elements take values only between 0 and 1. Care must be
taken to appropriately transform the domain of a function if it differs from
the unit cube before evaluating integrals using Halton samples. It is also
important to remember that quasi-random numbers without randomization are not
a replacement for pseudo-random numbers in every context. Quasi random numbers
are completely deterministic and typically have significant negative
autocorrelation unless randomization is used.

Computes the members of the low discrepancy Halton sequence in dimension
`dim`. The `dim`-dimensional sequence takes values in the unit hypercube in
`dim` dimensions. Currently, only dimensions up to 1000 are supported. The
prime base for the k-th axes is the k-th prime starting from 2. For example,
if `dim` = 3, then the bases will be [2, 3, 5] respectively and the first
element of the non-randomized sequence will be: [0.5, 0.333, 0.2]. For a more
complete description of the Halton sequences see
[here](https://en.wikipedia.org/wiki/Halton_sequence). For low discrepancy
sequences and their applications see
[here](https://en.wikipedia.org/wiki/Low-discrepancy_sequence).

If `randomized` is true, this function produces a scrambled version of the
Halton sequence introduced by [Owen (2017)][1]. For the advantages of
randomization of low discrepancy sequences see [here](
https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method#Randomization_of_quasi-Monte_Carlo).

The number of samples produced is controlled by the `num_results` and
`sequence_indices` parameters. The user must supply either `num_results` or
`sequence_indices` but not both.
The former is the number of samples to produce starting from the first
element. If `sequence_indices` is given instead, the specified elements of
the sequence are generated. For example, sequence_indices=tf.range(10) is
equivalent to specifying n=10.

#### Examples

```python
import tensorflow as tf
import tensorflow_probability as tfp

# Produce the first 1000 members of the Halton sequence in 3 dimensions.
num_results = 1000
dim = 3
sample = tfp.mcmc.sample_halton_sequence(
  dim,
  num_results=num_results,
  seed=127)

# Evaluate the integral of x_1 * x_2^2 * x_3^3  over the three dimensional
# hypercube.
powers = tf.range(1.0, limit=dim + 1)
integral = tf.reduce_mean(tf.reduce_prod(sample ** powers, axis=-1))
true_value = 1.0 / tf.reduce_prod(powers + 1.0)
with tf.Session() as session:
  values = session.run((integral, true_value))

# Produces a relative absolute error of 1.7%.
print ("Estimated: %f, True Value: %f" % values)

# Now skip the first 1000 samples and recompute the integral with the next
# thousand samples. The sequence_indices argument can be used to do this.


sequence_indices = tf.range(start=1000, limit=1000 + num_results,
                            dtype=tf.int32)
sample_leaped = tfp.mcmc.sample_halton_sequence(
    dim,
    sequence_indices=sequence_indices,
    seed=111217)

integral_leaped = tf.reduce_mean(tf.reduce_prod(sample_leaped ** powers,
                                                axis=-1))
with tf.Session() as session:
  values = session.run((integral_leaped, true_value))
# Now produces a relative absolute error of 0.05%.
print ("Leaped Estimated: %f, True Value: %f" % values)
```

#### Args:

* <b>`dim`</b>: Positive Python `int` representing each sample's `event_size.` Must
  not be greater than 1000.
* <b>`num_results`</b>: (Optional) Positive scalar `Tensor` of dtype int32. The number
  of samples to generate. Either this parameter or sequence_indices must
  be specified but not both. If this parameter is None, then the behaviour
  is determined by the `sequence_indices`.
  Default value: `None`.
* <b>`sequence_indices`</b>: (Optional) `Tensor` of dtype int32 and rank 1. The
  elements of the sequence to compute specified by their position in the
  sequence. The entries index into the Halton sequence starting with 0 and
  hence, must be whole numbers. For example, sequence_indices=[0, 5, 6] will
  produce the first, sixth and seventh elements of the sequence. If this
  parameter is None, then the `num_results` parameter must be specified
  which gives the number of desired samples starting from the first sample.
  Default value: `None`.
* <b>`dtype`</b>: (Optional) The dtype of the sample. One of: `float16`, `float32` or
  `float64`.
  Default value: `tf.float32`.
* <b>`randomized`</b>: (Optional) bool indicating whether to produce a randomized
  Halton sequence. If True, applies the randomization described in
  [Owen (2017)][1].
  Default value: `True`.
* <b>`seed`</b>: (Optional) Python integer to seed the random number generator. Only
  used if `randomized` is True. If not supplied and `randomized` is True,
  no seed is set.
  Default value: `None`.
* <b>`name`</b>:  (Optional) Python `str` describing ops managed by this function. If
  not supplied the name of this function is used.
  Default value: "sample_halton_sequence".


#### Returns:

* <b>`halton_elements`</b>: Elements of the Halton sequence. `Tensor` of supplied dtype
  and `shape` `[num_results, dim]` if `num_results` was specified or shape
  `[s, dim]` where s is the size of `sequence_indices` if `sequence_indices`
  were specified.


#### Raises:

  ValueError: if both `sequence_indices` and `num_results` were specified or
    if dimension `dim` is less than 1 or greater than 1000.

#### References

[1]: Art B. Owen. A randomized Halton algorithm in R. _arXiv preprint
     arXiv:1706.02808_, 2017. https://arxiv.org/abs/1706.02808