<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.SliceSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="max_doublings"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="step_size"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.SliceSampler

## Class `SliceSampler`

Runs one step of the slice sampler using a hit and run approach.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/slice_sampler_kernel.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/slice_sampler_kernel.py).

<!-- Placeholder for "Used in" -->

Slice Sampling is a Markov Chain Monte Carlo (MCMC) algorithm based, as stated
by [Neal (2003)][1], on the observation that "...one can sample from a
distribution by sampling uniformly from the region under the plot of its
density function. A Markov chain that converges to this uniform distribution
can be constructed by alternately uniform sampling in the vertical direction
with uniform sampling from the horizontal `slice` defined by the current
vertical position, or more generally, with some update that leaves the uniform
distribution over this slice invariant". Mathematical details and derivations
can be found in [Neal (2003)][1]. The one dimensional slice sampler is
extended to n-dimensions through use of a hit-and-run approach: choose a
random direction in n-dimensional space and take a step, as determined by the
one-dimensional slice sampling algorithm, along that direction
[Belisle at al. 1993][2].

The `one_step` function can update multiple chains in parallel. It assumes
that all leftmost dimensions of `current_state` index independent chain states
(and are therefore updated independently). The output of
`target_log_prob_fn(*current_state)` should sum log-probabilities across all
event dimensions. Slices along the rightmost dimensions may have different
target distributions; for example, `current_state[0, :]` could have a
different target distribution from `current_state[1, :]`. These semantics are
governed by `target_log_prob_fn(*current_state)`. (The number of independent
chains is `tf.size(target_log_prob_fn(*current_state))`.)

Note that the sampler only supports states where all components have a common
dtype.

### Examples:

#### Simple chain with warm-up.

In this example we sample from a standard univariate normal
distribution using slice sampling.

```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np

  tfd = tfp.distributions

  dtype = np.float32

  target = tfd.Normal(loc=dtype(0), scale=dtype(1))

  samples, _ = tfp.mcmc.sample_chain(
      num_results=1000,
      current_state=dtype(1),
      kernel=tfp.mcmc.SliceSampler(
          target.log_prob,
          step_size=1.0,
          max_doublings=5,
          seed=1234),
      num_burnin_steps=500,
      parallel_iterations=1)  # For determinism.

  sample_mean = tf.reduce_mean(samples, axis=0)
  sample_std = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(samples, sample_mean),
                   axis=0))

  with tf.Session() as sess:
    [sample_mean, sample_std] = sess.run([sample_mean, sample_std])

  print "Sample mean: ", sample_mean
  print "Sample Std: ", sample_std
```

#### Sample from a Two Dimensional Normal.

In the following example we sample from a two dimensional Normal
distribution using slice sampling.

```python
  import tensorflow as tf
  import tensorflow_probability as tfp
  import numpy as np

  tfd = tfp.distributions

  dtype = np.float32
  true_mean = dtype([0, 0])
  true_cov = dtype([[1, 0.5], [0.5, 1]])
  num_results = 500
  num_chains = 50

  # Target distribution is defined through the Cholesky decomposition
  chol = tf.linalg.cholesky(true_cov)
  target = tfd.MultivariateNormalTriL(loc=true_mean, scale_tril=chol)

  # Assume that the state is passed as a list of 1-d tensors `x` and `y`.
  # Then the target log-density is defined as follows:
  def target_log_prob(x, y):
    # Stack the input tensors together
    z = tf.stack([x, y], axis=-1) - true_mean
    return target.log_prob(z)

  # Initial state of the chain
  init_state = [np.ones([num_chains, 1], dtype=dtype),
                np.ones([num_chains, 1], dtype=dtype)]

  # Run Slice Samper for `num_results` iterations for `num_chains`
  # independent chains:
  [x, y], _ = tfp.mcmc.sample_chain(
      num_results=num_results,
      current_state=init_state,
      kernel=tfp.mcmc.SliceSampler(
          target_log_prob_fn=target_log_prob,
          step_size=1.0,
          max_doublings=5,
          seed=47),
      num_burnin_steps=200,
      num_steps_between_results=1,
      parallel_iterations=1)

  states = tf.stack([x, y], axis=-1)
  sample_mean = tf.reduce_mean(states, axis=[0, 1])
  z = states - sample_mean
  sample_cov = tf.reduce_mean(tf.matmul(z, z, transpose_a=True),
                              axis=[0, 1])

  with tf.Session() as sess:
    [sample_mean, sample_cov] = sess.run([
        sample_mean, sample_cov])

  print "sample_mean: ", sample_mean
  print "sample_cov: ", sample_cov
```

### References

[1]: Radford M. Neal. Slice Sampling. The Annals of Statistics. 2003, Vol 31,
     No. 3 , 705-767.
     https://projecteuclid.org/download/pdf_1/euclid.aos/1056562461

[2]: C.J.P. Belisle, H.E. Romeijn, R.L. Smith. Hit-and-run algorithms for
     generating multivariate distributions. Math. Oper. Res., 18(1993),
     225-266.
     https://www.jstor.org/stable/3690278?seq=1#page_scan_tab_contents

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    max_doublings,
    seed=None,
    name=None
)
```

Initializes this transition kernel.

#### Args:

* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
  `current_state` (or `*current_state` if it is a list) and returns its
  (possibly unnormalized) log-density under the target distribution.
* <b>`step_size`</b>: Scalar or `tf.Tensor` with same dtype as and shape compatible
  with `x_initial`. The size of the initial interval.
* <b>`max_doublings`</b>: Scalar positive int32 `tf.Tensor`. The maximum number of
doublings to consider.
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'slice_sampler_kernel').


#### Returns:

* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
  of the Markov chain(s) at each result step. Has same shape as
  `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="max_doublings"><code>max_doublings</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Returns `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



<h3 id="step_size"><code>step_size</code></h3>



<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>





## Methods

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(init_state)
```

Returns an object with the same type as returned by `one_step(...)[1]`.

#### Args:

* <b>`init_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  initial state(s) of the Markov chain(s).


#### Returns:

* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
  `Tensor`s representing internal calculations made within this function.

<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Runs one iteration of Slice Sampler.

#### Args:

* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s). The first `r` dimensions
  index independent chains,
  `r = tf.rank(target_log_prob_fn(*current_state))`.
* <b>`previous_kernel_results`</b>: `collections.namedtuple` containing `Tensor`s
  representing values from previous calls to this function (or from the
  `bootstrap_results` function.)


#### Returns:

* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
  of the Markov chain(s) after taking exactly one step. Has same type and
  shape as `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.


#### Raises:

* <b>`ValueError`</b>: if there isn't one `step_size` or a list with same length as
  `current_state`.
* <b>`TypeError`</b>: if `not target_log_prob.dtype.is_floating`.



