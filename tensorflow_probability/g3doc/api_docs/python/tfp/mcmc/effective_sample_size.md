<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.effective_sample_size" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.effective_sample_size

Estimate a lower bound on effective sample size for each independent chain.

``` python
tfp.mcmc.effective_sample_size(
    states,
    filter_threshold=0.0,
    filter_beyond_lag=None,
    name=None
)
```



Defined in [`python/mcmc/diagnostic.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/diagnostic.py).

<!-- Placeholder for "Used in" -->

Roughly speaking, "effective sample size" (ESS) is the size of an iid sample
with the same variance as `state`.

More precisely, given a stationary sequence of possibly correlated random
variables `X_1, X_2,...,X_N`, each identically distributed ESS is the number
such that

```Variance{ N**-1 * Sum{X_i} } = ESS**-1 * Variance{ X_1 }.```

If the sequence is uncorrelated, `ESS = N`.  In general, one should expect
`ESS <= N`, with more highly correlated sequences having smaller `ESS`.

#### Args:

* <b>`states`</b>:  `Tensor` or list of `Tensor` objects.  Dimension zero should index
  identically distributed states.
* <b>`filter_threshold`</b>:  `Tensor` or list of `Tensor` objects.
  Must broadcast with `state`.  The auto-correlation sequence is truncated
  after the first appearance of a term less than `filter_threshold`.
  Setting to `None` means we use no threshold filter.  Since `|R_k| <= 1`,
  setting to any number less than `-1` has the same effect.
* <b>`filter_beyond_lag`</b>:  `Tensor` or list of `Tensor` objects.  Must be
  `int`-like and scalar valued.  The auto-correlation sequence is truncated
  to this length.  Setting to `None` means we do not filter based on number
  of lags.
* <b>`name`</b>:  `String` name to prepend to created ops.


#### Returns:

* <b>`ess`</b>:  `Tensor` or list of `Tensor` objects.  The effective sample size of
  each component of `states`.  Shape will be `states.shape[1:]`.


#### Raises:

  ValueError:  If `states` and `filter_threshold` or `states` and
    `filter_beyond_lag` are both lists with different lengths.

#### Examples

We use ESS to estimate standard error.

```
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])

# Get 1000 states from one chain.
states = tfp.mcmc.sample_chain(
    num_burnin_steps=200,
    num_results=1000,
    current_state=tf.constant([0., 0.]),
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=target.log_prob,
      step_size=0.05,
      num_leapfrog_steps=20))
states.shape
==> (1000, 2)

ess = effective_sample_size(states)
==> Shape (2,) Tensor

mean, variance = tf.nn.moments(states, axis=0)
standard_error = tf.sqrt(variance / ess)
```

Some math shows that, with `R_k` the auto-correlation sequence,
`R_k := Covariance{X_1, X_{1+k}} / Variance{X_1}`, we have

```ESS(N) =  N / [ 1 + 2 * ( (N - 1) / N * R_1 + ... + 1 / N * R_{N-1}  ) ]```

This function estimates the above by first estimating the auto-correlation.
Since `R_k` must be estimated using only `N - k` samples, it becomes
progressively noisier for larger `k`.  For this reason, the summation over
`R_k` should be truncated at some number `filter_beyond_lag < N`.  Since many
MCMC methods generate chains where `R_k > 0`, a reasonable criteria is to
truncate at the first index where the estimated auto-correlation becomes
negative.

The arguments `filter_beyond_lag`, `filter_threshold` are filters intended to
remove noisy tail terms from `R_k`.  They combine in an "OR" manner meaning
terms are removed if they were to be filtered under the `filter_beyond_lag` OR
`filter_threshold` criteria.