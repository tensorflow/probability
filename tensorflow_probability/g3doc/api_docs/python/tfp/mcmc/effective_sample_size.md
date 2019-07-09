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
    filter_beyond_positive_pairs=False,
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

If the sequence is uncorrelated, `ESS = N`.  If the sequence is positively
auto-correlated, `ESS` will be less than `N`. If there are negative
correlations, then `ESS` can exceed `N`.

#### Args:


* <b>`states`</b>:  `Tensor` or list of `Tensor` objects.  Dimension zero should index
  identically distributed states.
* <b>`filter_threshold`</b>:  `Tensor` or list of `Tensor` objects.
  Must broadcast with `state`.  The auto-correlation sequence is truncated
  after the first appearance of a term less than `filter_threshold`.
  Setting to `None` means we use no threshold filter.  Since `|R_k| <= 1`,
  setting to any number less than `-1` has the same effect. Ignored if
  `filter_beyond_positive_pairs` is `True`.
* <b>`filter_beyond_lag`</b>:  `Tensor` or list of `Tensor` objects.  Must be
  `int`-like and scalar valued.  The auto-correlation sequence is truncated
  to this length.  Setting to `None` means we do not filter based on number
  of lags.
* <b>`filter_beyond_positive_pairs`</b>: Python boolean. If `True`, only consider the
  initial auto-correlation sequence where the pairwise sums are positive.
* <b>`name`</b>:  `String` name to prepend to created ops.


#### Returns:


* <b>`ess`</b>:  `Tensor` or list of `Tensor` objects.  The effective sample size of
  each component of `states`.  Shape will be `states.shape[1:]`.


#### Raises:


* <b>`ValueError`</b>:  If `states` and `filter_threshold` or `states` and
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

ess = effective_sample_size(states, filter_beyond_positive_pairs=True)
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
`R_k` should be truncated at some number `filter_beyond_lag < N`. This
function provides two methods to perform this truncation.

* `filter_threshold` -- since many MCMC methods generate chains where `R_k >
  0`, a reasonable criteria is to truncate at the first index where the
  estimated auto-correlation becomes negative. This method does not estimate
  the `ESS` of super-efficient chains (where `ESS > N`) correctly.

* `filter_beyond_positive_pairs` -- reversible MCMC chains produce
  auto-correlation sequence with the property that pairwise sums of the
  elements of that sequence are positive [1] (i.e. `R_{2k} + R_{2k + 1} > 0`
  for `k in {0, ..., N/2}`). Deviations are only possible due to noise. This
  method truncates the auto-correlation sequence where the pairwise sums
  become non-positive.

The arguments `filter_beyond_lag`, `filter_threshold` and
`filter_beyond_positive_pairs` are filters intended to remove noisy tail terms
from `R_k`.  You can combine `filter_beyond_lag` with `filter_threshold` or
`filter_beyond_positive_pairs. E.g. combining `filter_beyond_lag` and
`filter_beyond_positive_pairs` means that terms are removed if they were to be
filtered under the `filter_beyond_lag` OR `filter_beyond_positive_pairs`
criteria.

#### References

[1]: Geyer, C. J. Practical Markov chain Monte Carlo (with discussion).
     Statistical Science, 7:473-511, 1992.