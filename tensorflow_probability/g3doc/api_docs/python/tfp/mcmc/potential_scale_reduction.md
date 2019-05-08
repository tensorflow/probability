<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.potential_scale_reduction" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.potential_scale_reduction

Gelman and Rubin (1992)'s potential scale reduction for chain convergence.

``` python
tfp.mcmc.potential_scale_reduction(
    chains_states,
    independent_chain_ndims=1,
    name=None
)
```



Defined in [`python/mcmc/diagnostic.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/diagnostic.py).

<!-- Placeholder for "Used in" -->

Given `N > 1` states from each of `C > 1` independent chains, the potential
scale reduction factor, commonly referred to as R-hat, measures convergence of
the chains (to the same target) by testing for equality of means.
Specifically, R-hat measures the degree to which variance (of the means)
between chains exceeds what one would expect if the chains were identically
distributed. See [Gelman and Rubin (1992)][1]; [Brooks and Gelman (1998)][2].

#### Some guidelines:


* The initial state of the chains should be drawn from a distribution
  overdispersed with respect to the target.
* If all chains converge to the target, then as `N --> infinity`, R-hat --> 1.
  Before that, R-hat > 1 (except in pathological cases, e.g. if the chain
  paths were identical).
* The above holds for any number of chains `C > 1`.  Increasing `C` does
  improve effectiveness of the diagnostic.
* Sometimes, R-hat < 1.2 is used to indicate approximate convergence, but of
  course this is problem-dependent. See [Brooks and Gelman (1998)][2].
* R-hat only measures non-convergence of the mean. If higher moments, or
  other statistics are desired, a different diagnostic should be used. See
  [Brooks and Gelman (1998)][2].


#### Args:

* <b>`chains_states`</b>:  `Tensor` or Python `list` of `Tensor`s representing the
  states of a Markov Chain at each result step.  The `ith` state is
  assumed to have shape `[Ni, Ci1, Ci2,...,CiD] + A`.
  Dimension `0` indexes the `Ni > 1` result steps of the Markov Chain.
  Dimensions `1` through `D` index the `Ci1 x ... x CiD` independent
  chains to be tested for convergence to the same target.
  The remaining dimensions, `A`, can have any shape (even empty).
* <b>`independent_chain_ndims`</b>: Integer type `Tensor` with value `>= 1` giving the
  number of dimensions, from `dim = 1` to `dim = D`, holding independent
  chain results to be tested for convergence.
* <b>`name`</b>: `String` name to prepend to created tf.  Default:
  `potential_scale_reduction`.


#### Returns:

`Tensor` or Python `list` of `Tensor`s representing the R-hat statistic for
the state(s).  Same `dtype` as `state`, and shape equal to
`state.shape[1 + independent_chain_ndims:]`.


#### Raises:

  ValueError:  If `independent_chain_ndims < 1`.

#### Examples

Diagnosing convergence by monitoring 10 chains that each attempt to
sample from a 2-variate normal.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

target = tfd.MultivariateNormalDiag(scale_diag=[1., 2.])

# Get 10 (2x) overdispersed initial states.
initial_state = target.sample(10) * 2.
==> (10, 2)

# Get 1000 samples from the 10 independent chains.
chains_states, _ = tfp.mcmc.sample_chain(
    num_burnin_steps=200,
    num_results=1000,
    current_state=initial_state,
    kernel=tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target.log_prob,
        step_size=0.05,
        num_leapfrog_steps=20))
chains_states.shape
==> (1000, 10, 2)

rhat = tfp.mcmc.diagnostic.potential_scale_reduction(
    chains_states, independent_chain_ndims=1)

# The second dimension needed a longer burn-in.
rhat.eval()
==> [1.05, 1.3]
```

To see why R-hat is reasonable, let `X` be a random variable drawn uniformly
from the combined states (combined over all chains).  Then, in the limit
`N, C --> infinity`, with `E`, `Var` denoting expectation and variance,

```R-hat = ( E[Var[X | chain]] + Var[E[X | chain]] ) / E[Var[X | chain]].```

Using the law of total variance, the numerator is the variance of the combined
states, and the denominator is the total variance minus the variance of the
the individual chain means.  If the chains are all drawing from the same
distribution, they will have the same mean, and thus the ratio should be one.

#### References

[1]: Stephen P. Brooks and Andrew Gelman. General Methods for Monitoring
     Convergence of Iterative Simulations. _Journal of Computational and
     Graphical Statistics_, 7(4), 1998.

[2]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
     Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.