<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.mcmc.nuts" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfp.experimental.mcmc.nuts

No U-Turn Sampler.



Defined in [`python/experimental/mcmc/nuts.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/experimental/mcmc/nuts.py).

<!-- Placeholder for "Used in" -->

The implementation closely follows [1; Algorithm 3].
The path length is set adaptively; the step size is fixed.

Achieves batch execution across chains by using
`tensorflow_probability.python.internal.auto_batching` internally.

This code is not yet integrated into the tensorflow_probability.mcmc Markov
chain Monte Carlo library.

#### References

[1]: Matthew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively
     Setting Path Lengths in Hamiltonian Monte Carlo.
     In _Journal of Machine Learning Research_, 15(1):1593-1623, 2014.
     http://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf

## Classes

[`class NoUTurnSampler`](../../../tfp/experimental/mcmc/NoUTurnSampler.md): Runs one step of the No U-Turn Sampler.

