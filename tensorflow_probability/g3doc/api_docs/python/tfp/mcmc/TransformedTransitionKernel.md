<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.TransformedTransitionKernel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="bijector"/>
<meta itemprop="property" content="inner_kernel"/>
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.TransformedTransitionKernel

## Class `TransformedTransitionKernel`

TransformedTransitionKernel applies a bijector to the MCMC's state space.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/transformed_kernel.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/transformed_kernel.py).

<!-- Placeholder for "Used in" -->

The `TransformedTransitionKernel` `TransitionKernel` enables fitting
a [Bijector](
https://www.tensorflow.org/api_docs/python/tf/distributions/bijectors/Bijector)
which serves to decorrelate the Markov chain Monte Carlo (MCMC)
event dimensions thus making the chain mix faster. This is
particularly useful when the geometry of the target distribution is
unfavorable. In such cases it may take many evaluations of the
`target_log_prob_fn` for the chain to mix between faraway states.

The idea of training an affine function to decorrelate chain event dims was
presented in [Parno and Marzouk (2014)][1]. Used in conjunction with the
`HamiltonianMonteCarlo` `TransitionKernel`, the [Parno and Marzouk (2014)][1]
idea is an instance of Riemannian manifold HMC [(Girolami and Calderhead,
2011)][2].

The `TransformedTransitionKernel` enables arbitrary bijective transformations
of arbitrary `TransitionKernel`s, e.g., one could use bijectors
`tfp.distributions.bijectors.Affine`,
`tfp.distributions.bijectors.RealNVP`, etc. with transition kernels
<a href="../../tfp/mcmc/HamiltonianMonteCarlo.md"><code>tfp.mcmc.HamiltonianMonteCarlo</code></a>, <a href="../../tfp/mcmc/RandomWalkMetropolis.md"><code>tfp.mcmc.RandomWalkMetropolis</code></a>,
etc.

#### Examples

##### RealNVP + HamiltonianMonteCarlo

Note: this example is only meant to illustrate how to wire up a
`TransformedTransitionKernel`. As it is this won't work well because:
* a 1-layer RealNVP is a pretty weak density model, since it can't change the
density of the masked dimensions
* we're not actually training the bijector to do anything useful.

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

def make_likelihood(true_variances):
  return tfd.MultivariateNormalDiag(
      scale_diag=tf.sqrt(true_variances))

dims = 10
dtype = np.float32
true_variances = tf.linspace(dtype(1), dtype(3), dims)
likelihood = make_likelihood(true_variances)

realnvp_hmc = tfp.mcmc.TransformedTransitionKernel(
    inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=likelihood.log_prob,
      step_size=0.5,
      num_leapfrog_steps=2),
    bijector=tfb.RealNVP(
      num_masked=2,
      shift_and_log_scale_fn=tfb.real_nvp_default_template(
          hidden_layers=[512, 512])))

states, kernel_results = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=tf.zeros(dims),
    kernel=realnvp_hmc,
    num_burnin_steps=500)

# Compute sample stats.
sample_mean = tf.reduce_mean(states, axis=0)
sample_var = tf.reduce_mean(
    tf.squared_difference(states, sample_mean),
    axis=0)
```

#### References

[1]: Matthew Parno and Youssef Marzouk. Transport map accelerated Markov chain
     Monte Carlo. _arXiv preprint arXiv:1412.5492_, 2014.
     https://arxiv.org/abs/1412.5492

[2]: Mark Girolami and Ben Calderhead. Riemann manifold langevin and
     hamiltonian monte carlo methods. In _Journal of the Royal Statistical
     Society_, 2011. https://doi.org/10.1111/j.1467-9868.2010.00765.x

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    inner_kernel,
    bijector,
    name=None
)
```

Instantiates this object.


#### Args:


* <b>`inner_kernel`</b>: `TransitionKernel`-like object which has a
  `target_log_prob_fn` argument.
* <b>`bijector`</b>: `tfp.distributions.Bijector` or list of
  `tfp.distributions.Bijector`s. These bijectors use `forward` to map the
  `inner_kernel` state space to the state expected by
  `inner_kernel.target_log_prob_fn`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., "transformed_kernel").


#### Returns:


* <b>`transformed_kernel`</b>: Instance of `TransitionKernel` which copies the input
  transition kernel then modifies its `target_log_prob_fn` by applying the
  provided bijector(s).



## Properties

<h3 id="bijector"><code>bijector</code></h3>




<h3 id="inner_kernel"><code>inner_kernel</code></h3>




<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="name"><code>name</code></h3>




<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.




## Methods

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(
    init_state=None,
    transformed_init_state=None
)
```

Returns an object with the same type as returned by `one_step`.

Unlike other `TransitionKernel`s,
`TransformedTransitionKernel.bootstrap_results` has the option of
initializing the `TransformedTransitionKernelResults` from either an initial
state, eg, requiring computing `bijector.inverse(init_state)`, or
directly from `transformed_init_state`, i.e., a `Tensor` or list
of `Tensor`s which is interpretted as the `bijector.inverse`
transformed state.

#### Args:


* <b>`init_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the a
  state(s) of the Markov chain(s). Must specify `init_state` or
  `transformed_init_state` but not both.
* <b>`transformed_init_state`</b>: `Tensor` or Python `list` of `Tensor`s
  representing the a state(s) of the Markov chain(s). Must specify
  `init_state` or `transformed_init_state` but not both.


#### Returns:


* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
  `Tensor`s representing internal calculations made within this function.


#### Raises:


* <b>`ValueError`</b>: if `inner_kernel` results doesn't contain the member
  "target_log_prob".

#### Examples

To use `transformed_init_state` in context of
<a href="../../tfp/mcmc/sample_chain.md"><code>tfp.mcmc.sample_chain</code></a>, you need to explicitly pass the
`previous_kernel_results`, e.g.,

```python
transformed_kernel = tfp.mcmc.TransformedTransitionKernel(...)
init_state = ...        # Doesnt matter.
transformed_init_state = ... # Does matter.
results, _ = tfp.mcmc.sample_chain(
    num_results=...,
    current_state=init_state,
    previous_kernel_results=transformed_kernel.bootstrap_results(
        transformed_init_state=transformed_init_state),
    kernel=transformed_kernel)
```

<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Runs one iteration of the Transformed Kernel.


#### Args:


* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s
  representing the current state(s) of the Markov chain(s),
  _after_ application of `bijector.forward`. The first `r`
  dimensions index independent chains,
  `r = tf.rank(target_log_prob_fn(*current_state))`. The
  `inner_kernel.one_step` does not actually use `current_state`,
  rather it takes as input
  `previous_kernel_results.transformed_state` (because
  `TransformedTransitionKernel` creates a copy of the input
  inner_kernel with a modified `target_log_prob_fn` which
  internally applies the `bijector.forward`).
* <b>`previous_kernel_results`</b>: `collections.namedtuple` containing `Tensor`s
  representing values from previous calls to this function (or from the
  `bootstrap_results` function.)


#### Returns:


* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
  of the Markov chain(s) after taking exactly one step. Has same type and
  shape as `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.



