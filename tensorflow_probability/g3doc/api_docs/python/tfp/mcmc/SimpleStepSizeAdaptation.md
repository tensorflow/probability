<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.SimpleStepSizeAdaptation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="inner_kernel"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_adaptation_steps"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="log_accept_prob_getter_fn"/>
<meta itemprop="property" content="one_step"/>
<meta itemprop="property" content="step_size_getter_fn"/>
<meta itemprop="property" content="step_size_setter_fn"/>
</div>

# tfp.mcmc.SimpleStepSizeAdaptation

## Class `SimpleStepSizeAdaptation`

Adapts the inner kernel's `step_size` based on `log_accept_prob`.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)



Defined in [`python/mcmc/simple_step_size_adaptation.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/mcmc/simple_step_size_adaptation.py).

<!-- Placeholder for "Used in" -->

The simple policy multiplicatively increases or decreases the `step_size` of
the inner kernel based on the value of `log_accept_prob`. It is based on
[equation 19 of Andrieu and Thoms (2008)][1]. Given enough steps and small
enough `adaptation_rate` the median of the distribution of the acceptance
probability will converge to the `target_accept_prob`. A good target
acceptance probability depends on the inner kernel. If this kernel is
`HamiltonianMonteCarlo`, then 0.6-0.9 is a good range to aim for. For
`RandomWalkMetropolis` this should be closer to 0.25. See the individual
kernels' docstrings for guidance.

In general, adaptation prevents the chain from reaching a stationary
distribution, so obtaining consistent samples requires `num_adaptation_steps`
be set to a value [somewhat smaller][2] than the number of burnin steps.
However, it may sometimes be helpful to set `num_adaptation_steps` to a larger
value during development in order to inspect the behavior of the chain during
adaptation.

The step size is assumed to broadcast with the chain state, potentially having
leading dimensions corresponding to multiple chains. When there are fewer of
those leading dimensions than there are chain dimensions, the corresponding
dimensions in the `log_accept_prob` are averaged (in the direct space, rather
than the log space) before being used to adjust the step size. This means that
this kernel can do both cross-chain adaptation, or per-chain step size
adaptation, depending on the shape of the step size.

For example, if your problem has a state with shape `[S]`, your chain state
has shape `[C0, C1, Y]` (meaning that there are `C0 * C1` total chains) and
`log_accept_prob` has shape `[C0, C1]` (one acceptance probability per chain),
then depending on the shape of the step size, the following will happen:

- Step size has shape [], [S] or [1], the `log_accept_prob` will be averaged
  across its `C0` and `C1` dimensions. This means that you will learn a shared
  step size based on the mean acceptance probability across all chains. This
  can be useful if you don't have a lot of steps to adapt and want to average
  away the noise.

- Step size has shape [C1, 1] or [C1, S], the `log_accept_prob` will be
  averaged across its `C0` dimension. This means that you will learn a shared
  step size based on the mean acceptance probability across chains that share
  the coordinate across the `C1` dimension. This can be useful when the `C1`
  dimension indexes different distributions, while `C0` indexes replicas of a
  single distribution, all sampled in parallel.

- Step size has shape [C0, C1, 1] or [C0, C1, S], then no averaging will
  happen. This means that each chain will learn its own step size. This can be
  useful when all chains are sampling from different distributions. Even when
  all chains are for the same distribution, this can help during the initial
  warmup period.

- Step size has shape [C0, 1, 1] or [C0, 1, S], the `log_accept_prob` will be
  averaged across its `C1` dimension. This means that you will learn a shared
  step size based on the mean acceptance probability across chains that share
  the coordinate across the `C0` dimension. This can be useful when the `C0`
  dimension indexes different distributions, while `C1` indexes replicas of a
  single distribution, all sampled in parallel.

#### Examples

```python
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

target_log_prob_fn = tfd.Normal(loc=0., scale=1.).log_prob
num_burnin_steps = 500
num_results = 500
num_chains = 64
step_size = 0.1
# Or, if you want per-chain step size:
# step_size = tf.fill([num_chains], step_size)

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=target_log_prob_fn,
    num_leapfrog_steps=2,
    step_size=step_size)
kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    inner_kernel=kernel, num_adaptation_steps=int(num_burnin_steps * 0.8))

# The chain will be stepped for num_results + num_burnin_steps, adapting for
# the first num_adaptation_steps.
samples, [step_size, log_accept_ratio] = tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    current_state=tf.zeros(num_chains),
    kernel=kernel,
    trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                             pkr.inner_results.log_accept_ratio])

# ~0.75
p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))
```

#### References

[1]: Andrieu, Christophe, Thoms, Johannes. A tutorial on adaptive MCMC.
     _Statistics and Computing_, 2008.
     https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf

[2]:
http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    inner_kernel,
    num_adaptation_steps,
    target_accept_prob=0.75,
    adaptation_rate=0.01,
    step_size_setter_fn=_hmc_like_step_size_setter_fn,
    step_size_getter_fn=_hmc_like_step_size_getter_fn,
    log_accept_prob_getter_fn=_hmc_like_log_accept_prob_getter_fn,
    validate_args=False,
    name=None
)
```

Creates the step size adaptation kernel.

The default setter_fn and the getter_fn callbacks assume that the inner
kernel produces kernel results structurally the same as the
`HamiltonianMonteCarlo` kernel.

#### Args:

  inner_kernel: `TransitionKernel`-like object.
  num_adaptation_steps: Scalar `int` `Tensor` number of initial steps to
    during which to adjust the step size. This may be greater, less than, or
    equal to the number of burnin steps.
  target_accept_prob: A floating point `Tensor` representing desired
    acceptance probability. Must be a positive number less than 1. This can
    either be a scalar, or have shape [num_chains]. Default value: `0.75`
    (the [center of asymptotically optimal rate for HMC][1]).
  adaptation_rate: `Tensor` representing amount to scale the current
    `step_size`.
  step_size_setter_fn: A callable with the signature
    `(kernel_results, new_step_size) -> new_kernel_results` where
    `kernel_results` are the results of the `inner_kernel`, `new_step_size`
    is a `Tensor` or a nested collection of `Tensor`s with the same
    structure as returned by the `step_size_getter_fn`, and
    `new_kernel_results` are a copy of `kernel_results` with the step
    size(s) set.
  step_size_getter_fn: A callable with the signature
    `(kernel_results) -> step_size` where `kernel_results` are the results
    of the `inner_kernel`, and `step_size` is a floating point `Tensor` or a
    nested collection of such `Tensor`s.
  log_accept_prob_getter_fn: A callable with the signature
    `(kernel_results) -> log_accept_prob` where `kernel_results` are the
    results of the `inner_kernel`, and `log_accept_prob` is a floating point
    `Tensor`. `log_accept_prob` can either be a scalar, or have shape
    [num_chains]. If it's the latter, `step_size` should also have the same
    leading dimension.
  validate_args: Python `bool`. When `True` kernel parameters are checked
    for validity. When `False` invalid inputs may silently render incorrect
    outputs.
  name: Python `str` name prefixed to Ops created by this class. Default:
    'simple_step_size_adaptation'.

#### References

[1]: Betancourt, M. J., Byrne, S., & Girolami, M. (2014). _Optimizing The
     Integrator Step Size for Hamiltonian Monte Carlo_.
     http://arxiv.org/abs/1411.6669



## Properties

<h3 id="inner_kernel"><code>inner_kernel</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="num_adaptation_steps"><code>num_adaptation_steps</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.



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

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

``` python
is_calibrated()
```

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="log_accept_prob_getter_fn"><code>log_accept_prob_getter_fn</code></h3>

``` python
log_accept_prob_getter_fn(kernel_results)
```



<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Takes one step of the TransitionKernel.

Must be overridden by subclasses.

#### Args:

* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s).
* <b>`previous_kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or
  `list` of `Tensor`s representing internal calculations made within the
  previous call to this function (or as returned by `bootstrap_results`).


#### Returns:

* <b>`next_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  next state(s) of the Markov chain(s).
* <b>`kernel_results`</b>: A (possibly nested) `tuple`, `namedtuple` or `list` of
  `Tensor`s representing internal calculations made within this function.

<h3 id="step_size_getter_fn"><code>step_size_getter_fn</code></h3>

``` python
step_size_getter_fn(kernel_results)
```



<h3 id="step_size_setter_fn"><code>step_size_setter_fn</code></h3>

``` python
step_size_setter_fn(
    kernel_results,
    new_step_size
)
```





