<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.UncalibratedHamiltonianMonteCarlo" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="num_leapfrog_steps"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="seed"/>
<meta itemprop="property" content="state_gradients_are_stopped"/>
<meta itemprop="property" content="step_size"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.UncalibratedHamiltonianMonteCarlo

## Class `UncalibratedHamiltonianMonteCarlo`

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)

Runs one step of Uncalibrated Hamiltonian Monte Carlo.

Warning: this kernel will not result in a chain which converges to the
`target_log_prob`. To get a convergent MCMC, use `HamiltonianMonteCarlo(...)`
or `MetropolisHastings(UncalibratedHamiltonianMonteCarlo(...))`.

For more details on `UncalibratedHamiltonianMonteCarlo`, see
`HamiltonianMonteCarlo`.

## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>



<h3 id="name"><code>name</code></h3>



<h3 id="num_leapfrog_steps"><code>num_leapfrog_steps</code></h3>



<h3 id="parameters"><code>parameters</code></h3>

Return `dict` of ``__init__`` arguments and their values.

<h3 id="seed"><code>seed</code></h3>



<h3 id="state_gradients_are_stopped"><code>state_gradients_are_stopped</code></h3>



<h3 id="step_size"><code>step_size</code></h3>



<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>





## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    num_leapfrog_steps,
    state_gradients_are_stopped=False,
    seed=None,
    name=None
)
```

Initializes this transition kernel.

#### Args:

* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
    `current_state` (or `*current_state` if it's a list) and returns its
    (possibly unnormalized) log-density under the target distribution.
* <b>`step_size`</b>: `Tensor` or Python `list` of `Tensor`s representing the step
    size for the leapfrog integrator. Must broadcast with the shape of
    `current_state`. Larger step sizes lead to faster progress, but
    too-large step sizes make rejection exponentially more likely. When
    possible, it's often helpful to match per-variable step sizes to the
    standard deviations of the target distribution in each variable.
* <b>`num_leapfrog_steps`</b>: Integer number of steps to run the leapfrog integrator
    for. Total progress per HMC step is roughly proportional to
    `step_size * num_leapfrog_steps`.
* <b>`state_gradients_are_stopped`</b>: Python `bool` indicating that the proposed
    new state be run through `tf.stop_gradient`. This is particularly useful
    when combining optimization over samples from the HMC chain.
    Default value: `False` (i.e., do not apply `stop_gradient`).
* <b>`step_size_update_fn`</b>: Python `callable` taking current `step_size`
    (typically a `tf.Variable`) and `kernel_results` (typically
    `collections.namedtuple`) and returns updated step_size (`Tensor`s).
    Default value: `None` (i.e., do not update `step_size` automatically).
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
    Default value: `None` (i.e., 'hmc_kernel').


#### Returns:

* <b>`next_state`</b>: Tensor or Python list of `Tensor`s representing the state(s)
    of the Markov chain(s) at each result step. Has same shape as
    `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
    advance the chain.


#### Raises:

* <b>`ValueError`</b>: if there isn't one `step_size` or a list with same length as
    `current_state`.

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

``` python
bootstrap_results(init_state)
```

Creates initial `previous_kernel_results` using a supplied `state`.

<h3 id="one_step"><code>one_step</code></h3>

``` python
one_step(
    current_state,
    previous_kernel_results
)
```

Runs one iteration of Hamiltonian Monte Carlo.

#### Args:

* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
    current state(s) of the Markov chain(s). The first `r` dimensions index
    independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
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



