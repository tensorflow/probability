<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.NoUTurnSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="max_energy_diff"/>
<meta itemprop="property" content="max_tree_depth"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="parallel_iterations"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="read_instruction"/>
<meta itemprop="property" content="step_size"/>
<meta itemprop="property" content="target_log_prob_fn"/>
<meta itemprop="property" content="unrolled_leapfrog_steps"/>
<meta itemprop="property" content="write_instruction"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="loop_tree_doubling"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.mcmc.NoUTurnSampler


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/mcmc/nuts.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `NoUTurnSampler`

Runs one step of the No U-Turn Sampler.

Inherits From: [`TransitionKernel`](../../tfp/mcmc/TransitionKernel.md)

<!-- Placeholder for "Used in" -->

The No U-Turn Sampler (NUTS) is an adaptive variant of the Hamiltonian Monte
Carlo (HMC) method for MCMC. NUTS adapts the distance traveled in response to
the curvature of the target density. Conceptually, one proposal consists of
reversibly evolving a trajectory through the sample space, continuing until
that trajectory turns back on itself (hence the name, 'No U-Turn'). This class
implements one random NUTS step from a given `current_state`.
Mathematical details and derivations can be found in
[Hoffman, Gelman (2011)][1] and [Betancourt (2018)][2].

The `one_step` function can update multiple chains in parallel. It assumes
that a prefix of leftmost dimensions of `current_state` index independent
chain states (and are therefore updated independently).  The output of
`target_log_prob_fn(*current_state)` should sum log-probabilities across all
event dimensions.  Slices along the rightmost dimensions may have different
target distributions; for example, `current_state[0][0, ...]` could have a
different target distribution from `current_state[0][1, ...]`.  These
semantics are governed by `target_log_prob_fn(*current_state)`. (The number of
independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

#### References

[1]: Matthew D. Hoffman, Andrew Gelman.  The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.  2011.
https://arxiv.org/pdf/1111.4246.pdf.

[2]: Michael Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo.
_arXiv preprint arXiv:1701.02434_, 2018. https://arxiv.org/abs/1701.02434

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/mcmc/nuts.py">View source</a>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    max_tree_depth=10,
    max_energy_diff=1000.0,
    unrolled_leapfrog_steps=1,
    parallel_iterations=10,
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
* <b>`max_tree_depth`</b>: Maximum depth of the tree implicitly built by NUTS. The
  maximum number of leapfrog steps is bounded by `2**max_tree_depth` i.e.
  the number of nodes in a binary tree `max_tree_depth` nodes deep. The
  default setting of 10 takes up to 1024 leapfrog steps.
* <b>`max_energy_diff`</b>: Scaler threshold of energy differences at each leapfrog,
  divergence samples are defined as leapfrog steps that exceed this
  threshold. Default to 1000.
* <b>`unrolled_leapfrog_steps`</b>: The number of leapfrogs to unroll per tree
  expansion step. Applies a direct linear multipler to the maximum
  trajectory length implied by max_tree_depth. Defaults to 1.
* <b>`parallel_iterations`</b>: The number of iterations allowed to run in parallel.
  It must be a positive integer. See `tf.while_loop` for more details.
  Note that if you set the seed to have deterministic output you should
  also set `parallel_iterations` to 1.
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'nuts_kernel').



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="max_energy_diff"><code>max_energy_diff</code></h3>




<h3 id="max_tree_depth"><code>max_tree_depth</code></h3>




<h3 id="name"><code>name</code></h3>




<h3 id="parallel_iterations"><code>parallel_iterations</code></h3>




<h3 id="parameters"><code>parameters</code></h3>




<h3 id="read_instruction"><code>read_instruction</code></h3>




<h3 id="step_size"><code>step_size</code></h3>




<h3 id="target_log_prob_fn"><code>target_log_prob_fn</code></h3>




<h3 id="unrolled_leapfrog_steps"><code>unrolled_leapfrog_steps</code></h3>




<h3 id="write_instruction"><code>write_instruction</code></h3>






## Methods

<h3 id="bootstrap_results"><code>bootstrap_results</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/mcmc/nuts.py">View source</a>

``` python
bootstrap_results(init_state)
```

Creates initial `previous_kernel_results` using a supplied `state`.


<h3 id="loop_tree_doubling"><code>loop_tree_doubling</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/mcmc/nuts.py">View source</a>

``` python
loop_tree_doubling(
    step_size,
    momentum_state_memory,
    current_step_meta_info,
    iter_,
    initial_step_state,
    initial_step_metastate
)
```

Main loop for tree doubling.


<h3 id="one_step"><code>one_step</code></h3>

<a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/mcmc/nuts.py">View source</a>

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



