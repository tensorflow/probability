<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.mcmc.NoUTurnSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_calibrated"/>
<meta itemprop="property" content="parameters"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="bootstrap_results"/>
<meta itemprop="property" content="one_step"/>
</div>

# tfp.experimental.mcmc.NoUTurnSampler

## Class `NoUTurnSampler`

Runs one step of the No U-Turn Sampler.

Inherits From: [`TransitionKernel`](../../../tfp/mcmc/TransitionKernel.md)

### Aliases:

* Class `tfp.experimental.mcmc.NoUTurnSampler`
* Class `tfp.experimental.mcmc.nuts.NoUTurnSampler`



Defined in [`python/experimental/mcmc/nuts.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/experimental/mcmc/nuts.py).

<!-- Placeholder for "Used in" -->

The No U-Turn Sampler (NUTS) is an adaptive variant of the Hamiltonian Monte
Carlo (HMC) method for MCMC.  NUTS adapts the distance traveled in response to
the curvature of the target density.  Conceptually, one proposal consists of
reversibly evolving a trajectory through the sample space, continuing until
that trajectory turns back on itself (hence the name, "No U-Turn").  This
class implements one random NUTS step from a given
`current_state`.  Mathematical details and derivations can be found in
[Hoffman, Gelman (2011)][1].

The `one_step` function can update multiple chains in parallel. It assumes
that a prefix of leftmost dimensions of `current_state` index independent
chain states (and are therefore updated independently).  The output of
`target_log_prob_fn(*current_state)` should sum log-probabilities across all
event dimensions.  Slices along the rightmost dimensions may have different
target distributions; for example, `current_state[0][0, ...]` could have a
different target distribution from `current_state[0][1, ...]`.  These
semantics are governed by `target_log_prob_fn(*current_state)`. (The number of
independent chains is `tf.size(target_log_prob_fn(*current_state))`.)

TODO(axch): Examples (e.g., a la HMC).  For them to be sensible, need to
pick sensible step sizes, or implement step size adaptation, or both.

#### References

[1] Matthew D. Hoffman, Andrew Gelman.  The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.  2011.
https://arxiv.org/pdf/1111.4246.pdf.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    target_log_prob_fn,
    step_size,
    max_tree_depth=10,
    unrolled_leapfrog_steps=1,
    use_auto_batching=True,
    stackless=False,
    backend=None,
    seed=None,
    name=None
)
```

Initializes this transition kernel.


#### Args:


* <b>`target_log_prob_fn`</b>: Python callable which takes an argument like
  `current_state` (or `*current_state` if it's a list) and returns its
  (possibly unnormalized) log-density under the target distribution.  Due
  to limitations of the underlying auto-batching system,
  target_log_prob_fn may be invoked with junk data at some batch indexes,
  which it must process without crashing.  (The results at those indexes
  are ignored).
* <b>`step_size`</b>: `Tensor` or Python `list` of `Tensor`s representing the step
  size for the leapfrog integrator. Must broadcast with the shape of
  `current_state`. Larger step sizes lead to faster progress, but
  too-large step sizes make rejection exponentially more likely. When
  possible, it's often helpful to match per-variable step sizes to the
  standard deviations of the target distribution in each variable.
* <b>`max_tree_depth`</b>: Maximum depth of the tree implicitly built by NUTS. The
  maximum number of leapfrog steps is bounded by `2**max_tree_depth-1`
  i.e. the number of nodes in a binary tree `max_tree_depth` nodes deep.
  The default setting of 10 takes up to 1023 leapfrog steps.
* <b>`unrolled_leapfrog_steps`</b>: The number of leapfrogs to unroll per tree
  expansion step. Applies a direct linear multipler to the maximum
  trajectory length implied by max_tree_depth. Defaults to 1. This
  parameter can be useful for amortizing the auto-batching control flow
  overhead.
* <b>`use_auto_batching`</b>: Boolean.  If `False`, do not invoke the auto-batching
  system; operate on batch size 1 only.
* <b>`stackless`</b>: Boolean.  If `True`, invoke the stackless version of
  the auto-batching system.  Only works in Eager mode.
* <b>`backend`</b>: Auto-batching backend object. Falls back to a default
  TensorFlowBackend().
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'nuts_kernel').



## Properties

<h3 id="is_calibrated"><code>is_calibrated</code></h3>

Returns `True` if Markov chain converges to specified distribution.

`TransitionKernel`s which are "uncalibrated" are often calibrated by
composing them with the <a href="../../../tfp/mcmc/MetropolisHastings.md"><code>tfp.mcmc.MetropolisHastings</code></a> `TransitionKernel`.

<h3 id="parameters"><code>parameters</code></h3>






## Methods

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

Runs one iteration of the No U-Turn Sampler.


#### Args:


* <b>`current_state`</b>: `Tensor` or Python `list` of `Tensor`s representing the
  current state(s) of the Markov chain(s). The first `r` dimensions index
  independent chains, `r = tf.rank(target_log_prob_fn(*current_state))`.
* <b>`previous_kernel_results`</b>: `collections.namedtuple` containing `Tensor`s
  representing values from previous calls to this function (or from the
  `bootstrap_results` function.)


#### Returns:


* <b>`next_state`</b>: `Tensor` or Python list of `Tensor`s representing the state(s)
  of the Markov chain(s) after taking exactly one step. Has same type and
  shape as `current_state`.
* <b>`kernel_results`</b>: `collections.namedtuple` of internal calculations used to
  advance the chain.



