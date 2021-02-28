# PRNGs and seeds

_Current as of 9/28/2020_

TFP supports a couple different forms of pseudo-random number generators, and
it's worth explaining/understanding these. To do so requires some TF history.

*   __Stateful samplers, Graph mode__

    TensorFlow started the random sampler suite with `tf.random.normal` and
    friends. These take a single Python `int` seed argument, as well as looking
    at the "global" (graph level) seed, both of which become attrs of the TF
    graph node. A dedicated C++ kernel object is initialized for each TF graph
    node, and each time that kernel object is executed, it advances an internal
    PRNG state. Hence the origin of the term "stateful". The graph-level seed is
    specified by calling `tf.random.set_seed`.

    ```python
    @tf.function
    def f():
      r1 = tfd.Normal(0., 1.).sample(seed=234)
      r2 = tfd.Normal(0., 1.).sample(seed=234)
      # `r2` is the same as `r1`.
      # op is stateful, but separate graph nodes =>
      #                     dedicated C++ kernel instances =>
      #                     dedicated state per op.
      return r1, r2
    tf.random.set_seed(123)
    r1, r2 = f()
    assert r1 == r2

    run_mcmc = lambda: tfp.mcmc.sample_chain(
        num_results=10,
        current_state=0.,
        kernel=tfp.mcmc.RandomWalkMetropolis(tfd.Normal(0., 1.).log_prob),
        trace_fn=None,
        seed=234)

    run2 = tf.function(lambda: (run_mcmc(), run_mcmc()))
    m1, m2 = run2()
    assert np.all(m1 == m2)
    ```

*   __Stateful samplers, Eager mode__

    To roughly preserve the behavior of graph mode, but execute ops eagerly, the
    TF eager context object maintains a cache of C++ PRNG kernels keyed on some
    of the arguments, most notably including the `int` `seed` argument. When the
    user calls `tf.random.set_seed`, this cache is purged. Thus, to get
    reproducible randomness in eager requires a call to `tf.random.set_seed`.
    But `tf.random.set_seed` has different effects depending on whether in or
    out of a `tf.function` body (indeed, it seems to have very little effect
    inside a function body, though calling it from a `tf.py_function` can yield
    results).

    ```python
    tfd = tfp.distributions
    def f():
      r1 = tfd.Normal(0., 1.).sample(seed=234)
      # `r2` different from `r1` (stateful, non-`tf.function`)
      r2 = tfd.Normal(0., 1.).sample(seed=234)
      return r1, r2

    tf.random.set_seed(123)
    r1, r2 = f()
    assert r1 != r2

    tf.random.set_seed(123)
    r3, _ = f()
    assert r1 == r3

    run_mcmc = lambda: tfp.mcmc.sample_chain(
        num_results=10,
        current_state=0.,
        kernel=tfp.mcmc.RandomWalkMetropolis(tfd.Normal(0., 1.).log_prob),
        trace_fn=None,
        seed=234)
    run2 = lambda: (run_mcmc(), run_mcmc())
    tf.random.set_seed(123)
    m1, m2 = run2()
    assert np.all(m1 != m2)  # different from `m1` (stateful, non-`tf.function`)
    tf.random.set_seed(123)
    m3, _ = run2()  # same as `m1`
    assert np.all(m1 == m3)
    ```

*   __Stateful samplers, XLA__

    Under XLA, it has never been possible to get reproducible results using TF's
    stateful samplers, even when specifying a seed.

*   __`tf.random.Generator`, TF2__

    In TF2, a new concept was introduced for reproducible stateful randomness.
    The `Generator` maintains a `tf.Variable` instance internally, and calls
    sampler kernels which update this `Variable` and produce random results.

    TFP has not been able to embrace this paradigm, in part because the
    `Generator` must be provided by the end user in order to work well in all
    scenarios, e.g. under `tf.function` wrapping.

*   __Stateless samplers__

    To support use cases like dropout with gradient checkpointing in long
    recurrent models, TF introduced reproducible, "stateless" (pure functional)
    samplers, including `tf.random.stateless_normal` & friends. These became
    more important when the disparities between graph and eager PRNGs became
    evident in TF2 (see above). These are deterministic ops whose behavior is
    controlled by a `int32` `Tensor` `seed` of shape `[2]`. Notably, the seed
    argument can be controlled by TF graph ops such as the carried state of a
    while loop or a `tf.function` input. TFP added support for this type of seed
    in early 2020 throughout the `distributions` package and in the `mcmc`
    package. Note that this form never requires use of `tf.random.set_seed`. All
    random state is explicitly provided by the user as an argument to the
    sampling op.

    ```python
    tfd = tfp.distributions
    r1 = tfd.Normal(0., 1.).sample(seed=(1, 2))
    r2 = tfd.Normal(0., 1.).sample(seed=tf.constant([1, 2], tf.int32))
    assert r1 == r2

    run_mcmc = lambda: tfp.mcmc.sample_chain(
        num_results=10,
        current_state=0.,
        kernel=tfp.mcmc.RandomWalkMetropolis(tfd.Normal(0., 1.).log_prob),
        trace_fn=None,
        seed=(1, 2))
    m1 = run_mcmc()
    m2 = run_mcmc()
    m3 = tf.function(run_mcmc)()  # reproducible in both graph and eager modes.
    assert np.all(m1 == m2)
    assert np.all(m1 == m3)
    ```

*   __JAX: Functional purity__

    JAX supports only the stateless variety of random sampling<sup>\*</sup>,
    e.g. `jax.random.normal(key=jax.random.PRNGKey(1), shape=[])`. This is
    easily mapped to align with the TF stateless code paths, see
    [`SUBSTRATES.md`](SUBSTRATES.md) for more information about TFP on JAX.

    <sub><sup>\* Actually, JAX also supports XLA's seedless stateful samplers
    for top-end performance reasons, but TFP on JAX does not use
    these.</sup></sub>

    ```python
    tfp_mcmc = tfp.substrates.jax.mcmc
    tfd = tfp.substrates.jax.distributions
    r1 = tfd.Normal(0., 1.).sample(seed=jax.random.PRNGKey(1))
    r2 = tfd.Normal(0., 1.).sample(seed=jax.random.PRNGKey(1))
    assert r1 == r2

    run_mcmc = lambda: tfp_mcmc.sample_chain(
        num_results=10,
        current_state=0.,
        kernel=tfp_mcmc.RandomWalkMetropolis(tfd.Normal(0., 1.).log_prob),
        trace_fn=None,
        seed=jax.random.PRNGKey(1))

    m1 = run_mcmc()
    m2 = jax.jit(run_mcmc)()
    assert np.all(m1 == m2)
    ```

## Internal implementation details, or _How it works in TFP_.

So how does this all work under the hood in TFP? What do you as a contributor
need to know? TFP uses an internal library `samplers.py` to mediate most calls
to PRNGs. Here we will discuss the functions provided, and their behaviors.

Import: `from tensorflow_probability.python.internal import samplers`

*   `samplers.sanitize_seed(seed, salt=None, name=None)`

    This function ensures that a seed of any flavor becomes a
    "stateless-compatible" seed, i.e. an `int32[2]` `Tensor`. (In JAX, it
    matches the result type of `jax.random.PRNGKey(0)`, currently `uint32[2]`.)

    If the seed is an `int` or `None`, we use `tf.random.uniform` to
    *statefully* draw a pair of unbounded `int32`s. (This scenario is not
    permitted in TFP on JAX, where seeds are always required.)

    If the seed is already a stateless-compatible seed, or is a list or tuple
    with two `int`s, we will directly convert that to a `Tensor`.

    The seed can optionally be salted so that the randomness behavior of one
    function is uncorrelated to that of another. However, salting is generally
    only necessary "at the front door", since stateless seeds require the
    discipline of splitting, which ensures no two downstream consumer functions
    receive the same seed value.

*   `samplers.split_seed(seed, n=2, salt=None, name=None)`

    This function splits a single seed into `n` stateless-compatible seeds. The
    input seed will first be passed to `sanitize_seed`, which avoids
    daisy-chains like `split_seed(sanitize_seed(seed))`. Generally, the result
    is "unstacked" into a tuple of `n` separate seeds. However, if `n` is a
    `Tensor`, we will return the stacked result. (This function was inspired by
    `jax.random.split`, and in TFP on JAX will call `jax.random.split`.)

*   `samplers.normal`, `samplers.uniform`, `samplers.shuffle`,
    `samplers.categorical`

    These random sampling functions generally call `sanitize_seed` first, then
    delegate to an underlying `tf.random.stateless_*` sampler. The exact
    implementation details are subject to change, but the basic contract is
    "`int`/`None` seed => stateful sampling, `Tensor` seed => stateless
    sampling".

### Usage

As a first, relatively simple example, let's take a look at usage by
[`beta_binomial.py`](https://cs.opensource.google/tensorflow/probability/+/master:tensorflow_probability/python/distributions/beta_binomial.py;l=241?q=beta_binomial%20sample_n).
In the `_sample_n` function for `BetaBinomial`, we can see that the seed is
split three ways. Typically, we give each subsidiary seed a local variable name
reflecting the downstream usage. In this case, two seeds are passed to a
log-space gamma rejection sampler, and the third seed is passed to the binomial
sampler. In some cases these downstream consumers will further split the seeds,
e.g. for each iteration of a rejection sampling loop.

```python
def _sample_n(self, n, seed=None):
  gamma1_seed, gamma2_seed, binomial_seed = samplers.split_seed(
      seed, n=3, salt='beta_binomial')
  log_gamma1 = gamma_lib.random_gamma(
      shape=[n], concentration=self.concentration1, seed=gamma1_seed,
      log_space=True)
  log_gamma2 = gamma_lib.random_gamma(
      shape=[n], concentration=self.concentration0, seed=gamma2_seed,
      log_space=True)
  return binomial.Binomial(
      self.total_count, logits=log_gamma1 - log_gamma2,
      validate_args=self.validate_args).sample(seed=binomial_seed)
```

As a more complex example of seed splitting in a loop context, we can look at
[`hidden_markov_model.py`](https://cs.opensource.google/tensorflow/probability/+/master:tensorflow_probability/python/distributions/hidden_markov_model.py;l=283).
Here we see an initial split into 3 parts:

```python
init_seed, scan_seed, observation_seed = samplers.split_seed(
    seed, n=3, salt='HiddenMarkovModel')
```

followed by a scan (loop) with body function `generate_step`:

```python
def generate_step(state_and_seed, _):
  """Take a single step in Markov chain."""
  state, seed = state_and_seed
  sample_seed, next_seed = samplers.split_seed(seed)
  gen = self._transition_distribution.sample(n * transition_repeat,
                                             seed=sample_seed)
  ...
  return result, next_seed
```

Note that the initial carried state of the scan is the `scan_seed` split off by
`_sample_n` up above.

```python
hidden_states, _ = tf.scan(generate_step, dummy_index,
                           initializer=(init_state, scan_seed))
```

Similar discipline exists in the body `tfp.mcmc.sample_chain` to take care of
seed splitting and keep underlying `TransitionKernel` subclasses relatively
simple. `TransitionKernel` takes a `seed` argument to `one_step` and can use
this to drive randomness, in some cases splitting and passing a separate seed to
inner kernels. For example, see `tfp.mcmc.MetropolisHastings`
[`one_step`](https://cs.opensource.google/tensorflow/probability/+/master:tensorflow_probability/python/mcmc/metropolis_hastings.py;l=203?q=MetropolisHastings&ss=tensorflow%2Fprobability).
