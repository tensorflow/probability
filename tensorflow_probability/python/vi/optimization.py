# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utilities for fitting variational distributions."""

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.math.minimize import minimize
from tensorflow_probability.python.math.minimize import minimize_stateless
from tensorflow_probability.python.vi import csiszar_divergence


_trace_loss = lambda traceable_quantities: traceable_quantities.loss


def fit_surrogate_posterior_stateless(
    target_log_prob_fn,
    build_surrogate_posterior_fn,
    initial_parameters,
    optimizer,
    num_steps,
    convergence_criterion=None,
    trace_fn=_trace_loss,
    discrepancy_fn=csiszar_divergence.kl_reverse,
    sample_size=1,
    importance_sample_size=1,
    gradient_estimator=csiszar_divergence.GradientEstimators.REPARAMETERIZATION,
    jit_compile=False,
    seed=None,
    name='fit_surrogate_posterior'):
  """Fit a surrogate posterior to a target (unnormalized) log density.

  The default behavior constructs and minimizes the negative variational
  evidence lower bound (ELBO), given by

  ```python
  q_samples = surrogate_posterior.sample(num_draws)
  elbo_loss = -tf.reduce_mean(
    target_log_prob_fn(q_samples) - surrogate_posterior.log_prob(q_samples))
  ```

  This corresponds to minimizing the 'reverse' Kullback-Liebler divergence
  (`KL[q||p]`) between the variational distribution and the unnormalized
  `target_log_prob_fn`, and  defines a lower bound on the marginal log
  likelihood, `log p(x) >= -elbo_loss`. [1]

  More generally, this function supports fitting variational distributions that
  minimize any
  [Csiszar f-divergence](https://en.wikipedia.org/wiki/F-divergence).

  Args:
    target_log_prob_fn: Python callable that takes a set of `Tensor` arguments
      and returns a `Tensor` log-density. Given
      `q_sample = surrogate_posterior.sample(sample_size)`, this
      will be called as `target_log_prob_fn(*q_sample)` if `q_sample` is a list
      or a tuple, `target_log_prob_fn(**q_sample)` if `q_sample` is a
      dictionary, or `target_log_prob_fn(q_sample)` if `q_sample` is a `Tensor`.
      It should support batched evaluation, i.e., should return a result of
      shape `[sample_size]`.
    build_surrogate_posterior_fn: Python `callable` that takes parameter values
      and returns an instance of `tfd.Distribution`.
    initial_parameters: List or tuple of initial parameter values (`Tensor`s or
      structures of `Tensor`s), passed as positional arguments to
      `build_surrogate_posterior_fn`.
    optimizer: Pure functional optimizer to use. This may be an
      `optax.GradientTransformation` instance (in JAX), or any similar object
      that implements methods
      `optimizer_state = optimizer.init(parameters)` and
      `updates, optimizer_state = optimizer.update(grads, optimizer_state,
      parameters)`.
    num_steps: Python `int` number of steps to run the optimizer.
    convergence_criterion: Optional instance of
      `tfp.optimizer.convergence_criteria.ConvergenceCriterion`
      representing a criterion for detecting convergence. If `None`,
      the optimization will run for `num_steps` steps, otherwise, it will run
      for at *most* `num_steps` steps, as determined by the provided criterion.
      Default value: `None`.
    trace_fn: Python callable with signature `traced_values = trace_fn(
      traceable_quantities)`, where the argument is an instance of
      `tfp.math.MinimizeTraceableQuantities` and the returned `traced_values`
      may be a `Tensor` or nested structure of `Tensor`s. The traced values are
      stacked across steps and returned.
      The default `trace_fn` simply returns the loss. In general, trace
      functions may also examine the gradients, values of parameters,
      the state propagated by the specified `convergence_criterion`, if any (if
      no convergence criterion is specified, this will be `None`).
      Default value: `lambda traceable_quantities: traceable_quantities.loss`.
    discrepancy_fn: Python `callable` representing a Csiszar `f` function in
      in log-space. See the docs for `tfp.vi.monte_carlo_variational_loss` for
      examples.
      Default value: `tfp.vi.kl_reverse`.
    sample_size: Python `int` number of Monte Carlo samples to use
      in estimating the variational divergence. Larger values may stabilize
      the optimization, but at higher cost per step in time and memory.
      Default value: `1`.
    importance_sample_size: Python `int` number of terms used to define an
      importance-weighted divergence. If `importance_sample_size > 1`, then the
      `surrogate_posterior` is optimized to function as an importance-sampling
      proposal distribution. In this case, posterior expectations should be
      approximated by importance sampling, as demonstrated in the example below.
      Default value: `1`.
    gradient_estimator: Optional element from `tfp.vi.GradientEstimators`
      specifying the stochastic gradient estimator to associate with the
      variational loss.
      Default value: `csiszar_divergence.GradientEstimators.REPARAMETERIZATION`.
    jit_compile: If True, compiles the loss function and gradient update using
      XLA. XLA performs compiler optimizations, such as fusion, and attempts to
      emit more efficient code. This may drastically improve the performance.
      See the docs for `tf.function`. (In JAX, this will apply `jax.jit`).
      Default value: `False`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to ops created by this function.
      Default value: 'fit_surrogate_posterior'.
  Returns:
    optimized_parameters: Tuple of optimized parameter values, with the same
      structure and `Tensor` shapes as `initial_parameters`.
    results: `Tensor` or nested structure of `Tensor`s, according to the return
      type of `trace_fn`. Each `Tensor` has an added leading dimension of size
      `num_steps`, packing the trajectory of the result over the course of the
      optimization.

  #### Examples

  **Normal-Normal model**. We'll first consider a simple model
  `z ~ N(0, 1)`, `x ~ N(z, 1)`, where we suppose we are interested in the
  posterior `p(z | x=5)`:

  ```python
  import tensorflow_probability as tfp
  from tensorflow_probability import distributions as tfd

  def log_prob(z, x):
    return tfd.Normal(0., 1.).log_prob(z) + tfd.Normal(z, 1.).log_prob(x)
  conditioned_log_prob = lambda z: log_prob(z, x=5.)
  ```

  The posterior is itself normal by [conjugacy](
  https://en.wikipedia.org/wiki/Conjugate_prior), and can be computed
  analytically (it's `N(loc=5/2., scale=1/sqrt(2)`). But suppose we don't want
  to bother doing the math: we can use variational inference instead!

  ```python
  import optax  # Requires JAX backend.
  init_normal, build_normal = tfp.experimental.util.make_trainable_stateless(
    tfd.Normal, name='q_z')
  optimized_parameters, losses = tfp.vi.fit_surrogate_posterior_stateless(
      conditioned_log_prob,
      build_surrogate_posterior_fn=build_normal,
      initial_parameters=init_normal(seed=(42, 42)),
      optimizer=optax.adam(learning_rate=0.1),
      num_steps=100,
      seed=(42, 42))
  q_z = build_normal(*optimized_parameters)
  ```

  **Custom loss function**. Suppose we prefer to fit the same model using
    the forward KL divergence `KL[p||q]`. We can pass a custom discrepancy
    function:

  ```python
  optimized_parameters, losses = tfp.vi.fit_surrogate_posterior_stateless(
      conditioned_log_prob,
      build_surrogate_posterior_fn=build_normal,
      initial_parameters=init_normal(seed=(42, 42)),
      optimizer=optax.adam(learning_rate=0.1),
      num_steps=100,
      seed=(42, 42),
      discrepancy_fn=tfp.vi.kl_forward)
  q_z = build_normal(*optimized_parameters)
  ```

  Note that in practice this may have substantially higher-variance gradients
  than the reverse KL.

  **Importance weighting**. A surrogate posterior may be corrected by
  interpreting it as a proposal for an [importance sampler](
  https://en.wikipedia.org/wiki/Importance_sampling). That is, one can use
  weighted samples from the surrogate to estimate expectations under the true
  posterior:

  ```python
  zs, q_log_prob = surrogate_posterior.experimental_sample_and_log_prob(
    num_samples, seed=(42, 42))

  # Naive expectation under the surrogate posterior.
  expected_x = tf.reduce_mean(f(zs), axis=0)

  # Importance-weighted estimate of the expectation under the true posterior.
  self_normalized_log_weights = tf.nn.log_softmax(
    target_log_prob_fn(zs) - q_log_prob)
  expected_x = tf.reduce_sum(
    tf.exp(self_normalized_log_weights) * f(zs),
    axis=0)
  ```

  Any distribution may be used as a proposal, but it is often natural to
  consider surrogates that were themselves fit by optimizing an
  importance-weighted variational objective [2], which directly optimizes the
  surrogate's effectiveness as an proposal distribution. This may be specified
  by passing `importance_sample_size > 1`. The importance-weighted objective
  may favor different characteristics than the original objective.
  For example, effective proposals are generally overdispersed, whereas a
  surrogate optimizing reverse KL would otherwise tend to be underdispersed.

  Although importance sampling is guaranteed to tighten the variational bound,
  some research has found that this does not necessarily improve the quality
  of deep generative models, because it also introduces gradient noise that can
  lead to a weaker training signal [3]. As always, evaluation is important to
  choose the approach that works best for a particular task.

  When using an importance-weighted loss to fit a surrogate, it is also
  recommended to apply importance sampling when computing expectations
  under that surrogate.

  ```python
  # Fit `q` with an importance-weighted variational loss.
  optimized_parameters, losses = tfp.vi.fit_surrogate_posterior_stateless(
        conditioned_log_prob,
        build_surrogate_posterior_fn=build_normal,
        initial_parameters=init_normal(seed=(42, 42)),
        importance_sample_size=10,
        optimizer=optax.adam(0.1),
        num_steps=200,
        seed=(42, 42))
  q_z = build_normal(*optimized_parameters)

  # Estimate posterior statistics with importance sampling.
  zs, q_log_prob = q_z.experimental_sample_and_log_prob(1000, seed=(42, 42))
  self_normalized_log_weights = tf.nn.log_softmax(
    conditioned_log_prob(zs) - q_log_prob)
  posterior_mean = tf.reduce_sum(
    tf.exp(self_normalized_log_weights) * zs,
    axis=0)
  posterior_variance = tf.reduce_sum(
    tf.exp(self_normalized_log_weights) * (zs - posterior_mean)**2,
    axis=0)
  ```

  **Inhomogeneous Poisson Process**. For a more interesting example, let's
  consider a model with multiple latent variables as well as trainable
  parameters in the model itself. Given observed counts `y` from spatial
  locations `X`, consider an inhomogeneous Poisson process model
  `log_rates = GaussianProcess(index_points=X); y = Poisson(exp(log_rates))`
  in which the latent (log) rates are spatially correlated following a Gaussian
  process:

  ```python
  # Toy 1D data.
  index_points = np.array([-10., -7.2, -4., -0.1, 0.1, 4., 6.2, 9.]).reshape(
      [-1, 1]).astype(np.float32)
  observed_counts = np.array(
      [100, 90, 60, 13, 18, 37, 55, 42]).astype(np.float32)

  # Generative model.
  def model_fn():
    kernel_amplitude = yield tfd.LogNormal(
        loc=0., scale=1., name='kernel_amplitude')
    kernel_lengthscale = yield tfd.LogNormal(
        loc=0., scale=1., name='kernel_lengthscale')
    observation_noise_scale = yield tfd.LogNormal(
        loc=0., scale=1., name='observation_noise_scale')
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=kernel_amplitude,
        length_scale=kernel_lengthscale)
    latent_log_rates = yield tfd.GaussianProcess(
        kernel,
        index_points=index_points,
        observation_noise_variance=observation_noise_scale,
        name='latent_log_rates')
    y = yield tfd.Independent(tfd.Poisson(log_rate=latent_log_rates),
                              reinterpreted_batch_ndims=1,
                              name='y')
  model = tfd.JointDistributionCoroutineAutoBatched(model_fn)
  pinned = model.experimental_pin(y=observed_counts)
  ```

  Next we define a variational family. This is represented statelessly
  as a `build_surrogate_posterior_fn` from raw (unconstrained) parameters to
  a surrogate posterior distribution. Note that common variational families can
  be constructed automatically using the utilities in `tfp.experimental.vi`;
  here we demonstrate a manual approach.

  ```python

  initial_parameters = (0., 0., 0.,  # Raw kernel parameters.
                        tf.zeros_like(observed_counts),  # `logit_locs`
                        tf.zeros_like(observed_counts))  # `logit_raw_scales`

  def build_surrogate_posterior_fn(
    raw_kernel_amplitude, raw_kernel_lengthscale, raw_observation_noise_scale,
    logit_locs, logit_raw_scales):

    def variational_model_fn():
      # Fit the kernel parameters as point masses.
      yield tfd.Deterministic(
          tf.nn.softplus(raw_kernel_amplitude), name='kernel_amplitude')
      yield tfd.Deterministic(
          tf.nn.softplus(raw_kernel_lengthscale), name='kernel_lengthscale')
      yield tfd.Deterministic(
          tf.nn.softplus(raw_observation_noise_scale),
          name='kernel_observation_noise_scale')
      # Factored normal posterior over the GP logits.
      yield tfd.Independent(
          tfd.Normal(loc=logit_locs,
                     scale=tf.nn.softplus(logit_raw_scales)),
          reinterpreted_batch_ndims=1,
          name='latent_log_rates')
    return tfd.JointDistributionCoroutineAutoBatched(variational_model_fn)
  ```

  Finally, we fit the variational posterior and model variables jointly. We'll
  use a custom `trace_fn` to see how the kernel amplitudes and a set of sampled
  latent rates with fixed seed evolve during the course of the optimization:

  ```python
  [
      optimized_parameters,
      (losses, amplitude_path, sample_path)
  ] = tfp.vi.fit_surrogate_posterior_stateless(
      target_log_prob_fn=pinned.unnormalized_log_prob,
      build_surrogate_posterior_fn=build_surrogate_posterior_fn,
      initial_parameters=initial_parameters,
      optimizer=optax.adam(learning_rate=0.1),
      sample_size=1,
      num_steps=500,
      trace_fn=lambda traceable_quantities: (  # pylint: disable=g-long-lambda
            traceable_quantities.loss,
            tf.nn.softplus(traceable_quantities.parameters[0]),
            build_surrogate_posterior_fn(
                *traceable_quantities.parameters).sample(
                5, seed=(42, 42))[-1]),
      seed=(42, 42))
  surrogate_posterior = build_surrogate_posterior_fn(*optimized_parameters)
  ```

  #### References

  [1]: Christopher M. Bishop. Pattern Recognition and Machine Learning.
       Springer, 2006.

  [2]  Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance Weighted
       Autoencoders. In _International Conference on Learning
       Representations_, 2016. https://arxiv.org/abs/1509.00519

  [3]  Tom Rainforth, Adam R. Kosiorek, Tuan Anh Le, Chris J. Maddison,
       Maximilian Igl, Frank Wood, and Yee Whye Teh. Tighter Variational Bounds
       are Not Necessarily Better. In _International Conference on Machine
       Learning (ICML)_, 2018. https://arxiv.org/abs/1802.04537

  """
  def variational_loss_fn(*parameters, seed=None):
    surrogate_posterior = build_surrogate_posterior_fn(*parameters)
    stopped_surrogate_posterior = None
    if (gradient_estimator ==
        csiszar_divergence.GradientEstimators.DOUBLY_REPARAMETERIZED):
      stopped_surrogate_posterior = build_surrogate_posterior_fn(
          *tf.nest.map_structure(tf.stop_gradient, parameters))
    return csiszar_divergence.monte_carlo_variational_loss(
        target_log_prob_fn,
        surrogate_posterior=surrogate_posterior,
        discrepancy_fn=discrepancy_fn,
        importance_sample_size=importance_sample_size,
        sample_size=sample_size,
        gradient_estimator=gradient_estimator,
        stopped_surrogate_posterior=stopped_surrogate_posterior,
        seed=seed)

  return minimize_stateless(
      variational_loss_fn,
      init=initial_parameters,
      num_steps=num_steps,
      optimizer=optimizer,
      convergence_criterion=convergence_criterion,
      trace_fn=trace_fn,
      jit_compile=jit_compile,
      seed=seed,
      name=name)


def fit_surrogate_posterior(target_log_prob_fn,
                            surrogate_posterior,
                            optimizer,
                            num_steps,
                            convergence_criterion=None,
                            trace_fn=_trace_loss,
                            discrepancy_fn=csiszar_divergence.kl_reverse,
                            sample_size=1,
                            importance_sample_size=1,
                            trainable_variables=None,
                            jit_compile=None,
                            seed=None,
                            name='fit_surrogate_posterior'):
  """Fit a surrogate posterior to a target (unnormalized) log density.

  The default behavior constructs and minimizes the negative variational
  evidence lower bound (ELBO), given by

  ```python
  q_samples = surrogate_posterior.sample(num_draws)
  elbo_loss = -tf.reduce_mean(
    target_log_prob_fn(q_samples) - surrogate_posterior.log_prob(q_samples))
  ```

  This corresponds to minimizing the 'reverse' Kullback-Liebler divergence
  (`KL[q||p]`) between the variational distribution and the unnormalized
  `target_log_prob_fn`, and  defines a lower bound on the marginal log
  likelihood, `log p(x) >= -elbo_loss`. [1]

  More generally, this function supports fitting variational distributions that
  minimize any
  [Csiszar f-divergence](https://en.wikipedia.org/wiki/F-divergence).

  Args:
    target_log_prob_fn: Python callable that takes a set of `Tensor` arguments
      and returns a `Tensor` log-density. Given
      `q_sample = surrogate_posterior.sample(sample_size)`, this
      will be called as `target_log_prob_fn(*q_sample)` if `q_sample` is a list
      or a tuple, `target_log_prob_fn(**q_sample)` if `q_sample` is a
      dictionary, or `target_log_prob_fn(q_sample)` if `q_sample` is a `Tensor`.
      It should support batched evaluation, i.e., should return a result of
      shape `[sample_size]`.
    surrogate_posterior: A `tfp.distributions.Distribution`
      instance defining a variational posterior (could be a
      `tfd.JointDistribution`). Crucially, the distribution's `log_prob` and
      (if reparameterized) `sample` methods must directly invoke all ops
      that generate gradients to the underlying variables. One way to ensure
      this is to use `tfp.util.TransformedVariable` and/or
      `tfp.util.DeferredTensor` to represent any parameters defined as
      transformations of unconstrained variables, so that the transformations
      execute at runtime instead of at distribution creation.
    optimizer: Optimizer instance to use. This may be a TF1-style
      `tf.train.Optimizer`, TF2-style `tf.optimizers.Optimizer`, or any Python
      object that implements `optimizer.apply_gradients(grads_and_vars)`.
    num_steps: Python `int` number of steps to run the optimizer.
    convergence_criterion: Optional instance of
      `tfp.optimizer.convergence_criteria.ConvergenceCriterion`
      representing a criterion for detecting convergence. If `None`,
      the optimization will run for `num_steps` steps, otherwise, it will run
      for at *most* `num_steps` steps, as determined by the provided criterion.
      Default value: `None`.
    trace_fn: Python callable with signature `traced_values = trace_fn(
      traceable_quantities)`, where the argument is an instance of
      `tfp.math.MinimizeTraceableQuantities` and the returned `traced_values`
      may be a `Tensor` or nested structure of `Tensor`s. The traced values are
      stacked across steps and returned.
      The default `trace_fn` simply returns the loss. In general, trace
      functions may also examine the gradients, values of parameters,
      the state propagated by the specified `convergence_criterion`, if any (if
      no convergence criterion is specified, this will be `None`),
      as well as any other quantities captured in the closure of `trace_fn`,
      for example, statistics of a variational distribution.
      Default value: `lambda traceable_quantities: traceable_quantities.loss`.
    discrepancy_fn: Python `callable` representing a Csiszar `f` function in
      in log-space. See the docs for `tfp.vi.monte_carlo_variational_loss` for
      examples.
      Default value: `tfp.vi.kl_reverse`.
    sample_size: Python `int` number of Monte Carlo samples to use
      in estimating the variational divergence. Larger values may stabilize
      the optimization, but at higher cost per step in time and memory.
      Default value: `1`.
    importance_sample_size: Python `int` number of terms used to define an
      importance-weighted divergence. If `importance_sample_size > 1`, then the
      `surrogate_posterior` is optimized to function as an importance-sampling
      proposal distribution. In this case, posterior expectations should be
      approximated by importance sampling, as demonstrated in the example below.
      Default value: `1`.
    trainable_variables: Optional list of `tf.Variable` instances to optimize
      with respect to. If `None`, defaults to the set of all variables accessed
      during the computation of the variational bound, i.e., those defining
      `surrogate_posterior` and the model `target_log_prob_fn`.
      Default value: `None`
    jit_compile: If True, compiles the loss function and gradient update using
      XLA. XLA performs compiler optimizations, such as fusion, and attempts to
      emit more efficient code. This may drastically improve the performance.
      See the docs for `tf.function`. (In JAX, this will apply `jax.jit`).
      Default value: `None`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    name: Python `str` name prefixed to ops created by this function.
      Default value: 'fit_surrogate_posterior'.

  Returns:
    results: `Tensor` or nested structure of `Tensor`s, according to the return
      type of `trace_fn`. Each `Tensor` has an added leading dimension of size
      `num_steps`, packing the trajectory of the result over the course of the
      optimization.

  #### Examples

  **Normal-Normal model**. We'll first consider a simple model
  `z ~ N(0, 1)`, `x ~ N(z, 1)`, where we suppose we are interested in the
  posterior `p(z | x=5)`:

  ```python
  import tensorflow_probability as tfp
  from tensorflow_probability import distributions as tfd

  def log_prob(z, x):
    return tfd.Normal(0., 1.).log_prob(z) + tfd.Normal(z, 1.).log_prob(x)
  conditioned_log_prob = lambda z: log_prob(z, x=5.)
  ```

  The posterior is itself normal by [conjugacy](
  https://en.wikipedia.org/wiki/Conjugate_prior), and can be computed
  analytically (it's `N(loc=5/2., scale=1/sqrt(2)`). But suppose we don't want
  to bother doing the math: we can use variational inference instead!

  ```python
  q_z = tfp.experimental.util.make_trainable(tfd.Normal, name='q_z')
  losses = tfp.vi.fit_surrogate_posterior(
      conditioned_log_prob,
      surrogate_posterior=q_z,
      optimizer=tf.optimizers.Adam(learning_rate=0.1),
      num_steps=100)
  print(q_z.mean(), q_z.stddev())  # => approximately [2.5, 1/sqrt(2)]
  ```

  **Custom loss function**. Suppose we prefer to fit the same model using
    the forward KL divergence `KL[p||q]`. We can pass a custom discrepancy
    function:

  ```python
    losses = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob,
        surrogate_posterior=q_z,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=100,
        discrepancy_fn=tfp.vi.kl_forward)
  ```

  Note that in practice this may have substantially higher-variance gradients
  than the reverse KL.

  **Importance weighting**. A surrogate posterior may be corrected by
  interpreting it as a proposal for an [importance sampler](
  https://en.wikipedia.org/wiki/Importance_sampling). That is, one can use
  weighted samples from the surrogate to estimate expectations under the true
  posterior:

  ```python
  zs, q_log_prob = surrogate_posterior.experimental_sample_and_log_prob(
    num_samples)

  # Naive expectation under the surrogate posterior.
  expected_x = tf.reduce_mean(f(zs), axis=0)

  # Importance-weighted estimate of the expectation under the true posterior.
  self_normalized_log_weights = tf.nn.log_softmax(
    target_log_prob_fn(zs) - q_log_prob)
  expected_x = tf.reduce_sum(
    tf.exp(self_normalized_log_weights) * f(zs),
    axis=0)
  ```

  Any distribution may be used as a proposal, but it is often natural to
  consider surrogates that were themselves fit by optimizing an
  importance-weighted variational objective [2], which directly optimizes the
  surrogate's effectiveness as an proposal distribution. This may be specified
  by passing `importance_sample_size > 1`. The importance-weighted objective
  may favor different characteristics than the original objective.
  For example, effective proposals are generally overdispersed, whereas a
  surrogate optimizing reverse KL would otherwise tend to be underdispersed.

  Although importance sampling is guaranteed to tighten the variational bound,
  some research has found that this does not necessarily improve the quality
  of deep generative models, because it also introduces gradient noise that can
  lead to a weaker training signal [3]. As always, evaluation is important to
  choose the approach that works best for a particular task.

  When using an importance-weighted loss to fit a surrogate, it is also
  recommended to apply importance sampling when computing expectations
  under that surrogate.

  ```python
  # Fit `q` with an importance-weighted variational loss.
  losses = tfp.vi.fit_surrogate_posterior(
        conditioned_log_prob,
        surrogate_posterior=q_z,
        importance_sample_size=10,
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_steps=200)

  # Estimate posterior statistics with importance sampling.
  zs, q_log_prob = q_z.experimental_sample_and_log_prob(1000)
  self_normalized_log_weights = tf.nn.log_softmax(
    conditioned_log_prob(zs) - q_log_prob)
  posterior_mean = tf.reduce_sum(
    tf.exp(self_normalized_log_weights) * zs,
    axis=0)
  posterior_variance = tf.reduce_sum(
    tf.exp(self_normalized_log_weights) * (zs - posterior_mean)**2,
    axis=0)
  ```

  **Inhomogeneous Poisson Process**. For a more interesting example, let's
  consider a model with multiple latent variables as well as trainable
  parameters in the model itself. Given observed counts `y` from spatial
  locations `X`, consider an inhomogeneous Poisson process model
  `log_rates = GaussianProcess(index_points=X); y = Poisson(exp(log_rates))`
  in which the latent (log) rates are spatially correlated following a Gaussian
  process. We'll fit a variational model to the latent rates while also
  optimizing the GP kernel hyperparameters (largely for illustration; in
  practice we might prefer to 'be Bayesian' about these parameters and include
  them as latents in our model and variational posterior). First we define
  the model, including trainable variables:

  ```python
  # Toy 1D data.
  index_points = np.array([-10., -7.2, -4., -0.1, 0.1, 4., 6.2, 9.]).reshape(
      [-1, 1]).astype(np.float32)
  observed_counts = np.array(
      [100, 90, 60, 13, 18, 37, 55, 42]).astype(np.float32)

  # Trainable GP hyperparameters.
  kernel_log_amplitude = tf.Variable(0., name='kernel_log_amplitude')
  kernel_log_lengthscale = tf.Variable(0., name='kernel_log_lengthscale')
  observation_noise_log_scale = tf.Variable(
    0., name='observation_noise_log_scale')

  # Generative model.
  Root = tfd.JointDistributionCoroutine.Root
  def model_fn():
    kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
        amplitude=tf.exp(kernel_log_amplitude),
        length_scale=tf.exp(kernel_log_lengthscale))
    latent_log_rates = yield Root(tfd.GaussianProcess(
        kernel,
        index_points=index_points,
        observation_noise_variance=tf.exp(observation_noise_log_scale),
        name='latent_log_rates'))
    y = yield tfd.Independent(tfd.Poisson(log_rate=latent_log_rates, name='y'),
                              reinterpreted_batch_ndims=1)
  model = tfd.JointDistributionCoroutine(model_fn)
  ```

  Next we define a variational distribution. We incorporate the observations
  directly into the variational model using the 'trick' of representing them
  by a deterministic distribution (observe that the true posterior on an
  observed value is in fact a point mass at the observed value).

  ```
  logit_locs = tf.Variable(tf.zeros(observed_counts.shape), name='logit_locs')
  logit_softplus_scales = tf.Variable(tf.ones(observed_counts.shape) * -4,
                                      name='logit_softplus_scales')
  def variational_model_fn():
    latent_rates = yield Root(tfd.Independent(
      tfd.Normal(loc=logit_locs, scale=tf.nn.softplus(logit_softplus_scales)),
      reinterpreted_batch_ndims=1))
    y = yield tfd.VectorDeterministic(observed_counts)
  q = tfd.JointDistributionCoroutine(variational_model_fn)
  ```

  Note that here we could apply transforms to variables without using
  `DeferredTensor` because the `JointDistributionCoroutine` argument is a
  function, i.e., executed "on demand." (The same is true when
  distribution-making functions are supplied to `JointDistributionSequential`
  and `JointDistributionNamed`. That is, as long as variables are transformed
  *within* the callable, they will appear on the gradient tape when
  `q.log_prob()` or `q.sample()` are invoked.

  Finally, we fit the variational posterior and model variables jointly: by not
  explicitly specifying `trainable_variables`, the optimization will
  automatically include all variables accessed. We'll
  use a custom `trace_fn` to see how the kernel amplitudes and a set of sampled
  latent rates with fixed seed evolve during the course of the optimization:

  ```python
  losses, log_amplitude_path, sample_path = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=lambda *args: model.log_prob(args),
    surrogate_posterior=q,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    sample_size=1,
    num_steps=500,
    trace_fn=lambda loss, grads, vars: (loss, kernel_log_amplitude,
                                        q.sample(5, seed=42)[0]))
  ```

  #### References

  [1]: Christopher M. Bishop. Pattern Recognition and Machine Learning.
       Springer, 2006.

  [2]  Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance Weighted
       Autoencoders. In _International Conference on Learning
       Representations_, 2016. https://arxiv.org/abs/1509.00519

  [3]  Tom Rainforth, Adam R. Kosiorek, Tuan Anh Le, Chris J. Maddison,
       Maximilian Igl, Frank Wood, and Yee Whye Teh. Tighter Variational Bounds
       are Not Necessarily Better. In _International Conference on Machine
       Learning (ICML)_, 2018. https://arxiv.org/abs/1802.04537

  """

  variational_loss_fn = functools.partial(
      csiszar_divergence.monte_carlo_variational_loss,
      discrepancy_fn=discrepancy_fn,
      importance_sample_size=importance_sample_size,
      # Silent fallback to score-function gradients leads to
      # difficult-to-debug failures, so force reparameterization gradients by
      # default.
      gradient_estimator=(
          csiszar_divergence.GradientEstimators.REPARAMETERIZATION),
      )

  def complete_variational_loss_fn(seed=None):
    return variational_loss_fn(
        target_log_prob_fn,
        surrogate_posterior,
        sample_size=sample_size,
        seed=seed)

  return minimize(
      complete_variational_loss_fn,
      num_steps=num_steps,
      optimizer=optimizer,
      convergence_criterion=convergence_criterion,
      trace_fn=trace_fn,
      trainable_variables=trainable_variables,
      jit_compile=jit_compile,
      seed=seed,
      name=name)
