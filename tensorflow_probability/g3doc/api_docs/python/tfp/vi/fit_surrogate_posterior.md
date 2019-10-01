<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.fit_surrogate_posterior" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.fit_surrogate_posterior


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/optimization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Fit a surrogate posterior to a target (unnormalized) log density.

``` python
tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    optimizer,
    num_steps,
    trace_fn=_trace_loss,
    variational_loss_fn=_reparameterized_elbo,
    sample_size=1,
    trainable_variables=None,
    seed=None,
    name='fit_surrogate_posterior'
)
```



<!-- Placeholder for "Used in" -->

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

#### Args:


* <b>`target_log_prob_fn`</b>: Python callable that takes a set of `Tensor` arguments
  and returns a `Tensor` log-density. Given
  `q_sample = surrogate_posterior.sample(sample_size)`, this
  will be called as `target_log_prob_fn(*q_sample)` if `q_sample` is a list
  or a tuple, `target_log_prob_fn(**q_sample)` if `q_sample` is a
  dictionary, or `target_log_prob_fn(q_sample)` if `q_sample` is a `Tensor`.
  It should support batched evaluation, i.e., should return a result of
  shape `[sample_size]`.
* <b>`surrogate_posterior`</b>: A <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>
  instance defining a variational posterior (could be a
  `tfd.JointDistribution`). Crucially, the distribution's `log_prob` and
  (if reparameterized) `sample` methods must directly invoke all ops
  that generate gradients to the underlying variables. One way to ensure
  this is to use <a href="../../tfp/util/TransformedVariable.md"><code>tfp.util.TransformedVariable</code></a> and/or
  <a href="../../tfp/util/DeferredTensor.md"><code>tfp.util.DeferredTensor</code></a> to represent any parameters defined as
  transformations of unconstrained variables, so that the transformations
  execute at runtime instead of at distribution creation.
* <b>`optimizer`</b>: Optimizer instance to use. This may be a TF1-style
  `tf.train.Optimizer`, TF2-style `tf.optimizers.Optimizer`, or any Python
  object that implements `optimizer.apply_gradients(grads_and_vars)`.
* <b>`num_steps`</b>: Python `int` number of steps to run the optimizer.
* <b>`trace_fn`</b>: Python callable with signature `state = trace_fn(
  loss, grads, variables)`, where `state` may be a `Tensor` or nested
  structure of `Tensor`s. The state values are accumulated (by `tf.scan`)
  and returned. The default `trace_fn` simply returns the loss, but in
  general can depend on the gradients and variables (if
  `trainable_variables` is not `None` then `variables==trainable_variables`;
  otherwise it is the list of all variables accessed during execution of
  `loss_fn()`), as well as any other quantities captured in the closure of
  `trace_fn`, for example, statistics of a variational distribution.
  Default value: `lambda loss, grads, variables: loss`.
* <b>`variational_loss_fn`</b>: Python `callable` with signature
  `loss = variational_loss_fn(target_log_prob_fn, surrogate_posterior,
   sample_size, seed)` defining a variational loss function. The default is
   a Monte Carlo approximation to the standard evidence lower bound (ELBO),
   equivalent to minimizing the 'reverse' `KL[q||p]` divergence between the
   surrogate `q` and true posterior `p`. [1]
   Default value: `functools.partial(
     tfp.vi.monte_carlo_variational_loss,
     discrepancy_fn=tfp.vi.kl_reverse,
     use_reparameterization=True)`.
* <b>`sample_size`</b>: Python `int` number of Monte Carlo samples to use
  in estimating the variational divergence. Larger values may stabilize
  the optimization, but at higher cost per step in time and memory.
  Default value: `1`.
* <b>`trainable_variables`</b>: Optional list of `tf.Variable` instances to optimize
  with respect to. If `None`, defaults to the set of all variables accessed
  during the computation of the variational bound, i.e., those defining
  `surrogate_posterior` and the model `target_log_prob_fn`.
  Default value: `None`
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: 'fit_surrogate_posterior'.


#### Returns:


* <b>`results`</b>: `Tensor` or nested structure of `Tensor`s, according to the
  return type of `result_fn`. Each `Tensor` has an added leading dimension
  of size `num_steps`, packing the trajectory of the result over the course
  of the optimization.

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
q_z = tfd.Normal(loc=tf.Variable(0., name='q_z_loc'),
                 scale=tfp.util.TransformedVariable(1., tfb.Softplus(),
                                                    name='q_z_scale'),
                 name='q_z')
losses = tfp.vi.fit_surrogate_posterior(
    conditioned_log_prob,
    surrogate_posterior=q,
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_steps=100)
print(q_z.mean(), q_z.stddev())  # => approximately [2.5, 1/sqrt(2)]
```

Note that we ensure positive scale by using a softplus transformation of
the underlying variable, invoked via `TransformedVariable`. Deferring the
transformation causes it to be applied upon evaluation of the distribution's
methods, creating a gradient to the underlying variable. If we
had simply specified `scale=tf.nn.softplus(scale_var)` directly,
without the `TransformedVariable`, fitting would fail because calls to
`q.log_prob` and `q.sample` would never access the underlying variable. In
general, transformations of trainable parameters must be deferred to runtime,
using either `TransformedVariable` or `DeferredTensor` or by the callable
mechanisms available in joint distribution classes (demonstrated below).

**Custom loss function**. Suppose we prefer to fit the same model using
  the forward KL divergence `KL[p||q]`. We can pass a custom loss function:

```python
  import functools
  forward_kl_loss = functools.partial(
    tfp.vi.monte_carlo_csiszar_f_divergence, tfp.vi.kl_forward)
  losses = tfp.vi.fit_surrogate_posterior(
      conditioned_log_prob,
      surrogate_posterior=q,
      optimizer=tf.optimizers.Adam(learning_rate=0.1),
      num_steps=100,
      variational_loss_fn=forward_kl_loss)
```

Note that in practice this may have substantially higher-variance gradients
than the reverse KL.

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
index_points = np.array(
    [-10., -7.2, -4., -1., 0.8, 4., 6.2, 9.]).astype(np.float32)
observed_counts = np.array(
    [100, 90, 60, 1, 4, 37, 55, 42]).astype(np.float32)

# Trainable GP hyperparameters.
kernel_log_amplitude = tf.Variable(0., name='kernel_log_amplitude')
kernel_log_lengthscale = tf.Variable(0., name='kernel_log_lengthscale')
observation_noise_log_scale = tf.Variable(
  0., name='observation_noise_log_scale')

# Generative model.
Root = tfd.JointDistributionCoroutine.Root
def model_fn():
  kernel = tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
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
  y = yield tfd.VectorDeterministic(y)
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

[1]: Bishop, Christopher M. Pattern Recognition and Machine Learning.
     Springer, 2006.