<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.experimental.vi.build_factored_surrogate_posterior" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.experimental.vi.build_factored_surrogate_posterior


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/experimental/vi/surrogate_posteriors.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Builds a joint variational posterior that factors over model variables.

``` python
tfp.experimental.vi.build_factored_surrogate_posterior(
    event_shape=None,
    constraining_bijectors=None,
    initial_unconstrained_loc=_sample_uniform_initial_loc,
    initial_unconstrained_scale=0.01,
    trainable_distribution_fn=_build_trainable_normal_dist,
    seed=None,
    validate_args=False,
    name=None
)
```



<!-- Placeholder for "Used in" -->

By default, this method creates an independent trainable Normal distribution
for each variable, transformed using a bijector (if provided) to
match the support of that variable. This makes extremely strong
assumptions about the posterior: that it is approximately normal (or
transformed normal), and that all model variables are independent.

#### Args:


* <b>`event_shape`</b>: `Tensor` shape, or nested structure of `Tensor` shapes,
  specifying the event shape(s) of the posterior variables.
* <b>`constraining_bijectors`</b>: Optional `tfb.Bijector` instance, or nested
  structure of such instances, defining support(s) of the posterior
  variables. The structure must match that of `event_shape` and may
  contain `None` values. A posterior variable will
  be modeled as `tfd.TransformedDistribution(underlying_dist,
  constraining_bijector)` if a corresponding constraining bijector is
  specified, otherwise it is modeled as supported on the
  unconstrained real line.
* <b>`initial_unconstrained_loc`</b>: Optional Python `callable` with signature
  `tensor = initial_unconstrained_loc(shape, seed)` used to sample
  real-valued initializations for the unconstrained representation of each
  variable. May alternately be a nested structure of
  `Tensor`s, giving specific initial locations for each variable; these
  must have structure matching `event_shape` and shapes determined by the
  inverse image of `event_shape` under `constraining_bijectors`, which
  may optionally be prefixed with a common batch shape.
  Default value: `functools.partial(tf.random.uniform,
    minval=-2., maxval=2., dtype=tf.float32)`.
* <b>`initial_unconstrained_scale`</b>: Optional scalar float `Tensor` initial
  scale for the unconstrained distributions, or a nested structure of
  `Tensor` initial scales for each variable.
  Default value: `1e-2`.
* <b>`trainable_distribution_fn`</b>: Optional Python `callable` with signature
  `trainable_dist = trainable_distribution_fn(initial_loc, initial_scale,
  event_ndims, validate_args)`. This is called for each model variable to
  build the corresponding factor in the surrogate posterior. It is expected
  that the distribution returned is supported on unconstrained real values.
  Default value: `functools.partial(
    tfp.vi.experimental.build_trainable_location_scale_distribution,
    distribution_fn=tfd.Normal)`, i.e., a trainable Normal distribution.
* <b>`seed`</b>: Python integer to seed the random number generator. This is used
  only when `initial_loc` is not specified.
* <b>`validate_args`</b>: Python `bool`. Whether to validate input with asserts. This
  imposes a runtime cost. If `validate_args` is `False`, and the inputs are
  invalid, correct behavior is not guaranteed.
  Default value: `False`.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` (i.e., 'build_factored_surrogate_posterior').


#### Returns:


* <b>`surrogate_posterior`</b>: A `tfd.Distribution` instance whose samples have
  shape and structure matching that of `event_shape` or `initial_loc`.

### Examples

Consider a Gamma model with unknown parameters, expressed as a joint
Distribution:

```python
Root = tfd.JointDistributionCoroutine.Root
def model_fn():
  concentration = yield Root(tfd.Exponential(1.))
  rate = yield Root(tfd.Exponential(1.))
  y = yield tfd.Sample(tfd.Gamma(concentration=concentration, rate=rate),
                       sample_shape=4)
model = tfd.JointDistributionCoroutine(model_fn)
```

Let's use variational inference to approximate the posterior over the
data-generating parameters for some observed `y`. We'll build a
surrogate posterior distribution by specifying the shapes of the latent
`rate` and `concentration` parameters, and that both are constrained to
be positive.

```python
surrogate_posterior = tfp.vi.experimental.build_factored_surrogate_posterior(
  event_shape=model.event_shape_tensor()[:-1],  # Omit the observed `y`.
  constraining_bijectors=[tfb.Softplus(),   # Rate is positive.
                          tfb.Softplus()])  # Concentration is positive.
```

This creates a trainable joint distribution, defined by variables in
`surrogate_posterior.trainable_variables`. We use `fit_surrogate_posterior`
to fit this distribution by minimizing a divergence to the true posterior.

```python
y = [0.2, 0.5, 0.3, 0.7]
losses = tfp.vi.fit_surrogate_posterior(
  lambda rate, concentration: model.log_prob([rate, concentration, y]),
  surrogate_posterior=surrogate_posterior,
  num_steps=100,
  optimizer=tf.optimizers.Adam(0.1),
  sample_size=10)

# After optimization, samples from the surrogate will approximate
# samples from the true posterior.
samples = surrogate_posterior.sample(100)
posterior_mean = [tf.reduce_mean(x) for x in samples]     # mean ~= [1.1, 2.1]
posterior_std = [tf.math.reduce_std(x) for x in samples]  # std  ~= [0.3, 0.8]
```

If we wanted to initialize the optimization at a specific location, we can
specify one when we build the surrogate posterior. This function requires the
initial location to be specified in *unconstrained* space; we do this by
inverting the constraining bijectors (note this section also demonstrates the
creation of a dict-structured model).

```python
initial_loc = {'concentration': 0.4, 'rate': 0.2}
constraining_bijectors={'concentration': tfb.Softplus(),   # Rate is positive.
                        'rate': tfb.Softplus()}   # Concentration is positive.
initial_unconstrained_loc = tf.nest.map_fn(
  lambda b, x: b.inverse(x) if b is not None else x,
  constraining_bijectors, initial_loc)
surrogate_posterior = tfp.vi.experimental.build_factored_surrogate_posterior(
  event_shape=tf.nest.map_fn(tf.shape, initial_loc),
  constraining_bijectors=constraining_bijectors,
  initial_unconstrained_loc=initial_unconstrained_state,
  initial_unconstrained_scale=1e-4)
```