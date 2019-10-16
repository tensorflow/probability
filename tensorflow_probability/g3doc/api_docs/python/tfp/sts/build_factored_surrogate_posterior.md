<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.build_factored_surrogate_posterior" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.build_factored_surrogate_posterior


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/sts/fitting.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Build a variational posterior that factors over model parameters.

``` python
tfp.sts.build_factored_surrogate_posterior(
    model,
    batch_shape=(),
    seed=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The surrogate posterior consists of independent Normal distributions for
each parameter with trainable `loc` and `scale`, transformed using the
parameter's `bijector` to the appropriate support space for that parameter.

#### Args:


* <b>`model`</b>: An instance of `StructuralTimeSeries` representing a
    time-series model. This represents a joint distribution over
    time-series and their parameters with batch shape `[b1, ..., bN]`.
* <b>`batch_shape`</b>: Batch shape (Python `tuple`, `list`, or `int`) of initial
  states to optimize in parallel.
  Default value: `()`. (i.e., just run a single optimization).
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
  Default value: `None` (i.e., 'build_factored_surrogate_posterior').

#### Returns:


* <b>`variational_posterior`</b>: `tfd.JointDistributionNamed` defining a trainable
    surrogate posterior over model parameters. Samples from this
    distribution are Python `dict`s with Python `str` parameter names as
    keys.

### Examples

Assume we've built a structural time-series model:

```python
  day_of_week = tfp.sts.Seasonal(
      num_seasons=7,
      observed_time_series=observed_time_series,
      name='day_of_week')
  local_linear_trend = tfp.sts.LocalLinearTrend(
      observed_time_series=observed_time_series,
      name='local_linear_trend')
  model = tfp.sts.Sum(components=[day_of_week, local_linear_trend],
                      observed_time_series=observed_time_series)
```

To fit the model to data, we define a surrogate posterior and fit it
by optimizing a variational bound:

```python
  surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(
    model=model)
  loss_curve = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn=model.joint_log_prob(observed_time_series),
    surrogate_posterior=surrogate_posterior,
    num_steps=200)
  posterior_samples = surrogate_posterior.sample(50)

  # In graph mode, we would need to write:
  # with tf.control_dependencies([loss_curve]):
  #   posterior_samples = surrogate_posterior.sample(50)
```

For more control, we can also build and optimize a variational loss
manually:

```python
  @tf.function(autograph=False)  # Ensure the loss is computed efficiently
  def loss_fn():
    return tfp.vi.monte_carlo_variational_loss(
      model.joint_log_prob(observed_time_series),
      surrogate_posterior,
      sample_size=10)

  optimizer = tf.optimizers.Adam(learning_rate=0.1)
  for step in range(200):
    with tf.GradientTape() as tape:
      loss = loss_fn()
    grads = tape.gradient(loss, surrogate_posterior.trainable_variables)
    optimizer.apply_gradients(
      zip(grads, surrogate_posterior.trainable_variables))
    if step % 20 == 0:
      print('step {} loss {}'.format(step, loss))

  posterior_samples = surrogate_posterior.sample(50)
```