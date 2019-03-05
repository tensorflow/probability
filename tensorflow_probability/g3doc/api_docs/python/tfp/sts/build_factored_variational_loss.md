<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.sts.build_factored_variational_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.sts.build_factored_variational_loss

``` python
tfp.sts.build_factored_variational_loss(
    model,
    observed_time_series,
    init_batch_shape=(),
    seed=None,
    name=None
)
```

Build a loss function for variational inference in STS models.

Variational inference searches for the distribution within some family of
approximate posteriors that minimizes a divergence between the approximate
posterior `q(z)` and true posterior `p(z|observed_time_series)`. By converting
inference to optimization, it's generally much faster than sampling-based
inference algorithms such as HMC. The tradeoff is that the approximating
family rarely contains the true posterior, so it may miss important aspects of
posterior structure (in particular, dependence between variables) and should
not be blindly trusted. Results may vary; it's generally wise to compare to
HMC to evaluate whether inference quality is sufficient for your task at hand.

This method constructs a loss function for variational inference using the
Kullback-Liebler divergence `KL[q(z) || p(z|observed_time_series)]`, with an
approximating family given by independent Normal distributions transformed to
the appropriate parameter space for each parameter. Minimizing this loss (the
negative ELBO) maximizes a lower bound on the log model evidence `-log
p(observed_time_series)`. This is equivalent to the 'mean-field' method
implemented in [1]. and is a standard approach. The resulting posterior
approximations are unimodal; they will tend to underestimate posterior
uncertainty when the true posterior contains multiple modes (the `KL[q||p]`
divergence encourages choosing a single mode) or dependence between variables.

#### Args:

* <b>`model`</b>: An instance of `StructuralTimeSeries` representing a
    time-series model. This represents a joint distribution over
    time-series and their parameters with batch shape `[b1, ..., bN]`.
* <b>`observed_time_series`</b>: `float` `Tensor` of shape
    `concat([sample_shape, model.batch_shape, [num_timesteps, 1]]) where
    `sample_shape` corresponds to i.i.d. observations, and the trailing `[1]`
    dimension may (optionally) be omitted if `num_timesteps > 1`.
* <b>`init_batch_shape`</b>: Batch shape (Python `tuple`, `list`, or `int`) of initial
    states to optimize in parallel.
    Default value: `()`. (i.e., just run a single optimization).
* <b>`seed`</b>: Python integer to seed the random number generator.
* <b>`name`</b>: Python `str` name prefixed to ops created by this function.
    Default value: `None` (i.e., 'build_factored_variational_loss').


#### Returns:

* <b>`variational_loss`</b>: `float` `Tensor` of shape
    `concat([init_batch_shape, model.batch_shape])`, encoding a stochastic
    estimate of an upper bound on the negative model evidence `-log p(y)`.
    Minimizing this loss performs variational inference; the gap between the
    variational bound and the true (generally unknown) model evidence
    corresponds to the divergence `KL[q||p]` between the approximate and true
    posterior.
* <b>`variational_distributions`</b>: `collections.OrderedDict` giving
    the approximate posterior for each model parameter. The keys are
    Python `str` parameter names in order, corresponding to
    `[param.name for param in model.parameters]`. The values are
    `tfd.Distribution` instances with batch shape
    `concat([init_batch_shape, model.batch_shape])`; these will typically be
    of the form `tfd.TransformedDistribution(tfd.Normal(...),
    bijector=param.bijector)`.

#### Examples

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

To run variational inference, we simply construct the loss and optimize
it:

```python
  (variational_loss,
   variational_distributions) = tfp.sts.build_factored_variational_loss(
     model=model, observed_time_series=observed_time_series)

  train_op = tf.train.AdamOptimizer(0.1).minimize(variational_loss)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(200):
      _, loss_ = sess.run((train_op, variational_loss))

      if step % 20 == 0:
        print("step {} loss {}".format(step, loss_))

    posterior_samples_ = sess.run({
      param_name: q.sample(50)
      for param_name, q in variational_distributions.items()})
```

As a more complex example, we might try to avoid local optima by optimizing
from multiple initializations in parallel, and selecting the result with the
lowest loss:

```python
  (variational_loss,
   variational_distributions) = tfp.sts.build_factored_variational_loss(
     model=model, observed_time_series=observed_time_series,
     init_batch_shape=[10])

  train_op = tf.train.AdamOptimizer(0.1).minimize(variational_loss)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(200):
      _, loss_ = sess.run((train_op, variational_loss))

      if step % 20 == 0:
        print("step {} losses {}".format(step, loss_))

    # Draw multiple samples to reduce Monte Carlo error in the optimized
    # variational bounds.
    avg_loss = np.mean(
      [sess.run(variational_loss) for _ in range(25)], axis=0)
    best_posterior_idx = np.argmin(avg_loss, axis=0).astype(np.int32)
```

#### References

[1]: Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and
     David M. Blei. Automatic Differentiation Variational Inference. In
     _Journal of Machine Learning Research_, 2017.
     https://arxiv.org/abs/1603.00788