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
"""Joint distributions with inferred batch semantics."""

from tensorflow_probability.python.distributions import joint_distribution_coroutine
from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_sequential


class JointDistributionCoroutineAutoBatched(
    joint_distribution_coroutine.JointDistributionCoroutine):
  """Joint distribution parameterized by a distribution-making generator.

  This class provides automatic vectorization and alternative semantics for
  `tfd.JointDistributionCoroutine`, which in many cases allows for
  simplifications in the model specification.

  #### Automatic vectorization

  Auto-vectorized variants of JointDistribution allow the user to avoid
  explicitly annotating a model's vectorization semantics.
  When using manually-vectorized joint distributions, each operation in the
  model must account for the possibility of batch dimensions in Distributions
  and their samples. By contrast, auto-vectorized models need only describe
  a *single* sample from the joint distribution; any batch evaluation is
  automated using `tf.vectorized_map` as required. In many cases this
  allows for significant simplications. For example, the following
  manually-vectorized `tfd.JointDistributionCoroutine` model:

  ```python
  def model_fn():
    x = yield tfd.JointDistributionCoroutine.Root(
      tfd.Normal(0., tf.ones([3])))
    y = yield tfd.JointDistributionCoroutine.Root(
      tfd.Normal(0., 1.))
    z = yield tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)

  can be written in auto-vectorized form as

  ```python
  def model_fn():
    x = yield tfd.Normal(0., tf.ones([3]))
    y = yield tfd.Normal(0., 1.)
    z = yield tfd.Normal(x[:2] + y, 1.)
  ```

  in which we were able to drop the specification of `Root` nodes and to
  avoid explicitly accounting for batch dimensions when indexing and slicing
  computed quantities in the third line.

  Note: auto-vectorization is still experimental and some TensorFlow ops may
  be unsupported. It can be disabled by setting `use_vectorized_map=False`.

  #### Alternative batch semantics

  This class also provides alternative semantics for specifying a batch of
  independent (non-identical) joint distributions.

  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  `batch_ndims` is explicitly set to a nonzero value (in which case the result
  will have the corresponding tensor rank).

  The essential changes are:

  - An `event` of `JointDistributionCoroutineAutoBatched` is the list of
    tensors produced by `.sample()`; thus, the `event_shape` is the
    list of the shapes of sampled tensors. These combine both the event
    and batch dimensions of the component distributions. By contrast, the event
    shape of a base `JointDistribution`s does not include batch dimensions of
    component distributions.
  - The `batch_shape` is a global property of the entire model, rather
    than a per-component property as in base `JointDistribution`s.
    The global batch shape must be a prefix of the batch shapes of
    each component; the length of this prefix is specified by an optional
    argument `batch_ndims`. If `batch_ndims` is not specified, the model has
    batch shape `[]`.

  #### Examples

  A hierarchical model of Poisson log-rates, written using
  `tfd.JointDistributionCoroutineAutoBatched`:

  ```python
  tfd = tfp.distributions
  def model():
    global_log_rate = yield tfd.Normal(loc=0., scale=1.)
    local_log_rates = yield tfd.Normal(loc=0., scale=tf.ones([20]))
    observed_counts = yield tfd.Poisson(
      rate=tf.exp(global_log_rate + local_log_rates))
  joint = tfd.JointDistributionCoroutineAutoBatched(model)

  print(joint.event_shape)
  # ==> [[], [20], [20]]
  print(joint.batch_shape)
  # ==> []
  xs = joint.sample()
  print([x.shape for x in xs])
  # ==> [[], [20], [20]]
  lp = joint.log_prob(xs)
  print(lp.shape)
  # ==> []
  ```

  Note that the component distributions of this model would, by themselves,
  return batches of log-densities (because they are constructed with batch
  shape); the joint model implicitly sums over these to compute the single
  joint log-density.


  ```python
  ds, xs = joint.sample_distributions()
  print([d.event_shape for d in ds])
  # ==> [[], [], []] != model.event_shape
  print([d.batch_shape for d in ds])
  # ==> [[], [20], [20]] != model.batch_shape
  print([d.log_prob(x).shape for (d, x) in zip(ds, xs)])
  # ==> [[], [20], [20]]
  ```

  The behavior of `JointDistributionCoroutineAutoBatched` is (assuming that
  `batch_ndims` is not specified) equivalent to
  adding `tfp.distributions.Independent` wrappers to reinterpret all batch
  dimensions in a `JointDistributionCoroutine` model. That is, the model above
  would be equivalently written using `JointDistributionCoroutine` as:

  ```python
  def model_jdc():
    global_log_rate = yield Root(tfd.Normal(0., 1.))
    local_log_rates = yield Root(tfd.Independent(
      tfd.Normal(0., tf.ones([20])), reinterpreted_batch_ndims=1))
    observed_counts = yield Root(tfd.Independent(
      tfd.Poisson(tf.exp(global_log_rate + local_log_rates)),
      reinterpreted_batch_ndims=1))
  joint_jdc = tfd.JointDistributionCoroutine(model_jdc)
  ```

  To define a *batch* of joint distributions (independent, but not identical,
  joint distributions from the same family) using
  `JointDistributionCoroutineAutoBatched`, any batch dimensions must be a shared
  prefix of the batch dimensions for all components. The `batch_ndims` argument
  determines the size of the prefix to consider. For example, consider a simple
  joint model with two scalar normal random variables, where the second
  variable's mean is given by the first variable. We can write a batch of five
  such models as:

  ```python
  def model():
    x = yield tfd.Normal(0., scale=tf.ones([5]))
    y = yield tfd.Normal(x, scale=[3., 2., 5., 1., 6.])
  batch_joint = tfd.JointDistributionCoroutineAutoBatched(model, batch_ndims=1)

  print(batch_joint.event_shape)
  # ==> [[], []]
  print(batch_joint.batch_shape)
  # ==> [5]
  print(batch_joint.log_prob(batch_joint.sample()).shape)
  # ==> [5]
  ```

  Note that if we had not passed `batch_ndims`, this would be interpreted as a
  single model over vector-valued random variables (whose components happen to
  be independent):

  ```python
  alternate_joint = tfd.JointDistributionCoroutineAutoBatched(model)
  print(alternate_joint.event_shape)
  # ==> [[5], [5]]
  print(alternate_joint.batch_shape)
  # ==> []
  print(alternate_joint.log_prob(batch_joint.sample()).shape)
  # ==> []
  ```

  For improved readability of sampled values, the yielded distributions can also
  be named:

  ```python
  tfd = tfp.distributions
  def model():
    global_log_rate = yield tfd.Normal(
      loc=0., scale=1., name='global_log_rate')
    local_log_rates = yield tfd.Normal(
      loc=0., scale=tf.ones([20]), name='local_log_rates')
    observed_counts = yield tfd.Poisson(
      rate=tf.exp(global_log_rate + local_log_rates), name='observed_counts')
  joint = tfd.JointDistributionCoroutineAutoBatched(model)

  print(joint.event_shape)
  # ==> StructTuple(global_log_rate=[], local_log_rates=[20],
  #      observed_counts=[20])
  print(joint.batch_shape)
  # ==> []
  xs = joint.sample()
  print(['{}: {}'.format(k, x.shape) for k, x in xs._asdict().items()])
  # ==> global_log_scale: []
  #     local_log_rates: [20]
  #     observed_counts: [20]
  lp = joint.log_prob(xs)
  print(lp.shape)
  # ==> []

  # Passing via `kwargs` also works.
  lp = joint.log_prob(**xs._asdict())
  # Or:
  lp = joint.log_prob(
      global_log_scale=...,
      local_log_rates=...,
      observed_counts=...,
  )
  ```

  If any of the yielded distributions are not explicitly named, they will
  automatically be given a name of the form `var#` where `#` is the index of the
  associated distribution. E.g. the first yielded distribution will have a
  default name of `var0`.

  """

  def __init__(
      self,
      model,
      sample_dtype=None,
      batch_ndims=0,
      use_vectorized_map=True,
      validate_args=False,
      experimental_use_kahan_sum=False,
      name=None,
  ):
    """Construct the `JointDistributionCoroutineAutoBatched` distribution.

    Args:
      model: A generator that yields a sequence of `tfd.Distribution`-like
        instances.
      sample_dtype: Samples from this distribution will be structured like
        `tf.nest.pack_sequence_as(sample_dtype, list_)`. `sample_dtype` is only
        used for `tf.nest.pack_sequence_as` structuring of outputs, never
        casting (which is the responsibility of the component distributions).
        Default value: `None` (i.e. `namedtuple`).
      batch_ndims: `int` `Tensor` number of batch dimensions. The `batch_shape`s
        of all component distributions must be such that the prefixes of
        length `batch_ndims` broadcast to a consistent joint batch shape.
        Default value: `0`.
      use_vectorized_map: Python `bool`. Whether to use `tf.vectorized_map`
        to automatically vectorize evaluation of the model. This allows the
        model specification to focus on drawing a single sample, which is often
        simpler, but some ops may not be supported.
        Default value: `True`.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `JointDistributionCoroutine`).
    """
    parameters = dict(locals())
    super(JointDistributionCoroutineAutoBatched, self).__init__(
        model, sample_dtype=sample_dtype, batch_ndims=batch_ndims,
        use_vectorized_map=use_vectorized_map, validate_args=validate_args,
        experimental_use_kahan_sum=experimental_use_kahan_sum,
        name=name or 'JointDistributionCoroutineAutoBatched')
    self._parameters = self._no_dependency(parameters)

  @property
  def _require_root(self):
    return not self._use_vectorized_map


# TODO(b/159723894): Reduce complexity by eliminating use of mixins.
class JointDistributionNamedAutoBatched(
    joint_distribution_named.JointDistributionNamed):
  """Joint distribution parameterized by named distribution-making functions.

  This class provides automatic vectorization and alternative semantics for
  `tfd.JointDistributionNamed`, which in many cases allows for
  simplifications in the model specification.

  #### Automatic vectorization

  Auto-vectorized variants of JointDistribution allow the user to avoid
  explicitly annotating a model's vectorization semantics.
  When using manually-vectorized joint distributions, each operation in the
  model must account for the possibility of batch dimensions in Distributions
  and their samples. By contrast, auto-vectorized models need only describe
  a *single* sample from the joint distribution; any batch evaluation is
  automated using `tf.vectorized_map` as required. In many cases this
  allows for significant simplications. For example, the following
  manually-vectorized `tfd.JointDistributionNamed` model:

  ```python
  model = tfd.JointDistributionNamed({
    'x': tfd.Normal(0., tf.ones([3])),
    'y': tfd.Normal(0., 1.),
    'z': lambda x, y: tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)
  })
  ```

  can be written in auto-vectorized form as

  ```python
  model = tfd.JointDistributionNamedAutoBatched({
    'x': tfd.Normal(0., tf.ones([3])),
    'y': tfd.Normal(0., 1.),
    'z': lambda x, y: tfd.Normal(x[:2] + y, 1.)
  })
  ```

  in which we were able to avoid explicitly accounting for batch dimensions
  when indexing and slicing computed quantities in the third line.

  Note: auto-vectorization is still experimental and some TensorFlow ops may
  be unsupported. It can be disabled by setting `use_vectorized_map=False`.

  #### Alternative batch semantics

  This class also provides alternative semantics for specifying a batch of
  independent (non-identical) joint distributions.

  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  `batch_ndims` is explicitly set to a nonzero value (in which case the result
  will have the corresponding tensor rank).

  The essential changes are:

  - An `event` of `JointDistributionNamedAutoBatched` is the dictionary of
    tensors produced by `.sample()`; thus, the `event_shape` is the
    dictionary containing the shapes of sampled tensors. These combine both
    the event and batch dimensions of the component distributions. By contrast,
    the event shape of a base `JointDistribution`s does not include batch
    dimensions of component distributions.
  - The `batch_shape` is a global property of the entire model, rather
    than a per-component property as in base `JointDistribution`s.
    The global batch shape must be a prefix of the batch shapes of
    each component; the length of this prefix is specified by an optional
    argument `batch_ndims`. If `batch_ndims` is not specified, the model has
    batch shape `[]`.

  #### Examples

  Consider the following generative model:

  ```
  e ~ Exponential(rate=[100,120])
  g ~ Gamma(concentration=e[0], rate=e[1])
  n ~ Normal(loc=0, scale=2.)
  m ~ Normal(loc=n, scale=g)
  for i = 1, ..., 12:
    x[i] ~ Bernoulli(logits=m)
  ```

  We can code this as:

  ```python
  tfd = tfp.distributions
  joint = tfd.JointDistributionNamedAutoBatched(dict(
      e=             tfd.Exponential(rate=[100, 120]),
      g=lambda    e: tfd.Gamma(concentration=e[0], rate=e[1]),
      n=             tfd.Normal(loc=0, scale=2.),
      m=lambda n, g: tfd.Normal(loc=n, scale=g),
      x=lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12),
  ))
  ```

  Notice the 1:1 correspondence between "math" and "code". In a standard
  `JointDistributionNamed`, we would have wrapped the first variable as
  `e = tfd.Independent(tfd.Exponential(rate=[100, 120]),
   reinterpreted_batch_ndims=1)` to specify that `log_prob` of the `Exponential`
  should be a scalar, summing over both dimensions. We would also have had to
  extend indices as `tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])` to
  account for possible batch dimensions. Both of these behaviors are implicit
  in `JointDistributionNamedAutoBatched`.

  """

  def __init__(self, model, batch_ndims=0, use_vectorized_map=True,
               validate_args=False, experimental_use_kahan_sum=False,
               name=None):
    """Construct the `JointDistributionNamedAutoBatched` distribution.

    Args:
      model: A generator that yields a sequence of `tfd.Distribution`-like
        instances.
      batch_ndims: `int` `Tensor` number of batch dimensions. The `batch_shape`s
        of all component distributions must be such that the prefixes of
        length `batch_ndims` broadcast to a consistent joint batch shape.
        Default value: `0`.
      use_vectorized_map: Python `bool`. Whether to use `tf.vectorized_map`
        to automatically vectorize evaluation of the model. This allows the
        model specification to focus on drawing a single sample, which is often
        simpler, but some ops may not be supported.
        Default value: `True`.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `JointDistributionNamed`).
    """
    parameters = dict(locals())
    super(JointDistributionNamedAutoBatched, self).__init__(
        model, batch_ndims=batch_ndims, use_vectorized_map=use_vectorized_map,
        validate_args=validate_args,
        experimental_use_kahan_sum=experimental_use_kahan_sum,
        name=name or 'JointDistributionNamedAutoBatched')
    self._parameters = self._no_dependency(parameters)


class JointDistributionSequentialAutoBatched(
    joint_distribution_sequential.JointDistributionSequential):
  """Joint distribution parameterized by distribution-making functions.

  This class provides automatic vectorization and alternative semantics for
  `tfd.JointDistributionNamed`, which in many cases allows for
  simplifications in the model specification.

  #### Automatic vectorization

  Auto-vectorized variants of JointDistribution allow the user to avoid
  explicitly annotating a model's vectorization semantics.
  When using manually-vectorized joint distributions, each operation in the
  model must account for the possibility of batch dimensions in Distributions
  and their samples. By contrast, auto-vectorized models need only describe
  a *single* sample from the joint distribution; any batch evaluation is
  automated using `tf.vectorized_map` as required. In many cases this
  allows for significant simplications. For example, the following
  manually-vectorized `tfd.JointDistributionSequential` model:

  ```python
  model = tfd.JointDistributionSequential([
      tfd.Normal(0., tf.ones([3])),
      tfd.Normal(0., 1.),
      lambda y, x: tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)
    ])
  ```

  can be written in auto-vectorized form as

  ```python
  model = tfd.JointDistributionAutoBatchedSequential([
      tfd.Normal(0., tf.ones([3])),
      tfd.Normal(0., 1.),
      lambda y, x: tfd.Normal(x[:2] + y, 1.)
    ])
  ```

  in which we were able to avoid explicitly accounting for batch dimensions
  when indexing and slicing computed quantities in the third line.

  Note: auto-vectorization is still experimental and some TensorFlow ops may
  be unsupported. It can be disabled by setting `use_vectorized_map=False`.

  #### Alternative batch semantics

  This class also provides alternative semantics for specifying a batch of
  independent (non-identical) joint distributions.

  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  `batch_ndims` is explicitly set to a nonzero value (in which case the result
  will have the corresponding tensor rank).

  The essential changes are:

  - An `event` of `JointDistributionSequentialAutoBatched` is the list of
    tensors produced by `.sample()`; thus, the `event_shape` is the
    list containing the shapes of sampled tensors. These combine both
    the event and batch dimensions of the component distributions. By contrast,
    the event shape of a base `JointDistribution`s does not include batch
    dimensions of component distributions.
  - The `batch_shape` is a global property of the entire model, rather
    than a per-component property as in base `JointDistribution`s.
    The global batch shape must be a prefix of the batch shapes of
    each component; the length of this prefix is specified by an optional
    argument `batch_ndims`. If `batch_ndims` is not specified, the model has
    batch shape `[]`.

  #### Examples

  Consider the following generative model:

  ```
  e ~ Exponential(rate=[100,120])
  g ~ Gamma(concentration=e[0], rate=e[1])
  n ~ Normal(loc=0, scale=2.)
  m ~ Normal(loc=n, scale=g)
  for i = 1, ..., 12:
    x[i] ~ Bernoulli(logits=m)
  ```

  We can code this as:

  ```python
  tfd = tfp.distributions
  joint = tfd.JointDistributionSequentialAutoBatched([
                   tfd.Exponential(rate=[100, 120]), 1,         # e
      lambda    e: tfd.Gamma(concentration=e[0], rate=e[1]),    # g
                   tfd.Normal(loc=0, scale=2.),                 # n
      lambda n, g: tfd.Normal(loc=n, scale=g)                   # m
      lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12)      # x
  ])
  ```

  Notice the 1:1 correspondence between "math" and "code". In a standard
  `JointDistributionSequential`, we would have wrapped the first variable as
  `e = tfd.Independent(tfd.Exponential(rate=[100, 120]),
   reinterpreted_batch_ndims=1)` to specify that `log_prob` of the `Exponential`
  should be a scalar, summing over both dimensions. We would also have had to
  extend indices as `tfd.Gamma(concentration=e[..., 0], rate=e[..., 1])` to
  account for possible batch dimensions. Both of these behaviors are implicit
  in `JointDistributionSequentialAutoBatched`.

  """

  def __init__(self, model, batch_ndims=0, use_vectorized_map=True,
               validate_args=False, experimental_use_kahan_sum=False,
               name=None):
    """Construct the `JointDistributionSequentialAutoBatched` distribution.

    Args:
      model: A generator that yields a sequence of `tfd.Distribution`-like
        instances.
      batch_ndims: `int` `Tensor` number of batch dimensions. The `batch_shape`s
        of all component distributions must be such that the prefixes of
        length `batch_ndims` broadcast to a consistent joint batch shape.
        Default value: `0`.
      use_vectorized_map: Python `bool`. Whether to use `tf.vectorized_map`
        to automatically vectorize evaluation of the model. This allows the
        model specification to focus on drawing a single sample, which is often
        simpler, but some ops may not be supported.
        Default value: `True`.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `JointDistributionSequential`).
    """
    parameters = dict(locals())
    super(JointDistributionSequentialAutoBatched, self).__init__(
        model, batch_ndims=batch_ndims, use_vectorized_map=use_vectorized_map,
        validate_args=validate_args,
        experimental_use_kahan_sum=experimental_use_kahan_sum,
        name=name or 'JointDistributionSequentialAutoBatched')
    self._parameters = self._no_dependency(parameters)
