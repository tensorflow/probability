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

from __future__ import absolute_import
from __future__ import division

from tensorflow_probability.python.distributions import joint_distribution_coroutine
from tensorflow_probability.python.distributions import joint_distribution_named
from tensorflow_probability.python.distributions import joint_distribution_sample_path_mixin
from tensorflow_probability.python.distributions import joint_distribution_sequential


class JointDistributionCoroutineAutoBatched(
    joint_distribution_sample_path_mixin.JointDistributionSamplePathMixin,
    joint_distribution_coroutine.JointDistributionCoroutine):
  """Joint distribution parameterized by a distribution-making generator.

  This class provides alternate vectorization semantics for
  `tfd.JointDistributionCoroutine`, which in many cases eliminate the need to
  explicitly account for batch shapes in the model specification.
  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  otherwise specified.

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
  Root = tfd.JointDistributionCoroutineAutoBatched.Root  # Convenient alias.
  def model():
    global_log_rate = yield Root(tfd.Normal(loc=0., scale=1.))
    local_log_rates = yield Root(tfd.Normal(loc=0., scale=tf.ones([20])))
    observed_counts = yield Root(tfd.Poisson(
      rate=tf.exp(global_log_rate + local_log_rates)))
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
    x = yield Root(tfd.Normal(0., scale=tf.ones([5])))
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

  """

  def __init__(self, *args, **kwargs):
    kwargs['name'] = kwargs.get('name', 'JointDistributionCoroutineAutoBatched')
    super(JointDistributionCoroutineAutoBatched, self).__init__(*args, **kwargs)


class JointDistributionNamedAutoBatched(
    joint_distribution_sample_path_mixin.JointDistributionSamplePathMixin,
    joint_distribution_named.JointDistributionNamed):
  """Joint distribution parameterized by named distribution-making functions.

  This class provides alternate vectorization semantics for
  `tfd.JointDistributionNamed`, which in many cases eliminate the need to
  explicitly account for batch shapes in the model specification.
  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  otherwise specified.

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
      g=lambda    e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
      n=             tfd.Normal(loc=0, scale=2.),
      m=lambda n, g: tfd.Normal(loc=n, scale=g),
      x=lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12),
  ))
  ```

  Notice the 1:1 correspondence between "math" and "code". In a standard
  `JointDistributionNamed`, we would have wrapped the first variable as
  `e = tfd.Independent(tfd.Exponential(rate=[100, 120]),
   reinterpreted_batch_ndims=1)` to specify that `log_prob` of the `Exponential`
  should be a scalar, summing over both dimensions. This behavior is implicit
  in `JointDistributionNamedAutoBatched`.

  """

  def __init__(self, *args, **kwargs):
    kwargs['name'] = kwargs.get('name', 'JointDistributionNamedAuto')
    super(JointDistributionNamedAutoBatched, self).__init__(*args, **kwargs)


class JointDistributionSequentialAutoBatched(
    joint_distribution_sample_path_mixin.JointDistributionSamplePathMixin,
    joint_distribution_sequential.JointDistributionSequential):
  """Joint distribution parameterized by distribution-making functions.

  This class provides alternate vectorization semantics for
  `tfd.JointDistributionSequential`, which in many cases eliminate the need to
  explicitly account for batch shapes in the model specification.
  Instead of simply summing the `log_prob`s of component distributions
  (which may have different shapes), it first reduces the component `log_prob`s
  to ensure that `jd.log_prob(jd.sample())` always returns a scalar, unless
  otherwise specified.

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
                   tfd.Exponential(rate=[100, 120]), 1,                   # e
      lambda    e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),    # g
                   tfd.Normal(loc=0, scale=2.),                           # n
      lambda n, g: tfd.Normal(loc=n, scale=g)                             # m
      lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12)                # x
  ])
  ```

  Notice the 1:1 correspondence between "math" and "code". In a standard
  `JointDistributionSequential`, we would have wrapped the first variable as
  `e = tfd.Independent(tfd.Exponential(rate=[100, 120]),
   reinterpreted_batch_ndims=1)` to specify that `log_prob` of the `Exponential`
  should be a scalar, summing over both dimensions. This behavior is implicit
  in `JointDistributionSequentialAutoBatched`.

  """
