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
"""The `JointDistributionCoroutine` class."""

import collections
import warnings

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import joint_distribution as joint_distribution_lib

from tensorflow_probability.python.internal import structural_tuple
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'JointDistributionCoroutine',
]


JAX_MODE = False

# Cause all warnings to always be triggered.
# Not having this means subsequent calls wont trigger the warning.
warnings.filterwarnings(
    'always',
    module='tensorflow_probability.*joint_distribution_coroutine',
    append=True)  # Don't override user-set filters.


class JointDistributionCoroutine(
    joint_distribution_lib.JointDistribution,
    distribution_lib.AutoCompositeTensorDistribution):
  """Joint distribution parameterized by a distribution-making generator.

  This distribution enables both sampling and joint probability computation from
  a single model specification.

  A joint distribution is a collection of possibly interdependent distributions.
  The `JointDistributionCoroutine` is specified by a generator that
  generates the elements of this collection.

  #### Mathematical Details

  The `JointDistributionCoroutine` implements the chain rule of probability.
  That is, the probability function of a length-`d` vector `x` is,

  ```none
  p(x) = prod{ p(x[i] | x[:i]) : i = 0, ..., (d - 1) }
  ```

  The `JointDistributionCoroutine` is parameterized by a generator
  that yields `tfp.distributions.Distribution`-like instances.

  Each element yielded implements the `i`-th *full conditional distribution*,
  `p(x[i] | x[:i])`. Within the generator, the return value from the yield
  is a sample from the distribution that may be used to construct subsequent
  yielded `Distribution`-like instances. This allows later instances
  to be conditional on earlier ones.

  **Name resolution**: The names of `JointDistributionCoroutine` components
  may be specified by passing `name` arguments to distribution constructors (
  `tfd.Normal(0., 1., name='x')). Components without an explicit name will be
  assigned a dummy name.

  #### Vectorized sampling and model evaluation

  When a joint distribution's `sample` method is called with
  a `sample_shape` (or the `log_prob` method is called on an input with
  multiple sample dimensions) the model must be equipped to handle
  additional batch dimensions. This may be done manually, or automatically
  by passing `use_vectorized_map=True`. Manual vectorization has historically
  been the default, but we now recommend that most users enable automatic
  vectorization unless they are affected by a specific issue; some
  known issues are listed below.

  When using manually-vectorized joint distributions, each operation in the
  model must account for the possibility of batch dimensions in Distributions
  and their samples. By contrast, auto-vectorized models need only describe
  a *single* sample from the joint distribution; any batch evaluation is
  automated as required using `tf.vectorized_map` (`vmap` in JAX). In many
  cases this allows for significant simplications. For example, the following
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

  **Root annotations**: When the `sample` method for a manually-vectorized
  `JointDistributionCoroutine` is called with a `sample_shape`, the `sample`
  method for each of the yielded distributions is called.
  The distributions that have been wrapped in the
  `JointDistributionCoroutine.Root` class will be called with `sample_shape`
  as the `sample_shape` argument, and the unwrapped distributions
  will be called with `()` as the `sample_shape` argument. It is the user's
  responsibility to ensure that each of the distributions generates samples
  with the specified sample size; generally this means applying `Root` wrappers
  around any distributions whose parameters are not already a function of other
  random variables. The `Root` annotation can be omitted if you never intend to
  use a `sample_shape` other than `()`.

  **Known limitations of automatic vectorization:**
  - A small fraction of TensorFlow ops are unsupported; models that use an
    unsupported op will raise an error and must be manually vectorized.
  - Sampling large batches may be slow under automatic vectorization because
    TensorFlow's stateless samplers are currently converted using a
    non-vectorized `while_loop`. This limitation applies only in TensorFlow;
    vectorized samplers in JAX should be approximately as fast as manually
    vectorized code.
  - Calling `sample_distributions` with nontrivial `sample_shape` will raise
    an error if the model contains any distributions that are not registered as
    CompositeTensors (TFP's basic distributions are usually fine, but support
    for wrapper distributions like `tfd.Sample` is a work in progress).

  #### Batch semantics and (log-)densities

  **tl;dr:** pass `batch_ndims=0` unless you have a good reason not to.

  Joint distributions now support 'auto-batching' semantics, in which
  the distribution's batch shape is derived by broadcasting the leftmost
  `batch_ndims` dimensions of its components' batch shapes. All remaining
  dimensions are considered to form a single 'event' of the joint distribution.
  If `batch_ndims==0`, then the joint distribution has batch shape `[]`, and all
  component dimensions are treated as event shape. For example, the model

  ```python
  def model_fn():
    x = yield tfd.Normal(0., tf.ones([3]))
    y = yield tfd.Normal(x[..., tf.newaxis], tf.ones([3, 2]))
  jd = tfd.JointDistributionCoroutine(model_fn, batch_ndims=0)
  ```

  creates a joint distribution with batch shape `[]` and event shape
  `([3], [3, 2])`. The log-density of a sample always has shape
  `batch_shape`, so this guarantees that
  `jd.log_prob(jd.sample())` will evaluate to a scalar value. We could
  alternately construct a joint distribution with batch shape `[3]` and event
  shape `([], [2])` by setting `batch_ndims=1`, in which case
  `jd.log_prob(jd.sample())` would evaluate to a value of shape `[3]`.

  Setting `batch_ndims=None` recovers the 'classic' batch semantics (currently
  still the default for backwards-compatibility reasons), in which the joint
  distribution's `log_prob` is computed by naively summing log densities from
  the component distributions. Since these component densities have shapes equal
  to the batch shapes of the individual components, to avoid broadcasting
  errors it is usually necessary to construct the components with identical
  batch shapes. For example, the component distributions in the model above
  have batch shapes of `[3]` and `[3, 2]` respectively, which would raise an
  error if summed directly, but can be aligned by wrapping with
  `tfd.Independent`, as in this model:

  ```python
  def model_fn():
    x = yield tfd.Normal(0., tf.ones([3]))
    y = yield tfd.Independent(tfd.Normal(x[..., tf.newaxis], tf.ones([3, 2])),
                              reinterpreted_batch_ndims=1)
  jd = tfd.JointDistributionCoroutine(model_fn, batch_ndims=None)
  ```

  Here the components both have batch shape `[3]`, so
  `jd.log_prob(jd.sample())` returns a value of shape `[3]`, just as in the
  `batch_ndims=1` case above. In fact, auto-batching semantics are equivalent to
  implicitly wrapping each component `dist` as `tfd.Independent(dist,
  reinterpreted_batch_ndim=(dist.batch_shape.ndims - jd.batch_ndims))`; the only
  vestigial difference is that under auto-batching semantics, the joint
  distribution has a single batch shape `[3]`, while under the classic semantics
  the value of `jd.batch_shape` is a *structure* of the component batch shapes
  `([3], [3])`. Such structured batch shapes will be deprecated in the future,
  since they are inconsistent with the definition of batch shapes used
  elsewhere in TFP.

  **Note**: If `model_fn` closes over a `Tensor`, the
  `JointDistributionCoroutine` instance cannot cross the boundary of a
  `tf.function`.

  #### Examples

  ```python
  tfd = tfp.distributions
  def model():
    global_log_rate = yield tfd.Normal(loc=0., scale=1.)
    local_log_rates = yield tfd.Normal(loc=0., scale=tf.ones([20]))
    observed_counts = yield tfd.Poisson(
      rate=tf.exp(global_log_rate + local_log_rates))
  joint = tfd.JointDistributionCoroutine(model,
                                         use_vectorized_map=True,
                                         batch_ndims=0)

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
  joint = tfd.JointDistributionCoroutine(model,
                                         use_vectorized_map=True,
                                         batch_ndims=0)

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

  #### References

  [1] Dan Piponi, Dave Moore, and Joshua V. Dillon. Joint distributions for
      TensorFlow Probability. _arXiv preprint arXiv:2001.11819__,
      2020. https://arxiv.org/abs/2001.11819
  """

  def __init__(self,
               model,
               sample_dtype=None,
               batch_ndims=None,
               use_vectorized_map=False,
               validate_args=False,
               experimental_use_kahan_sum=False,
               name=None):
    """Construct the `JointDistributionCoroutine` distribution.

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
        Default value: `None`.
      use_vectorized_map: Python `bool`. Whether to use `tf.vectorized_map`
        to automatically vectorize evaluation of the model. This allows the
        model specification to focus on drawing a single sample, which is often
        simpler, but some ops may not be supported.
        Default value: `False`.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`. This argument has no effect if
        `batch_ndims is None`.
        Default value: `False`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `JointDistributionCoroutine`).
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'JointDistributionCoroutine') as name:
      self._model_coroutine = model
      # Hint `no_dependency` to tell tf.Module not to screw up the sample dtype
      # with extraneous wrapping (list => ListWrapper, etc.).
      self._sample_dtype = self._no_dependency(sample_dtype)
      super(JointDistributionCoroutine, self).__init__(
          dtype=sample_dtype,
          batch_ndims=batch_ndims,
          use_vectorized_map=use_vectorized_map,
          validate_args=validate_args,
          parameters=parameters,
          experimental_use_kahan_sum=experimental_use_kahan_sum,
          name=name)

  @property
  def model(self):
    return self._model_coroutine

  def _model_unflatten(self, xs):
    if self._sample_dtype is None:
      return structural_tuple.structtuple(self._flat_resolve_names())(*xs)
    # Cast `xs` as `tuple` so we can handle generators.
    return tf.nest.pack_sequence_as(self._sample_dtype, tuple(xs))

  def _model_flatten(self, xs):
    if self._sample_dtype is None:
      return tuple((xs[k] for k in self._flat_resolve_names())
                   if isinstance(xs, collections.abc.Mapping) else xs)
    return nest.flatten_up_to(self._sample_dtype, xs)

  _composite_tensor_shape_params = ('batch_ndims',)
  _composite_tensor_nonshape_params = ()
