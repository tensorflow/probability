# Copyright 2020 The TensorFlow Probability Authors.
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
"""`JointDistribution` mixin class implementing automatic vectorization."""

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.distributions import joint_distribution as joint_distribution_lib
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import vectorization_util

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


JAX_MODE = False


def _might_have_excess_ndims(flat_value, flat_core_ndims):
  for v, nd in zip(flat_value, flat_core_ndims):
    static_excess_ndims = (
        0 if v is None else
        tf.get_static_value(ps.convert_to_shape_tensor(ps.rank(v) - nd)))
    if static_excess_ndims is None or static_excess_ndims > 0:
      return True
  return False


# Lint doesn't know that docstrings are defined in the base JD class.
# pylint: disable=missing-docstring
class JointDistributionVmapMixin(object):
  """A joint distribution with automatically vectorized sample and log-prob.

  Auto-vectorized variants of JointDistribution treat the underlying
  model as describing a single possible world, or equivalently, as
  specifying the process of generating a single sample from the model.
  Drawing multiple samples, and computing batched log-probs, is accomplished
  using `tf.vectorized_map`. In many cases this allows for significant
  simplication of the model. For example, the following
  manually-vectorized `tfd.JointDistributionCoroutine` model:

  ```python
  def model_fn():
    x = yield tfd.JointDistributionCoroutine.Root(
      tfd.Normal(0., tf.ones([3])))
    y = yield tfd.JointDistributionCoroutine.Root(
      tfd.Normal(0., 1.)))
    z = yield tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)

  can be written in auto-vectorized form as

  ```python
  def model_fn():
    x = yield tfd.Normal(0., tf.ones([3]))
    y = yield tfd.Normal(0., 1.))
    z = yield tfd.Normal(x[:2] + y, 1.)
  ```

  in which we were able to drop the specification of `Root` nodes and to
  avoid explicitly accounting for batch dimensions when indexing and slicing
  computed quantities in the third line.

  Note: auto-vectorization is still experimental and some TensorFlow ops may
  be unsupported.

  A limitation relative to standard `JointDistribution`s is that the
  `sample_distributions()` method requires all component distributions to be
  registered as `CompositeTensor`s (see
  `tfp.experimental.auto_composite_tensor`) when called with nontrivial sample
  shape.
  """

  def __init__(self, *args, **kwargs):
    self._use_vectorized_map = kwargs.pop('use_vectorized_map', True)
    super(JointDistributionVmapMixin, self).__init__(*args, **kwargs)

  # TODO(b/166658748): Drop this (make it always True).
  _stateful_to_stateless = JAX_MODE

  @property
  def use_vectorized_map(self):
    return self._use_vectorized_map

  @property
  def _single_sample_ndims(self):
    """Computes the rank of values produced by executing the base model."""
    # Attempt to get static ranks without running the model.
    attrs = self._get_static_distribution_attributes()
    batch_ndims, event_ndims = tf.nest.map_structure(
        tensorshape_util.rank, [attrs.batch_shape, attrs.event_shape])
    if any(nd is None for nd in tf.nest.flatten([batch_ndims, event_ndims])):
      batch_ndims, event_ndims = tf.nest.map_structure(
          ps.rank_from_shape,
          [list(self._map_attr_over_dists('batch_shape_tensor')),
           list(self._map_attr_over_dists('event_shape_tensor'))])
    return [tf.nest.map_structure(lambda x, b=b: b + x, e)
            for (b, e) in zip(batch_ndims, event_ndims)]

  def _call_execute_model(self,
                          sample_shape,
                          seed,
                          value=None,
                          sample_and_trace_fn=None):
    """Wraps the base `_call_execute_model` with vectorized_map."""
    value_might_have_sample_dims = (
        value is not None and _might_have_excess_ndims(
            # Double-flatten in case any components have structured events.
            flat_value=nest.flatten_up_to(self._single_sample_ndims,
                                          self._model_flatten(value),
                                          check_types=False),
            flat_core_ndims=tf.nest.flatten(self._single_sample_ndims)))
    sample_shape_may_be_nontrivial = (
        distribution_util.shape_may_be_nontrivial(sample_shape))

    if not self.use_vectorized_map or not (
        sample_shape_may_be_nontrivial or  # pylint: disable=protected-access
        value_might_have_sample_dims):
      # No need to auto-vectorize.
      return joint_distribution_lib.JointDistribution._call_execute_model(  # pylint: disable=protected-access
          self, sample_shape=sample_shape, seed=seed, value=value,
          sample_and_trace_fn=sample_and_trace_fn)

    # Set up for autovectorized sampling. To support the `value` arg, we need to
    # first understand which dims are from the model itself, then wrap
    # `_call_execute_model` to batch over all remaining dims.
    value_core_ndims = None
    if value is not None:
      value_core_ndims = tf.nest.map_structure(
          lambda v, nd: None if v is None else nd,
          value, self._model_unflatten(self._single_sample_ndims),
          check_types=False)

    vectorized_execute_model_helper = vectorization_util.make_rank_polymorphic(
        lambda v, seed: (  # pylint: disable=g-long-lambda
            joint_distribution_lib.JointDistribution._call_execute_model(  # pylint: disable=protected-access
                self,
                sample_shape=(),
                seed=seed,
                value=v,
                sample_and_trace_fn=sample_and_trace_fn)),
        core_ndims=[value_core_ndims, None],
        validate_args=self.validate_args)
    # Redefine the polymorphic fn to hack around `make_rank_polymorphic`
    # not currently supporting keyword args. This is needed because the
    # `iid_sample` wrapper below expects to pass through a `seed` kwarg.
    vectorized_execute_model = (
        lambda v, seed: vectorized_execute_model_helper(v, seed))  # pylint: disable=unnecessary-lambda

    if sample_shape_may_be_nontrivial:
      vectorized_execute_model = vectorization_util.iid_sample(
          vectorized_execute_model, sample_shape)

    return vectorized_execute_model(value, seed=seed)

  def _default_event_space_bijector(self, *args, **kwargs):
    bijector_class = joint_distribution_lib._DefaultJointBijector  # pylint: disable=protected-access
    if self.use_vectorized_map:
      bijector_class = _DefaultJointBijectorAutoBatched
    if bool(args) or bool(kwargs):
      return self.experimental_pin(
          *args, **kwargs).experimental_default_event_space_bijector()
    return bijector_class(self)


class _DefaultJointBijectorAutoBatched(bijector_lib.Bijector):
  """Automatically vectorized support bijector for autobatched JDs."""

  def __init__(self, jd, **kwargs):
    parameters = dict(locals())
    self._jd = jd
    self._bijector_kwargs = kwargs
    self._joint_bijector = joint_distribution_lib._DefaultJointBijector(
        jd=self._jd, **self._bijector_kwargs)
    super(_DefaultJointBijectorAutoBatched, self).__init__(
        forward_min_event_ndims=self._joint_bijector.forward_min_event_ndims,
        inverse_min_event_ndims=self._joint_bijector.inverse_min_event_ndims,
        validate_args=self._joint_bijector.validate_args,
        parameters=parameters,
        name=self._joint_bijector.name)
    # Any batch dimensions of the JD must be included in the core
    # 'event' processed by autobatched bijector methods. This is because
    # `vectorized_map` has no visibility into the internal batch vs event
    # semantics of the methods being vectorized. More precisely, if we
    # didn't do this, then:
    #  1. Calling `self.inverse_log_det_jacobian(y)` with a `y` of shape
    #     `jd.event_shape` would in general return a result of shape
    #     `jd.batch_shape` (since each batch member can define a different
    #     transformation).
    #  2. By the semantics of `vectorized_map`, calling
    #     `self.inverse_log_det_jacobian(y)` with an `y` of shape
    #     `concat([jd.batch_shape, jd.event_shape])` would therefore return
    #     a result of shape `concat([jd.batch_shape, jd.batch_shape])`, in
    #     which the batch shape appears *twice*.
    #  3. This breaks the TFP shape contract and is bad.
    # We avoid this by requiring that `y` is at least of shape
    # `jd.sample().shape`.
    jd_batch_ndims = ps.rank_from_shape(jd.batch_shape_tensor())
    forward_core_ndims = tf.nest.map_structure(
        lambda nd: jd_batch_ndims + nd, self.forward_min_event_ndims)
    inverse_core_ndims = tf.nest.map_structure(
        lambda nd: jd_batch_ndims + nd, self.inverse_min_event_ndims)
    # Wrap the non-batched `joint_bijector` to take batched args.
    # pylint: disable=protected-access
    self._forward = self._vectorize_member_fn(
        lambda bij, x: bij._forward(x), core_ndims=[forward_core_ndims])
    self._inverse = self._vectorize_member_fn(
        lambda bij, y: bij._inverse(y), core_ndims=[inverse_core_ndims])
    self._forward_log_det_jacobian = self._vectorize_member_fn(
        # Need to explicitly broadcast LDJ if `bij` has constant Jacobian.
        lambda bij, x: tf.broadcast_to(  # pylint: disable=g-long-lambda
            bij._forward_log_det_jacobian(
                x, event_ndims=self.forward_min_event_ndims),
            jd.batch_shape_tensor()),
        core_ndims=[forward_core_ndims])
    self._inverse_log_det_jacobian = self._vectorize_member_fn(
        # Need to explicitly broadcast LDJ if `bij` has constant Jacobian.
        lambda bij, y: tf.broadcast_to(  # pylint: disable=g-long-lambda
            bij._inverse_log_det_jacobian(
                y, event_ndims=self.inverse_min_event_ndims),
            jd.batch_shape_tensor()),
        core_ndims=[inverse_core_ndims])
    for attr in ('_forward_event_shape',
                 '_forward_event_shape_tensor',
                 '_inverse_event_shape',
                 '_inverse_event_shape_tensor',
                 '_forward_dtype',
                 '_inverse_dtype',
                 'forward_event_ndims',
                 'inverse_event_ndims',):
      setattr(self, attr, getattr(self._joint_bijector, attr))
    # pylint: enable=protected-access

  @property
  def _parts_interact(self):
    return self._joint_bijector._parts_interact  # pylint: disable=protected-access

  def _vectorize_member_fn(self, member_fn, core_ndims):
    return vectorization_util.make_rank_polymorphic(
        lambda x: member_fn(self._joint_bijector, x),
        core_ndims=core_ndims)
