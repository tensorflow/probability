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
  `sample_distributions()` method does not currently support (nontrivial) sample
  shapes.
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
    result = []
    for d in self._get_single_sample_distributions():
      batch_ndims = ps.rank_from_shape(d.batch_shape_tensor, d.batch_shape)
      result.append(tf.nest.map_structure(
          lambda a, b, nd=batch_ndims: nd + ps.rank_from_shape(a, b),
          d.event_shape_tensor(),
          d.event_shape))
    return result

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

    if not self.use_vectorized_map or not (
        distribution_util.shape_may_be_nontrivial(sample_shape) or  # pylint: disable=protected-access
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
    batch_execute_model = vectorization_util.make_rank_polymorphic(
        lambda v, seed: (  # pylint: disable=g-long-lambda
            joint_distribution_lib.JointDistribution._call_execute_model(  # pylint: disable=protected-access
                self,
                sample_shape=(),
                seed=seed,
                value=v,
                sample_and_trace_fn=sample_and_trace_fn)),
        core_ndims=[value_core_ndims, None],
        validate_args=self.validate_args)

    # Draw samples.
    vectorized_execute_model = vectorization_util.iid_sample(
        # Redefine the polymorphic fn to hack around `make_rank_polymorphic`
        # not currently supporting keyword args.
        lambda v, seed: batch_execute_model(v, seed), sample_shape)  # pylint: disable=unnecessary-lambda
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
    # Wrap the non-batched `joint_bijector` to take batched args.
    # pylint: disable=protected-access
    self._forward = self._vectorize_member_fn(
        lambda bij, x: bij._forward(x),
        core_ndims=[self._joint_bijector.forward_min_event_ndims])
    self._inverse = self._vectorize_member_fn(
        lambda bij, y: bij._inverse(y),
        core_ndims=[self._joint_bijector.inverse_min_event_ndims])
    self._forward_log_det_jacobian = self._vectorize_member_fn(
        lambda bij, x: bij._forward_log_det_jacobian(  # pylint: disable=g-long-lambda
            x, event_ndims=bij.forward_min_event_ndims),
        core_ndims=[self._joint_bijector.forward_min_event_ndims])
    self._inverse_log_det_jacobian = self._vectorize_member_fn(
        lambda bij, y: bij._inverse_log_det_jacobian(  # pylint: disable=g-long-lambda
            y, event_ndims=bij.inverse_min_event_ndims),
        core_ndims=[self._joint_bijector.inverse_min_event_ndims])
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
