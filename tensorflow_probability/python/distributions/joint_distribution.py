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
"""The `JointDistribution` base class."""

import abc
import collections
import functools
import itertools
import warnings
import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.bijectors import bijector as bijector_lib
from tensorflow_probability.python.bijectors import composition
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.bijectors import ldj_ratio as ldj_ratio_lib
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import vectorization_util
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.util.seed_stream import TENSOR_SEED_MSG_PREFIX

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'JointDistribution',
]


JAX_MODE = False


@auto_composite_tensor.auto_composite_tensor
class StaticDistributionAttributes(auto_composite_tensor.AutoCompositeTensor):
  """Container to smuggle static attributes out of a tf.function trace."""

  def __init__(self,
               batch_shape,
               dtype,
               event_shape,
               experimental_shard_axis_names,
               name,
               reparameterization_type):
    self.batch_shape = batch_shape
    self.dtype = dtype
    self.event_shape = event_shape
    self.experimental_shard_axis_names = experimental_shard_axis_names
    self.name = name
    self.reparameterization_type = reparameterization_type

  def __iter__(self):
    """Yields parameters in order matching __init__ signature."""
    return iter((self.batch_shape, self.dtype, self.event_shape,
                 self.experimental_shard_axis_names, self.name,
                 self.reparameterization_type))

  _composite_tensor_shape_params = ('batch_shape', 'event_shape')


class ValueWithTrace(collections.namedtuple(
    'ValueWithTrace',
    ['value', 'traced'])):
  """Represents an RV's realized value, and related quantity(s) to trace.

     Traced quantities may include the RV's distribution, log prob, etc. The
     realized value is not itself traced unless explicitly included in `traced`.

     During model execution, each call to `sample_and_trace_fn` returns a
     `ValueWithTrace` structure. The model uses the realized `value`, e.g., to
     define the distributions of downstream random variables, while the
     `traced` quantities are accumulated and returned to the caller (for JDs
     that use `vectorized_map`, they will be returned from inside the vmap, and
     so must be `Tensor`s or `BatchableCompositeTensor`s).

  Components:
    value: The realized value of this random variable. May be a Tensor or
      nested structure of Tensors.
    traced: Quantitiy(s) to be accumulated and
      returned from the execution of the probabilistic program. These
      may have any type (for example, they may be structures, contain
      `tfd.Distribution` instances, etc).
  """


def trace_distributions_and_values(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces both the distribution and sampled value."""
  value = _sanitize_value(dist, value)
  if value is None:
    value = dist.sample(sample_shape, seed=seed)
  elif tf.nest.is_nested(dist.dtype) and any(
      v is None for v in tf.nest.flatten(value)):
    # TODO(siege): This is making an assumption that nested dtype => partial
    # value support, which is not necessarily reasonable.
    value = dist.sample(sample_shape, seed=seed, value=value)
  return ValueWithTrace(value=value, traced=(dist, value))


def trace_distributions_only(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces the sampled value."""
  ret = trace_distributions_and_values(dist, sample_shape, seed, value)
  return ret._replace(traced=ret.traced[0])


def trace_values_only(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces the sampled value."""
  ret = trace_distributions_and_values(dist, sample_shape, seed, value)
  return ret._replace(traced=ret.traced[1])


def trace_values_and_log_probs(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces both the sampled value and its log density."""
  value = _sanitize_value(dist, value)
  if value is None:
    value, lp = dist.experimental_sample_and_log_prob(sample_shape, seed=seed)
  elif tf.nest.is_nested(dist.dtype) and any(
      v is None for v in tf.nest.flatten(value)):
    # TODO(siege): This is making an assumption that nested dtype => partial
    # value support, which is not necessarily reasonable.
    value, lp = dist.experimental_sample_and_log_prob(
        sample_shape, seed=seed, value=value)
  else:
    lp = dist.log_prob(value)
  return ValueWithTrace(value=value, traced=(value, lp))


def trace_static_attributes(dist, sample_shape, seed, value):
  """Extracts the current distribution's static attributes as Tensor specs."""
  del sample_shape
  if value is None:
    value = dist.sample(seed=seed)
  return ValueWithTrace(
      value=value,
      traced=StaticDistributionAttributes(
          batch_shape=dist.batch_shape,
          dtype=dist.dtype,
          experimental_shard_axis_names=dist.experimental_shard_axis_names,
          event_shape=dist.event_shape,
          name=get_explicit_name_for_component(dist),
          reparameterization_type=dist.reparameterization_type))


CALLING_CONVENTION_DESCRIPTION = """
The measure methods of `JointDistribution` (`log_prob`, `prob`, etc.)
can be called either by passing a single structure of tensors or by using
named args for each part of the joint distribution state. For example,

```python
jd = tfd.JointDistributionSequential([
    tfd.Normal(0., 1.),
    lambda z: tfd.Normal(z, 1.)
], validate_args=True)
jd.dtype
# ==> [tf.float32, tf.float32]
z, x = sample = jd.sample()
# The following calling styles are all permissable and produce the exactly
# the same output.
assert (jd.{method}(sample) ==
        jd.{method}(value=sample) ==
        jd.{method}(z, x) ==
        jd.{method}(z=z, x=x) ==
        jd.{method}(z, x=x))

# These calling possibilities also imply that one can also use `*`
# expansion, if `sample` is a sequence:
jd.{method}(*sample)
# and similarly, if `sample` is a map, one can use `**` expansion:
jd.{method}(**sample)
```

`JointDistribution` component distributions names are resolved via
`jd._flat_resolve_names()`, which is implemented by each `JointDistribution`
subclass (see subclass documentation for details). Generally, for components
where a name was provided---
either explicitly as the `name` argument to a distribution or as a key in a
dict-valued JointDistribution, or implicitly, e.g., by the argument name of
a `JointDistributionSequential` distribution-making function---the provided
name will be used. Otherwise the component will receive a dummy name; these
may change without warning and should not be relied upon.

Note: not all `JointDistribution` subclasses support all calling styles;
for example, `JointDistributionNamed` does not support positional arguments
(aka "unnamed arguments") unless the provided model specifies an ordering of
variables (i.e., is an `collections.OrderedDict` or `collections.namedtuple`
rather than a plain `dict`).

Note: care is taken to resolve any potential ambiguity---this is generally
possible by inspecting the structure of the provided argument and "aligning"
it to the joint distribution output structure (defined by `jd.dtype`). For
example,

```python
trivial_jd = tfd.JointDistributionSequential([tfd.Exponential(1.)])
trivial_jd.dtype  # => [tf.float32]
trivial_jd.{method}([4.])
# ==> Tensor with shape `[]`.
{method_abbr} = trivial_jd.{method}(4.)
# ==> Tensor with shape `[]`.
```

Notice that in the first call, `[4.]` is interpreted as a list of one
scalar while in the second call the input is a scalar. Hence both inputs
result in identical scalar outputs. If we wanted to pass an explicit
vector to the `Exponential` component---creating a vector-shaped batch
of `{method}`s---we could instead write
`trivial_jd.{method}(np.array([4]))`.

Args:
  *args: Positional arguments: a `value` structure or component values
    (see above).
  **kwargs: Keyword arguments: a `value` structure or component values
    (see above). May also include `name`, specifying a Python string name
    for ops generated by this method.
"""


# Avoids name collision with measure function (`log_prob`, `prob`, etc.) args.
FORBIDDEN_COMPONENT_NAMES = ('value', 'name')


@six.add_metaclass(abc.ABCMeta)
class JointDistribution(distribution_lib.Distribution):
  """Joint distribution over one or more component distributions.

  This distribution enables both sampling and joint probability computation from
  a single model specification.

  A joint distribution is a collection of possibly interdependent distributions.

  **Note**: unlike other non-`JointDistribution` distributions in
  `tfp.distributions`, `JointDistribution.sample` (and subclasses) return a
  structure of  `Tensor`s rather than a `Tensor`.  A structure can be a `list`,
  `tuple`, `dict`, `collections.namedtuple`, etc. Accordingly
  `joint.batch_shape` returns a structure of `TensorShape`s for each of the
  distributions' batch shapes and `joint.batch_shape_tensor()` returns a
  structure of `Tensor`s for each of the distributions' event shapes. (Same with
  `event_shape` analogues.)

  #### Subclass Requirements

  Subclasses implement:

  - `_model_coroutine`: A generator that yields a sequence of
    `tfd.Distribution`-like instances.

  - `_model_flatten`: takes a structured input and returns a sequence. The
    sequence order must match the order distributions are yielded from
    `_model_coroutine`.

  - `_model_unflatten`: takes a sequence and returns a structure matching the
    semantics of the `JointDistribution` subclass.

  """

  class Root(collections.namedtuple('Root', ['distribution'])):
    """Wrapper for coroutine distributions which lack distribution parents."""
    __slots__ = ()

  def __init__(self,
               dtype,
               validate_args,
               parameters,
               name,
               use_vectorized_map=False,
               batch_ndims=None,
               experimental_use_kahan_sum=False):

    if use_vectorized_map and batch_ndims is None:
      raise ValueError('Autovectorization with `use_vectorized_map=True` '
                       'is unsupported when `batch_ndims is None`. Please '
                       'specify an integer number of batch dimensions.')

    self._use_vectorized_map = use_vectorized_map
    self._batch_ndims = batch_ndims
    self._experimental_use_kahan_sum = experimental_use_kahan_sum
    self._single_sample_distributions = {}
    super(JointDistribution, self).__init__(
        dtype=dtype,
        reparameterization_type=None,  # Ignored; we'll override.
        allow_nan_stats=False,
        validate_args=validate_args,
        parameters=parameters,
        name=name)

  @property
  def _require_root(self):
    return True

  @property
  def _stateful_to_stateless(self):
    # TODO(b/166658748): Once the bug is resolved, we should be able to
    #   eliminate this workaround that disables sanitize_seed for autovectorized
    #   JDs.
    if self.use_vectorized_map:
      return JAX_MODE
    return True

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

  def _get_single_sample_distributions(self, candidate_dists=None):
    """Returns a list of dists from a single sample of the model."""

    # If we have cached distributions with Eager tensors, return those.
    ds = self._single_sample_distributions.get(-1, None)
    if ds is not None and all([d is not None for d in ds]):
      return ds

    # Otherwise, retrieve or build distributions for the current graph.
    graph_id = -1 if tf.executing_eagerly() else id(tf.constant(True).graph)
    ds = self._single_sample_distributions.get(graph_id, None)
    if ds is None or any([d is None for d in ds]):
      if candidate_dists is not None:
        ds = candidate_dists
      else:
        ds = self._execute_model(
            seed=samplers.zeros_seed(),  # Constant seed for CSE.
            sample_and_trace_fn=trace_distributions_only)
      self._single_sample_distributions[graph_id] = ds
    return ds

  def _get_static_distribution_attributes(self, seed=None):
    if not hasattr(self, '_cached_static_attributes'):
      flat_list_of_static_attributes = callable_util.get_output_spec(
          lambda: self._execute_model(  # pylint: disable=g-long-lambda
              sample_and_trace_fn=trace_static_attributes,
              seed=seed if seed is not None else samplers.zeros_seed()))
      self._cached_static_attributes = StaticDistributionAttributes(
          *zip(*flat_list_of_static_attributes))

    return self._cached_static_attributes

  # Override `tf.Module`'s `_flatten` method to ensure that distributions are
  # instantiated, so that accessing `.variables` or `.trainable_variables` gives
  # consistent results.
  def _flatten(self, *args, **kwargs):
    self._get_single_sample_distributions()
    return super(JointDistribution, self)._flatten(*args, **kwargs)

  @abc.abstractmethod
  def _model_unflatten(self, xs):
    raise NotImplementedError()

  @abc.abstractmethod
  def _model_flatten(self, xs):
    raise NotImplementedError()

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `Distribution`."""
    return self._model_unflatten(
        self._get_static_distribution_attributes().dtype)

  @property
  def reparameterization_type(self):
    """Describes how samples from the distribution are reparameterized.

    Currently this is one of the static instances
    `tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

    Returns:
      reparameterization_type: `ReparameterizationType` of each distribution in
        `model`.
    """
    return self._model_unflatten(
        self._get_static_distribution_attributes().reparameterization_type)

  @property
  def experimental_shard_axis_names(self):
    """Indicates whether part distributions have active shard axis names."""
    return self._model_unflatten(
        self._get_static_distribution_attributes().
        experimental_shard_axis_names)

  @property
  def use_vectorized_map(self):
    return self._use_vectorized_map

  @property
  def batch_ndims(self):
    return self._batch_ndims

  def _batch_shape_parts(self):
    static_batch_shapes = self._get_static_distribution_attributes().batch_shape
    if self.batch_ndims is None:
      return static_batch_shapes
    static_batch_ndims = tf.get_static_value(self.batch_ndims)
    if static_batch_ndims is None:
      return [tf.TensorShape(None)] * len(static_batch_shapes)
    return [batch_shape[:static_batch_ndims]
            for batch_shape in static_batch_shapes]

  def _batch_shape(self):
    batch_shape_parts = self._batch_shape_parts()
    if self.batch_ndims is None:
      return self._model_unflatten(batch_shape_parts)
    reduce_fn = ((lambda a, b: a.merge_with(b)) if self.validate_args
                 else tf.broadcast_static_shape)  # Allows broadcasting.
    return functools.reduce(reduce_fn, batch_shape_parts)

  def _batch_shape_tensor_parts(self):
    batch_shapes = self._map_attr_over_dists('batch_shape_tensor')
    if self.batch_ndims is None:
      return batch_shapes
    return [s[:self.batch_ndims] for s in batch_shapes]

  def _batch_shape_tensor(self):
    batch_shape_tensor_parts = self._batch_shape_tensor_parts()
    if self.batch_ndims is None:
      return self._model_unflatten(batch_shape_tensor_parts)
    return tf.convert_to_tensor(functools.reduce(
        ps.broadcast_shape, batch_shape_tensor_parts))

  def _compute_event_shape(self):
    part_attrs = self._get_static_distribution_attributes()
    flat_event_shape = part_attrs.event_shape
    if self.batch_ndims is not None:
      static_batch_ndims = tf.get_static_value(self.batch_ndims)
      if static_batch_ndims is None:
        flat_event_shape = [tf.TensorShape(None)] * len(part_attrs.dtype)
      else:
        flat_event_shape = [
            # Recurse over joint component dists.
            tf.nest.map_structure(  # pylint: disable=g-complex-comprehension
                part_batch_shape[static_batch_ndims:].concatenate,
                part_event_shape)
            for part_event_shape, part_batch_shape in zip(
                part_attrs.event_shape, part_attrs.batch_shape)]
    return self._model_unflatten(flat_event_shape)

  def _event_shape(self):
    if not hasattr(self, '_cached_event_shape'):
      self._cached_event_shape = self._no_dependency(
          self._compute_event_shape())
    return self._cached_event_shape

  def _event_shape_tensor(self):
    ds = self._get_single_sample_distributions()
    component_event_shapes = []
    for d in ds:
      batch_shape_treated_as_event = (
          [] if self.batch_ndims is None
          else d.batch_shape_tensor()[self.batch_ndims:])
      component_event_shapes.append(
          tf.nest.map_structure(
              lambda e, b=batch_shape_treated_as_event: (  # pylint: disable=g-long-lambda
                  ps.concat([b, e], axis=0)),
              d.event_shape_tensor()))
    return self._model_unflatten(component_event_shapes)

  def sample_distributions(self, sample_shape=(), seed=None, value=None,
                           name='sample_distributions', **kwargs):
    """Generate samples and the (random) distributions.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
      value: `list` of `Tensor`s in `distribution_fn` order to use to
        parameterize other ("downstream") distribution makers.
        Default value: `None` (i.e., draw a sample from each distribution).
      name: name prepended to ops created by this function.
        Default value: `"sample_distributions"`.
      **kwargs: This is an alternative to passing a `value`, and achieves the
        same effect. Named arguments will be used to parameterize other
        dependent ("downstream") distribution-making functions. If a `value`
        argument is also provided, raises a ValueError.

    Returns:
      distributions: a `tuple` of `Distribution` instances for each of
        `distribution_fn`.
      samples: a `tuple` of `Tensor`s with prepended dimensions `sample_shape`
        for each of `distribution_fn`.
    """
    # Use the user-provided seed to trace static distribution attributes, if
    # they're not already cached. This ensures we don't try to pass a stateless
    # seed to a stateful sampler, or vice versa.
    self._get_static_distribution_attributes(seed=seed)

    with self._name_and_control_scope(name):
      value = self._resolve_value(value=value,
                                  allow_partially_specified=True,
                                  **kwargs)
      try:
        ds, xs = self._call_flat_sample_distributions(
            sample_shape, seed=seed, value=value)
      except TypeError as e:
        if 'Failed to convert elements' in str(e):
          raise TypeError(
              'Some component distribution(s) cannot be returned from '
              'the vectorized model because they are not `CompositeTensor`s. '
              'You may avoid this error by ensuring that all components are '
              '`CompositeTensor`s, using, e.g., '
              '`tfp.experimental.auto_composite_tensor`, or by constructing '
              'the joint distribution with `use_vectorized_map=False`.') from e
        raise
      if (value is None and
          not distribution_util.shape_may_be_nontrivial(sample_shape)):
        # This is a single sample with no pinned values; this call will cache
        # the distributions if they are not already cached.
        self._get_single_sample_distributions(candidate_dists=ds)

      return self._model_unflatten(ds), self._model_unflatten(xs)

  def _call_flat_sample_distributions(self,
                                      sample_shape=(),
                                      seed=None,
                                      value=None):
    return zip(*self._call_execute_model(
        sample_shape,
        seed=seed,
        value=value,
        sample_and_trace_fn=trace_distributions_and_values))

  def log_prob_parts(self, value, name='log_prob_parts'):
    """Log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `log_prob_parts` and to parameterize other ("downstream")
        distributions.
      name: name prepended to ops created by this function.
        Default value: `"log_prob_parts"`.

    Returns:
      log_prob_parts: a `tuple` of `Tensor`s representing the `log_prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      sum_fn = tf.reduce_sum
      if self._experimental_use_kahan_sum:
        sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('log_prob', value),
              sum_fn))

  def prob_parts(self, value, name='prob_parts'):
    """Probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `prob_parts` and to parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"prob_parts"`.

    Returns:
      prob_parts: a `tuple` of `Tensor`s representing the `prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('prob', value),
              tf.reduce_prod))

  def unnormalized_log_prob_parts(
      self, value, name='unnormalized_log_prob_parts'):
    """Unnormalized log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which
        we compute the `unnormalized_log_prob_parts` and to
        parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"unnormalized_log_prob_parts"`.

    Returns:
      unnormalized_log_prob_parts: a `tuple` of `Tensor`s representing
        the `unnormalized_log_prob` for each `distribution_fn`
        evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name or 'unnormalized_log_prob_parts'):
      sum_fn = tf.reduce_sum
      if self._experimental_use_kahan_sum:
        sum_fn = lambda x, axis: tfp_math.reduce_kahan_sum(x, axis=axis).total
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('unnormalized_log_prob', value),
              sum_fn))

  def unnormalized_prob_parts(self, value, name='unnormalized_prob_parts'):
    """Unnormalized probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which
        we compute the `unnormalized_prob_parts` and to parameterize
        other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"unnormalized_prob_parts"`.

    Returns:
      unnormalized_prob_parts: a `tuple` of `Tensor`s representing the
        `unnormalized_prob` for each `distribution_fn` evaluated at
        each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          self._reduce_measure_over_dists(
              self._map_measure_over_dists('unnormalized_prob', value),
              tf.reduce_prod))

  def is_scalar_event(self, name='is_scalar_event'):
    """Indicates that `event_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_event: `bool` scalar `Tensor` for each distribution in `model`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          [self._is_scalar_helper(shape, shape_tensor)  # pylint: disable=g-complex-comprehension
           for (shape, shape_tensor) in zip(
               self._model_flatten(self.event_shape),
               self._model_flatten(self.event_shape_tensor()))])

  def is_scalar_batch(self, name='is_scalar_batch'):
    """Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor` for each distribution in `model`.
    """
    with self._name_and_control_scope(name):
      if self.batch_ndims is None:
        return self._model_unflatten(
            self._map_attr_over_dists('is_scalar_batch'))
      return self._is_scalar_helper(self.batch_shape, self.batch_shape_tensor())

  def _log_prob(self, value):
    return self._reduce_log_probs_over_dists(
        self._map_measure_over_dists('log_prob', value))

  def _unnormalized_log_prob(self, value):
    return self._reduce_log_probs_over_dists(
        self._map_measure_over_dists('unnormalized_log_prob', value))

  @distribution_util.AppendDocstring(kwargs_dict={
      'value': ('`Tensor`s structured like `type(model)` used to parameterize '
                'other dependent ("downstream") distribution-making functions. '
                'Using `None` for any element will trigger a sample from the '
                'corresponding distribution. Default value: `None` '
                '(i.e., draw a sample from each distribution).')})
  def _sample_n(self, sample_shape, seed, value=None):
    # Use the user-provided seed to trace static distribution attributes, if
    # they're not already cached. This ensures we don't try to pass a stateless
    # seed to a stateful sampler, or vice versa.
    self._get_static_distribution_attributes(seed=seed)

    might_have_batch_dims = (
        distribution_util.shape_may_be_nontrivial(sample_shape)
        or value is not None)
    if might_have_batch_dims:
      xs = self._call_execute_model(
          sample_shape,
          seed=seed,
          value=value,
          sample_and_trace_fn=trace_values_only)
    else:
      ds, xs = zip(*self._call_execute_model(
          sample_shape,
          seed=seed,
          value=value,
          sample_and_trace_fn=trace_distributions_and_values))
      # This is a single sample with no pinned values; this call will cache
      # the distributions if they are not already cached.
      self._get_single_sample_distributions(candidate_dists=ds)

    return self._model_unflatten(xs)

  # TODO(b/189122177): Implement _sample_and_log_prob for distributed JDs.
  def _sample_and_log_prob(self, sample_shape, seed, value=None, **kwargs):
    # Use the user-provided seed to trace static distribution attributes, if
    # they're not already cached. This ensures we don't try to pass a stateless
    # seed to a stateful sampler, or vice versa.
    self._get_static_distribution_attributes(seed=seed)

    xs, lps = zip(
        *self._call_execute_model(
            sample_shape,
            seed=seed,
            value=self._resolve_value(value=value,
                                      allow_partially_specified=True,
                                      **kwargs),
            sample_and_trace_fn=trace_values_and_log_probs))
    return (self._model_unflatten(xs),
            self._reduce_log_probs_over_dists(lps))

  def _map_measure_over_dists(self, attr, value):
    if any(x is None for x in tf.nest.flatten(value)):
      raise ValueError('No `value` part can be `None`; saw: {}.'.format(value))

    if not callable(attr):
      attr_name = attr
      attr = lambda d, x: getattr(d, attr_name)(x)

    return self._call_execute_model(
        sample_shape=(),
        value=value,
        seed=samplers.zeros_seed(),
        sample_and_trace_fn=(
            lambda dist, value, **_: ValueWithTrace(value=value,  # pylint: disable=g-long-lambda
                                                    traced=attr(dist, value))))

  def _reduce_log_probs_over_dists(self, lps):
    """Sum computed log probs across joint distribution parts."""
    if self._experimental_use_kahan_sum:
      reduced_lps = self._reduce_measure_over_dists(
          lps, reduce_fn=tfp_math.reduce_kahan_sum)
      broadcasting_checks = maybe_check_wont_broadcast(
          [v.total for v in reduced_lps], self.validate_args)
      with tf.control_dependencies(broadcasting_checks):
        return sum(reduced_lps).total
    else:
      return sum(maybe_check_wont_broadcast(
          self._reduce_measure_over_dists(lps, reduce_fn=tf.reduce_sum),
          self.validate_args))

  def _reduce_measure_over_dists(self, xs, reduce_fn):
    if self.batch_ndims is None:
      return xs
    num_trailing_batch_dims_treated_as_event = [
        ps.rank_from_shape(d.batch_shape_tensor()) - self.batch_ndims
        for d in self._get_single_sample_distributions()]

    with tf.control_dependencies(self._maybe_check_batch_shape()):
      return [reduce_fn(unreduced_x, axis=_get_reduction_axes(unreduced_x, nd))
              for (unreduced_x, nd) in zip(
                  xs, num_trailing_batch_dims_treated_as_event)]

  def _maybe_check_batch_shape(self):
    assertions = []
    if self.validate_args:
      parts = self._batch_shape_tensor_parts()
      for s in parts[1:]:
        assertions.append(assert_util.assert_equal(
            parts[0], s, message='Component batch shapes are inconsistent.'))
    return assertions

  def _map_attr_over_dists(self, attr, dists=None):
    dists = (self._get_single_sample_distributions()
             if dists is None else dists)
    return (getattr(d, attr)() for d in dists)

  def _resolve_value(self, *args, allow_partially_specified=False, **kwargs):
    """Resolves a `value` structure from user-passed arguments."""
    value = kwargs.pop('value', None)
    if not (args or kwargs):
      # Fast path when `value` is the only kwarg. The case where `value` is
      # passed as a positional arg is handled by `_resolve_value_from_args`
      # below.
      return _sanitize_value(self, value)
    elif value is not None:
      raise ValueError('Supplied both `value` and keyword '
                       'arguments to parameterize sampling. Supplied keyword '
                       'arguments were: {}. '.format(
                           ', '.join(map(str, kwargs))))

    names = self._flat_resolve_names()
    if allow_partially_specified:
      kwargs.update({k: kwargs.get(k, None) for k in names})  # In place update.
    value, unmatched_kwargs = _resolve_value_from_args(
        args,
        kwargs,
        dtype=self.dtype,
        flat_names=names,
        model_flatten_fn=self._model_flatten,
        model_unflatten_fn=self._model_unflatten)
    if unmatched_kwargs:
      join = lambda args: ', '.join(str(j) for j in args)
      kwarg_names = join(k for k, v in kwargs.items() if v is not None)
      dist_name_str = join(names)
      unmatched_str = join(unmatched_kwargs)
      raise ValueError(
          'Found unexpected keyword arguments. Distribution names '
          'are\n{}\nbut received\n{}\nThese names were '
          'invalid:\n{}'.format(dist_name_str, kwarg_names, unmatched_str))
    return _sanitize_value(self, value)

  def _call_execute_model(self,
                          sample_shape=(),
                          seed=None,
                          value=None,
                          sample_and_trace_fn=trace_distributions_and_values):
    """Wraps the base `_call_execute_model` with vectorized_map."""
    flat_value = None if value is None else self._model_flatten(value)

    use_vectorized_map = False
    if self.use_vectorized_map:
      # Check for batch/sample dimensions so as to only incur vmap overhead if
      # it might actually be needed.
      value_might_have_sample_dims = (
          flat_value is not None and _might_have_excess_ndims(
              # Double-flatten in case any components have structured events.
              flat_value=nest.flatten_up_to(self._single_sample_ndims,
                                            flat_value,
                                            check_types=False),
              flat_core_ndims=tf.nest.flatten(self._single_sample_ndims)))
      sample_shape_may_be_nontrivial = (
          distribution_util.shape_may_be_nontrivial(sample_shape))
      use_vectorized_map = (sample_shape_may_be_nontrivial or
                            value_might_have_sample_dims)

    if not use_vectorized_map:
      return self._execute_model(
          sample_shape=sample_shape, seed=seed, value=flat_value,
          sample_and_trace_fn=sample_and_trace_fn)

    # Set up for autovectorized sampling. To support the `value` arg, we need to
    # first understand which dims are from the model itself, then wrap
    # `_call_execute_model` to batch over all remaining dims.
    flat_value_core_ndims = None
    if value is not None:
      flat_value_core_ndims = tf.nest.map_structure(
          lambda v, nd: None if v is None else nd,
          flat_value, self._single_sample_ndims,
          check_types=False)

    vectorized_execute_model_helper = vectorization_util.make_rank_polymorphic(
        lambda fv, seed: (  # pylint: disable=g-long-lambda
            self._execute_model(
                sample_shape=(),
                seed=seed,
                value=fv,
                sample_and_trace_fn=sample_and_trace_fn)),
        core_ndims=[flat_value_core_ndims, None])
    # Redefine the polymorphic fn to hack around `make_rank_polymorphic`
    # not currently supporting keyword args. This is needed because the
    # `iid_sample` wrapper below expects to pass through a `seed` kwarg.
    vectorized_execute_model = (
        lambda fv, seed: vectorized_execute_model_helper(fv, seed))  # pylint: disable=unnecessary-lambda

    if sample_shape_may_be_nontrivial:
      vectorized_execute_model = vectorization_util.iid_sample(
          vectorized_execute_model, sample_shape)

    return vectorized_execute_model(flat_value, seed=seed)

  # Override the base method to capture *args and **kwargs, so we can
  # implement more flexible custom calling semantics.
  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='log_prob', method_abbr='lp'))
  def log_prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Log probability density/mass function.

    ${calling_convention_description}

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    name = kwargs.pop('name', 'log_prob')
    return self._call_log_prob(self._resolve_value(*args, **kwargs), name=name)

  # Override the base method to capture *args and **kwargs, so we can
  # implement more flexible custom calling semantics.
  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='prob', method_abbr='prob'))
  def prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Probability density/mass function.

    ${calling_convention_description}

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    name = kwargs.pop('name', 'prob')
    return self._call_prob(self._resolve_value(*args, **kwargs), name=name)

  # Override the base method to capture *args and **kwargs, so we can
  # implement more flexible custom calling semantics.
  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='unnormalized_log_prob', method_abbr='lp'))
  def unnormalized_log_prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Unnormalized log probability density/mass function.

    ${calling_convention_description}

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    name = kwargs.pop('name', 'unnormalized_log_prob')
    return self._call_unnormalized_log_prob(
        self._resolve_value(*args, **kwargs), name=name)

  def _flat_resolve_names(self, dummy_name='var'):
    """Resolves a name for each random variable in the model."""
    names = []
    names_used = set()
    for dummy_idx, name in enumerate(
        self._get_static_distribution_attributes().name):
      if name is None:
        name = '{}{}'.format(dummy_name, dummy_idx)
      if name in names_used:
        raise ValueError('Duplicated distribution name: {}'.format(name))
      else:
        names_used.add(name)
      names.append(name)
    return names

  # We need to bypass base Distribution reshaping logic, so we
  # tactically implement the `_call_sample_n` redirector.  We don't want to
  # override the public level because then tfp.layers can't take generic
  # `Distribution.sample` as argument for the `convert_to_tensor_fn` parameter.
  def _call_sample_n(self, sample_shape, seed, value=None, **kwargs):
    return self._sample_n(
        sample_shape,
        seed=seed() if callable(seed) else seed,
        value=self._resolve_value(value=value,
                                  allow_partially_specified=True,
                                  **kwargs))

  def _execute_model(self,
                     sample_shape=(),
                     seed=None,
                     value=None,
                     stop_index=None,
                     sample_and_trace_fn=trace_distributions_and_values):
    """Executes `model`, creating both samples and distributions."""
    values_out = []
    if samplers.is_stateful_seed(seed):
      seed_stream = SeedStream(seed, salt='JointDistribution')
      if not self._stateful_to_stateless:
        seed = None
    else:
      seed_stream = None  # We got a stateless seed for seed=.

    # TODO(b/166658748): Make _stateful_to_stateless always True (eliminate it).
    if self._stateful_to_stateless and (seed is not None or not JAX_MODE):
      seed = samplers.sanitize_seed(seed, salt='JointDistribution')
    gen = self._model_coroutine()
    index = 0
    d = next(gen)
    if self._require_root:
      if distribution_util.shape_may_be_nontrivial(
          sample_shape) and not isinstance(d, self.Root):
        raise ValueError('First distribution yielded by coroutine must '
                         'be wrapped in `Root` when requesting a nontrivial '
                         f'sample_shape = {sample_shape}.')
    try:
      while True:
        actual_distribution = d.distribution if isinstance(d, self.Root) else d
        # Ensure reproducibility even when xs are (partially) set. Always split.
        stateful_sample_seed = None if seed_stream is None else seed_stream()
        if seed is None:
          stateless_sample_seed = None
        else:
          stateless_sample_seed, seed = samplers.split_seed(seed)

        value_at_index = None
        if (value is not None and len(value) > index and
            value[index] is not None):
          value_at_index = value[index]
        try:
          next_value, traced_values = sample_and_trace_fn(
              actual_distribution,
              sample_shape=sample_shape if isinstance(d, self.Root) else (),
              seed=(stateful_sample_seed if stateless_sample_seed is None
                    else stateless_sample_seed),
              value=value_at_index)
        except TypeError as e:
          if ('Expected int for argument' not in str(e) and
              TENSOR_SEED_MSG_PREFIX not in str(e)) or (
                  stateful_sample_seed is None):
            raise
          msg = (
              'Falling back to stateful sampling for distribution #{index} '
              '(0-based) of type `{dist_cls}` with component name '
              '{component_name} and `dist.name` "{dist_name}". Please '
              'update to use `tf.random.stateless_*` RNGs. This fallback may '
              'be removed after 20-Dec-2020. ({exc})')
          component_name = get_explicit_name_for_component(actual_distribution)
          if component_name is None:
            component_name = '[None specified]'
          else:
            component_name = '"{}"'.format(component_name)
          warnings.warn(msg.format(
              index=index,
              component_name=component_name,
              dist_name=actual_distribution.name,
              dist_cls=type(actual_distribution),
              exc=str(e)))
          next_value, traced_values = sample_and_trace_fn(
              actual_distribution,
              sample_shape=sample_shape if isinstance(d, self.Root) else (),
              seed=stateful_sample_seed,
              value=value_at_index)
        if self._validate_args:
          with tf.control_dependencies(
              itertools.chain.from_iterable(
                  self._assert_compatible_shape(index, sample_shape, value_part)
                  for value_part in tf.nest.flatten(next_value))):
            values_out.append(
                tf.nest.map_structure(
                    lambda x: tf.identity(x) if tf.is_tensor(x) else x,
                    traced_values))
        else:
          values_out.append(traced_values)

        index += 1
        if stop_index is not None and index == stop_index:
          break
        d = gen.send(next_value)
    except StopIteration:
      pass
    return values_out

  def _assert_compatible_shape(self, index, sample_shape, samples):
    requested_shape, _ = self._expand_sample_shape_to_vector(
        tf.convert_to_tensor(sample_shape, dtype=tf.int32),
        name='requested_shape')
    actual_shape = ps.shape(samples)
    actual_rank = ps.rank_from_shape(actual_shape)
    requested_rank = ps.rank_from_shape(requested_shape)

    # We test for two properties we expect of yielded distributions:
    # (1) The rank of the tensor of generated samples must be at least
    #     as large as the rank requested.
    # (2) The requested shape must be a prefix of the shape of the
    #     generated tensor of samples.
    # We attempt to perform test (1) statically first.
    # We don't need to do this explicitly for test (2) because
    # `assert_equal` evaluates statically if it can.
    static_actual_rank = tf.get_static_value(actual_rank)
    static_requested_rank = tf.get_static_value(requested_rank)

    assertion_message = ('Samples yielded by distribution #{} are not '
                         'consistent with `sample_shape` passed to '
                         '`JointDistributionCoroutine` '
                         'distribution.'.format(index))

    # TODO Remove this static check (b/138738650)
    if (static_actual_rank is not None and
        static_requested_rank is not None):
      # We're able to statically check the rank
      if static_actual_rank < static_requested_rank:
        raise ValueError(assertion_message)
      else:
        control_dependencies = []
    else:
      # We're not able to statically check the rank
      control_dependencies = [
          assert_util.assert_greater_equal(
              actual_rank, requested_rank,
              message=assertion_message)
          ]

    with tf.control_dependencies(control_dependencies):
      trimmed_actual_shape = actual_shape[:requested_rank]

    control_dependencies = [
        assert_util.assert_equal(
            requested_shape, trimmed_actual_shape,
            message=assertion_message)
    ]

    return control_dependencies

  def _default_event_space_bijector(self, *args, **kwargs):
    if bool(args) or bool(kwargs):
      return self.experimental_pin(
          *args, **kwargs).experimental_default_event_space_bijector()

    if self.use_vectorized_map:
      return _DefaultJointBijectorAutoBatched(self)
    return _DefaultJointBijector(self)

  def experimental_pin(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Pins some parts, returning an unnormalized distribution object.

    The calling convention is much like other `JointDistribution` methods (e.g.
    `log_prob`), but with the difference that not all parts are required. In
    this respect, the behavior is similar to that of the `sample` function's
    `value` argument.

    ### Examples:

    ```
    # Given the following joint distribution:
    jd = tfd.JointDistributionSequential([
        tfd.Normal(0., 1., name='z'),
        tfd.Normal(0., 1., name='y'),
        lambda y, z: tfd.Normal(y + z, 1., name='x')
    ], validate_args=True)

    # The following calls are all permissible and produce
    # `JointDistributionPinned` objects behaving identically.
    PartialXY = collections.namedtuple('PartialXY', 'x,y')
    PartialX = collections.namedtuple('PartialX', 'x')
    assert (jd.experimental_pin(x=2.).pins ==
            jd.experimental_pin(x=2., z=None).pins ==
            jd.experimental_pin(dict(x=2.)).pins ==
            jd.experimental_pin(dict(x=2., y=None)).pins ==
            jd.experimental_pin(PartialXY(x=2., y=None)).pins ==
            jd.experimental_pin(PartialX(x=2.)).pins ==
            jd.experimental_pin(None, None, 2.).pins ==
            jd.experimental_pin([None, None, 2.]).pins)
    ```

    Args:
      *args: Positional arguments: a value structure or component values (see
        above).
      **kwargs: Keyword arguments: a value structure or component values (see
        above). May also include `name`, specifying a Python string name for ops
        generated by this method.

    Returns:
      pinned: a `tfp.experimental.distributions.JointDistributionPinned` with
        the given values pinned.
    """
    # Import deferred to avoid circular dependency
    # JD.experimental_pin <> JDPinned <> experimental.marginalize.JDCoroutine.
    from tensorflow_probability.python.experimental.distributions import joint_distribution_pinned as jd_pinned  # pylint: disable=g-import-not-at-top
    return jd_pinned.JointDistributionPinned(self, *args, **kwargs)


def get_explicit_name_for_component(d):
  """Returns the explicitly-passed `name` of a Distribution, or None."""
  if hasattr(d, '_build_module'):
    # Ensure we inspect the distribution itself, not a DeferredModule wrapper.
    d = d._build_module()  # pylint: disable=protected-access
  name = d.parameters.get('name', None)
  if name and d.__class__.__name__ in name:
    name = None

  if name and hasattr(d, '__init__'):
    spec = tf_inspect.getfullargspec(d.__init__)
    default_name = dict(
        zip(spec.args[len(spec.args) - len(spec.defaults or ()):],
            spec.defaults or ())
        ).get('name', None)
    if name == default_name:
      name = None

  if name in FORBIDDEN_COMPONENT_NAMES:
    raise ValueError('Distribution name "{}" is not allowed as a '
                     'JointDistribution component; please choose a different '
                     'name.'.format(name))
  return name


def _resolve_value_from_args(args,
                             kwargs,
                             dtype,
                             flat_names,
                             model_flatten_fn,
                             model_unflatten_fn):
  """Resolves a `value` structure matching `dtype` from a function call.

  This offers semantics equivalent to a Python callable `f(x1, x2, ..., xN)`,
  where `'x1', 'x2', ..., 'xN' = self._flat_resolve_names()` are the names of
  the model's component distributions. Arguments may be passed by position
  (`f(1., 2., 3.)`), by name (`f(x1=1., x2=2., x3=3.)`), or by a combination
  of approaches (`f(1., 2., x3=3.)`).

  Passing a `value` structure directly (as in `jd.log_prob(jd.sample())`) is
  supported by an optional `value` kwarg (`f(value=[1., 2., 3.])`), or by
  simply passing the value as the sole positional argument
  (`f([1., 2., 3.])`). For models having only a single component, a single
  positional argument that matches the structural type (e.g., a single Tensor,
  or a nested list or dict of Tensors) of that component is interpreted as
  specifying it; otherwise a single positional argument is interpreted as
  the overall `value`.

  Args:
    args: Positional arguments passed to the function being called.
    kwargs: Keyword arguments passed to the function being called.
    dtype: Nested structure of `dtype`s of model components.
    flat_names: Iterable of Python `str` names of model components.
    model_flatten_fn: Python `callable` that takes a structure and returns a
      list representing the flattened structure.
    model_unflatten_fn: Python `callable` that takes an iterable and returns a
      structure.
  Returns:
    value: A structure in which the observed arguments are arranged to match
      `dtype`.
    unmatched_kwargs: Python `dict` containing any keyword arguments that don't
      correspond to model components.
  Raises:
    ValueError: if the number of args passed doesn't match the number of
      model components, or if positional arguments are passed to a dict-valued
      distribution.
  """

  value = kwargs.pop('value', None)
  if value is not None:  # Respect 'value' as an explicit kwarg.
    return value, kwargs

  matched_kwargs = {k for k in flat_names if k in kwargs}
  unmatched_kwargs = {k: v for (k, v) in kwargs.items()
                      if k not in matched_kwargs}

  # If we have only a single positional arg, we need to disambiguate it by
  # examining the model structure.
  if len(args) == 1 and not matched_kwargs:
    if len(dtype) > 1:  # Model has multiple variables; arg must be a structure.
      return args[0], unmatched_kwargs
    # Otherwise the model has one variable. If its structure matches the arg,
    # interpret the arg as its value.
    first_component_dtype = model_flatten_fn(dtype)[0]
    try:
      # TODO(davmre): this assertion will falsely trigger if args[0] contains
      # nested lists that the user intends to be converted to Tensor. We should
      # try to relax it slightly (without creating false negatives).
      tf.nest.assert_same_structure(
          first_component_dtype, args[0], check_types=False)
      return model_unflatten_fn(args), unmatched_kwargs
    except (ValueError, TypeError):     # If RV doesn't match the arg, interpret
      return args[0], unmatched_kwargs  # the arg as a 'value' structure.

  num_components_specified = len(args) + len(kwargs) - len(unmatched_kwargs)
  if num_components_specified != len(flat_names):
    raise ValueError('Joint distribution expected values for {} components {}; '
                     'saw {} (from args {} and kwargs {}).'.format(
                         len(flat_names),
                         flat_names,
                         num_components_specified,
                         args,
                         kwargs))

  if args and (isinstance(dtype, dict) and not
               isinstance(dtype, collections.OrderedDict)):
    raise ValueError("Joint distribution with unordered variables can't "
                     "take positional args (saw {}).".format(args))

  value = model_unflatten_fn(kwargs[k] if k in kwargs else args[i]
                             for i, k in enumerate(flat_names))
  return value, unmatched_kwargs


def _get_reduction_axes(x, nd):
  """Enumerates the final `nd` axis indices of `x`."""
  x_rank = ps.rank(x)
  return ps.range(x_rank - 1, x_rank - nd - 1, -1)


def _might_have_excess_ndims(flat_value, flat_core_ndims):
  for v, nd in zip(flat_value, flat_core_ndims):
    static_excess_ndims = (
        0 if v is None else
        tf.get_static_value(ps.convert_to_shape_tensor(ps.rank(v) - nd)))
    if static_excess_ndims is None or static_excess_ndims > 0:
      return True
  return False


def maybe_check_wont_broadcast(flat_xs, validate_args):
  """Verifies that `parts` don't broadcast."""
  flat_xs = tuple(flat_xs)  # So we can receive generators.
  if not validate_args:
    # Note: we don't try static validation because it is theoretically
    # possible that a user wants to take advantage of broadcasting.
    # Only when `validate_args` is `True` do we enforce the validation.
    return flat_xs
  msg = 'Broadcasting probably indicates an error in model specification.'
  s = tuple(ps.shape(x) for x in flat_xs)
  def same_shape(a, b):
    # Can't just use np.all(a == b) because numpy's == may broadcast!
    # For instance, [3] == [3, 3] is [True, True], but that's not what
    # we want here.
    return len(a) == len(b) and np.all(a == b)
  if all(ps.is_numpy(s_) for s_ in s):
    if not all(same_shape(a, b) for a, b in zip(s[1:], s[:-1])):
      raise ValueError(msg)
    return flat_xs
  assertions = [assert_util.assert_equal(a, b, message=msg)
                for a, b in zip(s[1:], s[:-1])]
  with tf.control_dependencies(assertions):
    return tuple(tf.identity(x) for x in flat_xs)


# pylint: disable=protected-access
class _DefaultJointBijector(composition.Composition):
  """Minimally-viable event space bijector for `JointDistribution`."""

  def __init__(self, jd, parameters=None, bijector_fn=None):
    parameters = dict(locals()) if parameters is None else parameters

    with tf.name_scope('default_joint_bijector') as name:
      if bijector_fn is None:
        bijector_fn = lambda d: d.experimental_default_event_space_bijector()
      bijectors = tuple(bijector_fn(d)
                        for d in jd._get_single_sample_distributions())
      i_min_event_ndims = tf.nest.map_structure(
          ps.size, jd.event_shape)
      f_min_event_ndims = jd._model_unflatten([
          b.inverse_event_ndims(nd) for b, nd in
          zip(bijectors, jd._model_flatten(i_min_event_ndims))])
      super(_DefaultJointBijector, self).__init__(
          bijectors=bijectors,
          forward_min_event_ndims=f_min_event_ndims,
          inverse_min_event_ndims=i_min_event_ndims,
          validate_args=jd.validate_args,
          parameters=parameters,
          name=name)
      self._jd = jd
      self._bijector_fn = bijector_fn

  def _conditioned_bijectors(self, samples, constrained=False):
    if samples is None:
      return self.bijectors

    bijectors = []
    gen = self._jd._model_coroutine()
    cond = None
    for rv in self._jd._model_flatten(samples):
      d = gen.send(cond)
      dist = d.distribution if type(d).__name__ == 'Root' else d
      bij = self._bijector_fn(dist)

      if bij is None:
        bij = identity_bijector.Identity()
      bijectors.append(bij)

      # If the RV is not yet constrained, transform it.
      cond = rv if constrained else bij.forward(rv)
    return bijectors

  @property
  def _parts_interact(self):
    # The bijector that operates on input part B may in general be a
    # function of input part A. This dependence is not visible to the
    # Composition base class, so we annotate it explicitly.
    return True

  def _walk_forward(self, step_fn, values, _jd_conditioning=None):  # pylint: disable=invalid-name
    bijectors = self._conditioned_bijectors(_jd_conditioning, constrained=False)
    return self._jd._model_unflatten(tuple(
        step_fn(bij, value) for bij, value in
        zip(bijectors, self._jd._model_flatten(values))))

  def _walk_inverse(self, step_fn, values, _jd_conditioning=None):  # pylint: disable=invalid-name
    bijectors = self._conditioned_bijectors(_jd_conditioning, constrained=True)
    return self._jd._model_unflatten(tuple(
        step_fn(bij, value) for bij, value
        in zip(bijectors, self._jd._model_flatten(values))))

  def _forward(self, x, **kwargs):
    return super(_DefaultJointBijector, self)._forward(
        x, _jd_conditioning=x, **kwargs)

  def _forward_log_det_jacobian(self, x, event_ndims, **kwargs):
    return super(_DefaultJointBijector, self)._forward_log_det_jacobian(
        x, event_ndims, _jd_conditioning=x, **kwargs)

  def _inverse(self, y, **kwargs):
    return super(_DefaultJointBijector, self)._inverse(
        y, _jd_conditioning=y, **kwargs)

  def _inverse_log_det_jacobian(self, y, event_ndims, **kwargs):
    return super(_DefaultJointBijector, self)._inverse_log_det_jacobian(
        y, event_ndims, _jd_conditioning=y, **kwargs)


class _DefaultJointBijectorAutoBatched(bijector_lib.Bijector):
  """Automatically vectorized support bijector for autobatched JDs."""

  def __init__(self, jd, **kwargs):
    parameters = dict(locals())
    self._jd = jd
    self._bijector_kwargs = kwargs
    self._joint_bijector = _DefaultJointBijector(
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
    # We avoid this by specifying that `y` has core ndims matching
    # `jd.sample().shape.ndims`.
    jd_batch_ndims = ps.rank_from_shape(jd.batch_shape_tensor(), jd.batch_shape)
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


@ldj_ratio_lib.RegisterFLDJRatio(_DefaultJointBijector)
def _fldj_ratio_jd_default(p, x, q, y, event_ndims, p_kwargs, q_kwargs):
  return composition._fldj_ratio_composition(p, x, q, y, event_ndims, {
      '_jd_conditioning': x,
      **p_kwargs
  }, {
      '_jd_conditioning': y,
      **q_kwargs
  })


@ldj_ratio_lib.RegisterILDJRatio(_DefaultJointBijector)
def _ildj_ratio_jd_default(p, x, q, y, event_ndims, p_kwargs, q_kwargs):
  return composition._ildj_ratio_composition(p, x, q, y, event_ndims, {
      '_jd_conditioning': x,
      **p_kwargs
  }, {
      '_jd_conditioning': y,
      **q_kwargs
  })


def _sanitize_value(distribution, value):
  """Ensures `value` matches `distribution.dtype`, adding `None`s as needed."""
  if value is None:
    return value

  if not tf.nest.is_nested(distribution.dtype):
    return tf.convert_to_tensor(value, dtype_hint=distribution.dtype)

  if len(value) < len(distribution.dtype):
    # Fill in missing entries with `None`.
    if hasattr(distribution.dtype, 'keys'):
      value = {k: value.get(k, None) for k in distribution.dtype.keys()}
    else:  # dtype is a sequence.
      value = [value[i] if i < len(value) else None
               for i in range(len(distribution.dtype))]

  value = nest_util.cast_structure(value, distribution.dtype)
  jdlike_attrs = [
      '_get_single_sample_distributions',
      '_model_flatten',
      '_model_unflatten',
  ]
  if all(hasattr(distribution, attr) for attr in jdlike_attrs):
    flat_dists = distribution._get_single_sample_distributions()
    flat_value = distribution._model_flatten(value)
    flat_value = map(_sanitize_value, flat_dists, flat_value)
    return distribution._model_unflatten(flat_value)
  else:
    # A joint distribution that isn't tfd.JointDistribution-like; assume it has
    # some reasonable dtype semantics. We can't use this for
    # tfd.JointDistribution because we might have a None standing in for a
    # sub-tree (e.g. consider omitting a nested JD).
    return nest.map_structure_up_to(
        distribution.dtype,
        lambda x, d: x if x is None else tf.convert_to_tensor(x, dtype_hint=d),
        value,
        distribution.dtype,
    )


@log_prob_ratio.RegisterLogProbRatio(JointDistribution)
def _jd_log_prob_ratio(p, x, q, y, name=None):
  """Implements `log_prob_ratio` for tfd.JointDistribution*."""
  with tf.name_scope(name or 'jd_log_prob_ratio'):
    tf.nest.assert_same_structure(x, y)
    p_dists, _ = p.sample_distributions(value=x, seed=samplers.zeros_seed())
    q_dists, _ = q.sample_distributions(value=y, seed=samplers.zeros_seed())
    tf.nest.assert_same_structure(p_dists, q_dists)
    log_prob_ratio_parts = nest.map_structure_up_to(
        p_dists, log_prob_ratio.log_prob_ratio, p_dists, x, q_dists, y)
    return tf.add_n(tf.nest.flatten(log_prob_ratio_parts))
