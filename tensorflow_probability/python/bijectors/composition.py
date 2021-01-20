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
"""Composition base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import sys

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'Composition',
]


JAX_MODE = False


def pack_structs_like(template, *structures):
  """Converts a tuple of structs like `template` to a structure of tuples."""
  if not structures:
    return nest.map_structure(lambda x: (), template)
  return nest.map_structure_up_to(template, (lambda *args: args),
                                  *structures, check_types=False)


def unpack_structs_like(template, packed):
  """Converts a structure of tuples like `template` to a tuple of structures."""
  return tuple(nest.pack_sequence_as(template, flat) for flat in
               zip(*nest.flatten_up_to(template, packed, check_types=False)))


def _event_size(tensor_structure, event_ndims):
  """Returns the number of elements in the event-portion of a structure."""
  event_shapes = nest.map_structure(
      lambda t, nd: ps.slice(ps.shape(t), [ps.rank(t)-nd], [nd]),
      tensor_structure, event_ndims)
  return sum(ps.reduce_prod(shape) for shape in nest.flatten(event_shapes))


def _max_precision_sum(a, b):
  """Coerces `a` or `b` to the higher-precision dtype, and returns the sum."""
  if not dtype_util.base_equal(a.dtype, b.dtype):
    if dtype_util.size(a.dtype) >= dtype_util.size(b.dtype):
      b = tf.cast(b, a.dtype)
    else:
      a = tf.cast(a, b.dtype)
  return a + b


class Composition(bijector.Bijector):
  """Base class for Composition bijectors (Chain, JointMap).

  A Composition represents a partially ordered set of invertible
  transformations. These transformations may happen in series (Chain), in
  parallel (JointMap), or they could be an arbitrary DAG. Composition handles
  the common machinery of such transformations, delegating graph-traversal to
  `_walk_forward` and `_walk_inverse` (which must be overridden by subclasses).

  The `_walk_{direction}` methods take a `step_fn`, a single (structured)
  `argument` (representing zipped `*args`), and arbitrary `**kwargs`. They are
  responsible for invoking `step_fn(bij, bij_inputs, **bij_kwds)`
  for each nested bijector. See `Chain` and `JointMap` for examples.

  These methods are typically invoked using `_call_walk_{direction}`, which
  wraps `step_fn` and converts structured `*args` into a single structure of
  tuples, allowing users to provide a `step_fn` with multiple positional
  arguments (e.g., `foward_log_det_jacobian`).

  In practice, Bijector methods are defined in the base-class, and users
  should not need to invoke `walk` methods directly.
  """

  def __init__(self,
               bijectors,
               forward_min_event_ndims,
               inverse_min_event_ndims,
               name,
               parameters,
               validate_event_size=False,
               **kwargs):
    """Instantiates a Composition of bijectors.

    Args:
      bijectors: A nest-compatible structure of bijector instances.
      forward_min_event_ndims: A (structure of) integer describing both the
        multi-part structure of inputs to `forward` and the _aligned_ mininimum
        valid event-ndims. Compositions that allow different relative ranks
        should pass structures of `None`.
      inverse_min_event_ndims: A (structure of) integer describing both the
        multi-part structure of inputs to `inverse` and the _aligned_ mininimum
        valid event-ndims. Compositions that allow different relative ranks
        should pass structures of `None`.
      name: Name of this bijector.
      parameters: Dictionary of parameters used to initialize this bijector.
        These must be the exact values passed to `__init__`.
      validate_event_size: Checks that bijectors are not applied to inputs with
        incomplete support. For example, the following LDJ would be incorrect:
        `Chain([Scale(), SoftmaxCentered()]).forward_log_det_jacobian([1], [1])`
        The jacobian contribution from `Scale` applies to a 2-dimensional input,
        but the output from `SoftMaxCentered` is a 1-dimensional input embedded
        in a 2-dimensional space. Setting `validate_event_size=True` (default)
        prints warnings in these cases. When `validate_args` is also `True`, the
        warning is promoted to an exception.
      **kwargs: Additional parameters forwarded to the bijector base-class.
    """

    with tf.name_scope(name):
      is_constant_jacobian = True
      is_injective = True
      is_permutation = True
      for bij in nest.flatten(bijectors):
        is_injective &= bij._is_injective
        is_constant_jacobian &= bij.is_constant_jacobian
        is_permutation &= bij._is_permutation

      super(Composition, self).__init__(
          forward_min_event_ndims=forward_min_event_ndims,
          inverse_min_event_ndims=inverse_min_event_ndims,
          is_constant_jacobian=is_constant_jacobian,
          parameters=parameters,
          name=name,
          **kwargs)

      # Copy the nested structure so we don't mutate arguments during tracking.
      self._bijectors = nest.map_structure(lambda b: b, bijectors)
      self._validate_event_size = validate_event_size
      self.__is_injective = is_injective
      self.__is_permutation = is_permutation

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  @property
  def bijectors(self):
    return self._bijectors

  @property
  def validate_event_size(self):
    return self._validate_event_size

  @property
  def _is_injective(self):
    return self.__is_injective

  @property
  def _is_permutation(self):
    return self.__is_permutation

  # pylint: disable=redefined-builtin

  def _call_walk_forward(self, step_fn, *args, **kwargs):
    """Prepares args and calls `_walk_forward`.

    Converts a tuple of structured positional arguments to a structure of
    argument tuples, and wraps `step_fn` to unpack inputs and re-pack
    returned values. This way, users may invoke walks using `map_structure`
    semantics, and the concrete `_walk` implementations can operate on
    a single structure of inputs (without worrying about tuple unpacking).


    For example, the `forward` method looks roughly like:
    ```python

    MyComposition()._call_walk_forward(
        lambda bij, x, **kwargs: bij.forward(x, **kwargs),
        composite_inputs, **composite_kwargs)
    ```

    More complex methods may need to mutate external state from `step_fn`:
    ```python

    shape_trace = {}

    def trace_step(bijector, x_shape):
      shape_trace[bijector.name] = x_shape
      return bijector.forward_event_shape(x_shape)

    # Calling this populates the `shape_trace` dictionary
    composition.walk_forward(trace_step, composite_input_shape)
    ```

    Args:
      step_fn: Callable applied to each wrapped bijector.
        Must accept a bijector instance followed by `len(args)` positional
        arguments whose structures match `bijector.forward_min_event_ndims`,
        and return `len(args)` structures matching
        `bijector.inverse_min_event_ndims`.
      *args: Input arguments propagated to nested bijectors.
      **kwargs: Keyword arguments forwarded to `_walk_forward`.
    Returns:
      The transformed output. If multiple positional arguments are provided, a
        tuple of matching length will be returned.
    """
    args = tuple(nest_util.coerce_structure(self.forward_min_event_ndims, x)
                 for x in args)

    if len(args) == 1:
      return self._walk_forward(step_fn, *args, **kwargs)

    # Convert a tuple of structures to a structure of tuples. This
    # allows `_walk` methods to route aligned structures of inputs/outputs
    # independently, obviates the need for conditional tuple unpacking.
    packed_args = pack_structs_like(self.forward_min_event_ndims, *args)

    def transform_wrapper(bij, packed_xs, **nested):
      xs = unpack_structs_like(bij.forward_min_event_ndims, packed_xs)
      ys = step_fn(bij, *xs, **nested)
      return pack_structs_like(bij.inverse_min_event_ndims, *ys)

    packed_result = self._walk_forward(
        transform_wrapper, packed_args, **kwargs)
    return unpack_structs_like(self.inverse_min_event_ndims, packed_result)

  def _call_walk_inverse(self, step_fn, *args, **kwargs):
    """Prepares args and calls `_walk_inverse`.

    Converts a tuple of structured positional arguments to a structure of
    argument tuples, and wraps `step_fn` to unpack inputs and re-pack
    returned values. This way, users may invoke walks using `map_structure`
    semantics, and the concrete `_walk` implementations can operate on
    single-structure of inputs (without worrying about tuple unpacking).

    For example, the `inverse` method looks roughly like:
    ```python

    MyComposition()._call_walk_inverse(
        lambda bij, y, **kwargs: bij.inverse(y, **kwargs),
        composite_inputs, **composite_kwargs)
    ```

    More complex methods may need to mutate external state from `step_fn`:
    ```python

    shape_trace = {}

    def trace_step(bijector, y_shape):
      shape_trace[bijector.name] = y_shape
      return bijector.inverse_event_shape(y_shape)

    # Calling this populates the `shape_trace` dictionary
    composition.walk_forward(trace_step, composite_y_shape)
    ```

    Args:
      step_fn: Callable applied to each wrapped bijector.
        Must accept a bijector instance followed by `len(args)` positional
        arguments whose structures match `bijector.inverse_min_event_ndims`,
        and return `len(args)` structures matching
        `bijector.forward_min_event_ndims`.
      *args: Input arguments propagated to nested bijectors.
      **kwargs: Keyword arguments forwarded to `_walk_inverse`.
    Returns:
      The transformed output. If multiple positional arguments are provided, a
        tuple of matching length will be returned.
    """
    args = tuple(nest_util.coerce_structure(self.inverse_min_event_ndims, y)
                 for y in args)

    if len(args) == 1:
      return self._walk_inverse(step_fn, *args, **kwargs)

    # Convert a tuple of structures to a structure of tuples. This
    # allows `_walk` methods to route aligned structures of inputs/outputs
    # independently, obviates the need for conditional tuple unpacking.
    packed_args = pack_structs_like(self.inverse_min_event_ndims, *args)

    def transform_wrapper(bij, packed_ys, **nested):
      ys = unpack_structs_like(bij.inverse_min_event_ndims, packed_ys)
      xs = step_fn(bij, *ys, **nested)
      return pack_structs_like(bij.forward_min_event_ndims, *xs)

    packed_result = self._walk_inverse(
        transform_wrapper, packed_args, **kwargs)
    return unpack_structs_like(self.forward_min_event_ndims, packed_result)

  ### Abstract Methods

  @abc.abstractmethod
  def _walk_forward(self, step_fn, argument, **kwargs):
    """Subclass stub for forward-mode traversals.

    The `_walk_{direction}` methods define how arguments are routed through
    nested bijectors, expressing the directed topology of the underlying graph.

    Args:
      step_fn: A method taking a bijector, a single positional argument
        matching `bijector.forward_min_event_ndims`, and arbitrary **kwargs,
        and returning a structure matching `bijector.inverse_min_event_ndims`.
        In cases where multiple structured inputs are required, use
        `_call_walk_forward` instead.
      argument: A (structure of) Tensor matching `self.forward_min_event_ndims`.
      **kwargs: Keyword arguments to be forwarded to nested bijectors.
    """
    raise NotImplementedError('{}._walk_forward is not implemented'.format(
        type(self).__name__))

  @abc.abstractmethod
  def _walk_inverse(self, step_fn, argument, **kwargs):
    """Subclass stub for inverse-mode traversals.

    The `_walk_{direction}` methods define how arguments are routed through
    nested bijectors, expressing the directed topology of the underlying graph.

    Args:
      step_fn: A method taking a bijector, a single positional argument
        matching `bijector.inverse_min_event_ndims`, and arbitrary **kwargs,
        and returning a structure matching `bijector.forward_min_event_ndims`.
        In cases where multiple structured inputs are required, use
        `_call_walk_inverse` instead.
      argument: A (structure of) Tensor matching `self.inverse_min_event_ndims`.
      **kwargs: Keyword arguments to be forwarded to nested bijectors.
    """
    raise NotImplementedError('{}._walk_inverse is not implemented'.format(
        type(self).__name__))

  ###
  ### Nontrivial Methods
  ###

  ## LDJ Methods

  # DAGs of bijectors do not generally have statically-known `min_event_ndims`
  # in the way that most other bijectors do.

  # Consider a single bijector that applies Exp() to a 2-tuple of Tensors with
  # shapes `[2, 3]` and `[3, 2]`. Valid values for `event_ndims` may have
  # different relative-ranks (e.g., `(2,2)` and `(1,0)`). Meanwhile,
  # passing `(1,1)` would result in a broadcasting exception. This being the
  # case, we cannot return a "minimally-reduced LDJ" without knowing both the
  # event-dimensionality _and_ the shapes of inputs. As such, we forego
  # intermediate LDJ caching entirely, and request fully-reduced LDJ from nested
  # bijectors. This requires us to change the signature of
  # `_{direction}_log_det_jacobian` to include `event_ndims`.

  def _call_forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
    """Compute forward_log_det_jacobian over the composition."""
    with self._name_and_control_scope(name):
      dtype = self.inverse_dtype(**kwargs)
      x = nest_util.convert_to_nested_tensor(
          x, name='x', dtype_hint=dtype,
          dtype=None if bijector.SKIP_DTYPE_CHECKS else dtype,
          allow_packing=True)
      if event_ndims is None:
        if self._has_static_min_event_ndims:
          event_ndims = self.forward_min_event_ndims
        else:
          raise ValueError('Composition bijector with non-static '
                           '`min_event_ndims` does not support '
                           '`event_ndims=None`. Please pass a value '
                           'for `event_ndims`.')
      event_ndims = nest_util.coerce_structure(
          self.forward_min_event_ndims, event_ndims)
      return self._forward_log_det_jacobian(x, event_ndims, **kwargs)

  def _forward_log_det_jacobian(self, x, event_ndims, **kwargs):
    # Container for accumulated LDJ.
    ldj_sum = tf.zeros([], dtype=tf.float32)
    # Container for accumulated assertions.
    assertions = []

    def step(bij, x, x_event_ndims, increased_dof, **kwargs):  # pylint: disable=missing-docstring
      nonlocal ldj_sum

      # Compute the LDJ for this step, and add it to the rolling sum.
      component_ldj = tf.convert_to_tensor(
          bij.forward_log_det_jacobian(x, x_event_ndims, **kwargs),
          dtype_hint=ldj_sum.dtype)

      if not dtype_util.is_floating(component_ldj.dtype):
        raise TypeError(('Nested bijector "{}" of Composition "{}" returned '
                         'LDJ with a non-floating dtype: {}')
                        .format(bij.name, self.name, component_ldj.dtype))
      ldj_sum = _max_precision_sum(ldj_sum, component_ldj)

      # Transform inputs for the next bijector.
      y = bij.forward(x, **kwargs)
      y_event_ndims = bij.forward_event_ndims(x_event_ndims, **kwargs)

      # Check if the inputs to this bijector have increased degrees of freedom
      # due to some upstream bijector. We assume that the upstream bijector
      # produced a valid LDJ, but this one does not (unless LDJ is 0, in which
      # case it doesn't matter).
      increased_dof = ps.reduce_any(nest.flatten(increased_dof))
      if self.validate_event_size:
        assertions.append(self._maybe_warn_increased_dof(
            component_name=bij.name,
            component_ldj=component_ldj,
            increased_dof=increased_dof))
        increased_dof |= (_event_size(y, y_event_ndims)
                          > _event_size(x, x_event_ndims))

      increased_dof = nest_util.broadcast_structure(y, increased_dof)
      return y, y_event_ndims, increased_dof

    increased_dof = nest_util.broadcast_structure(event_ndims, False)
    self._call_walk_forward(step, x, event_ndims, increased_dof, **kwargs)
    with tf.control_dependencies([x for x in assertions if x is not None]):
      return tf.identity(ldj_sum, name='fldj')

  def _call_inverse_log_det_jacobian(self, y, event_ndims, name, **kwargs):
    """Compute inverse_log_det_jacobian over the composition."""
    with self._name_and_control_scope(name):
      dtype = self.forward_dtype(**kwargs)
      y = nest_util.convert_to_nested_tensor(
          y, name='y', dtype_hint=dtype,
          dtype=None if bijector.SKIP_DTYPE_CHECKS else dtype,
          allow_packing=True)
      if event_ndims is None:
        if self._has_static_min_event_ndims:
          event_ndims = self.inverse_min_event_ndims
        else:
          raise ValueError('Composition bijector with non-static '
                           '`min_event_ndims` does not support '
                           '`event_ndims=None`. Please pass a value '
                           'for `event_ndims`.')
      event_ndims = nest_util.coerce_structure(
          self.inverse_min_event_ndims, event_ndims)
      return self._inverse_log_det_jacobian(y, event_ndims, **kwargs)

  def _inverse_log_det_jacobian(self, y, event_ndims, **kwargs):
    # Container for accumulated LDJ.
    ldj_sum = tf.convert_to_tensor(0., dtype=tf.float32)
    # Container for accumulated assertions.
    assertions = []

    def step(bij, y, y_event_ndims, increased_dof=False, **kwargs):  # pylint: disable=missing-docstring
      nonlocal ldj_sum

      # Compute the LDJ for this step, and add it to the rolling sum.
      component_ldj = tf.convert_to_tensor(
          bij.inverse_log_det_jacobian(y, y_event_ndims, **kwargs),
          dtype_hint=ldj_sum.dtype)

      if not dtype_util.is_floating(component_ldj.dtype):
        raise TypeError(('Nested bijector "{}" of Composition "{}" returned '
                         'LDJ with a non-floating dtype: {}')
                        .format(bij.name, self.name, component_ldj.dtype))
      ldj_sum = _max_precision_sum(ldj_sum, component_ldj)

      # Transform inputs for the next bijector.
      x = bij.inverse(y, **kwargs)
      x_event_ndims = bij.inverse_event_ndims(y_event_ndims, **kwargs)

      # Check if the inputs to this bijector have increased degrees of freedom
      # due to some upstream bijector. We assume that the upstream bijector
      # produced a valid LDJ, but this one does not (unless LDJ is 0, in which
      # case it doesn't matter).
      increased_dof = ps.reduce_any(nest.flatten(increased_dof))
      if self.validate_event_size:
        assertions.append(self._maybe_warn_increased_dof(
            component_name=bij.name,
            component_ldj=component_ldj,
            increased_dof=increased_dof))
        increased_dof |= (_event_size(x, x_event_ndims)
                          > _event_size(y, y_event_ndims))

      increased_dof = nest_util.broadcast_structure(x, increased_dof)
      return x, x_event_ndims, increased_dof

    increased_dof = nest_util.broadcast_structure(event_ndims, False)
    self._call_walk_inverse(step, y, event_ndims, increased_dof, **kwargs)
    with tf.control_dependencies([x for x in assertions if x is not None]):
      return tf.identity(ldj_sum, name='ildj')

  def _maybe_warn_increased_dof(self,
                                component_name,
                                component_ldj,
                                increased_dof):
    """Warns or raises when `increased_dof` is True."""
    # Short-circuit when the component LDJ is statically zero.
    if (tf.get_static_value(tf.rank(component_ldj)) == 0
        and tf.get_static_value(component_ldj) == 0):
      return

    # Short-circuit when increased_dof is statically False.
    increased_dof_ = tf.get_static_value(increased_dof)
    if increased_dof_ is False:  # pylint: disable=g-bool-id-comparison
      return

    error_message = (
        'Nested component "{}" in composition "{}" operates on inputs '
        'with increased degrees of freedom. This may result in an '
        'incorrect log_det_jacobian.'
        ).format(component_name, self.name)

    # When validate_args is True, we raise on increased DoF.
    if self._validate_args:
      if increased_dof_:
        raise ValueError(error_message)
      return assert_util.assert_equal(False, increased_dof, error_message)

    if (not tf.executing_eagerly() and
        control_flow_util.GraphOrParentsInXlaContext(tf1.get_default_graph())):
      return  # No StringFormat or Print ops in XLA.

    # Otherwise, we print a warning and continue.
    return ps.cond(
        pred=increased_dof,
        false_fn=tf.no_op,
        true_fn=lambda: tf.print(  # pylint: disable=g-long-lambda
            'WARNING: ' + error_message, output_stream=sys.stderr))

  ###
  ### Trivial traversals
  ###

  def _forward(self, x, **kwargs):
    return self._call_walk_forward(
        lambda b, x, **kwargs: b.forward(x, **kwargs),
        x, **kwargs)

  def _inverse(self, y, **kwargs):
    if not self._is_injective:  # pylint: disable=protected-access
      raise NotImplementedError(
          'Invert is not implemented for compositions of '
          'non-injective bijectors.')
    return self._call_walk_inverse(
        lambda b, y, **kwargs: b.inverse(y, **kwargs),
        y, **kwargs)

  def _forward_event_shape_tensor(self, x, **kwargs):
    return self._call_walk_forward(
        lambda b, x, **kwds: b.forward_event_shape_tensor(x, **kwds),
        x, **kwargs)

  def _inverse_event_shape_tensor(self, y, **kwargs):
    return self._call_walk_inverse(
        lambda b, y, **kwds: b.inverse_event_shape_tensor(y, **kwds),
        y, **kwargs)

  def _forward_event_shape(self, x, **kwargs):
    return self._call_walk_forward(
        lambda b, x, **kwds: b.forward_event_shape(x, **kwds),
        x, **kwargs)

  def _inverse_event_shape(self, y, **kwargs):
    return self._call_walk_inverse(
        lambda b, y, **kwds: b.inverse_event_shape(y, **kwds),
        y, **kwargs)

  def _forward_dtype(self, x, **kwargs):
    return self._call_walk_forward(
        lambda b, x, **kwds: b.forward_dtype(x, **kwds),
        x, **kwargs)

  def _inverse_dtype(self, y, **kwargs):
    return self._call_walk_inverse(
        lambda b, y, **kwds: b.inverse_dtype(y, **kwds),
        y, **kwargs)

  def forward_event_ndims(self, event_ndims, **kwargs):
    if self._has_static_min_event_ndims:
      return super(Composition, self).forward_event_ndims(event_ndims, **kwargs)
    return self._call_walk_forward(
        lambda b, nd, **kwds: b.forward_event_ndims(nd, **kwds),
        event_ndims, **kwargs)

  def inverse_event_ndims(self, event_ndims, **kwargs):
    if self._has_static_min_event_ndims:
      return super(Composition, self).inverse_event_ndims(event_ndims, **kwargs)
    return self._call_walk_inverse(
        lambda b, nd, **kwds: b.inverse_event_ndims(nd, **kwds),
        event_ndims, **kwargs)
