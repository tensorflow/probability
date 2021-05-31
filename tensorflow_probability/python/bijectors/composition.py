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

import collections
import functools
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


MinEventNdimsInferenceDownstreamQuantities = collections.namedtuple(
    'MinEventNdimsInferenceDownstreamQuantities',
    ['forward_min_event_ndims',
     'parts_interact'])


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
               name,
               parameters,
               forward_min_event_ndims=None,
               inverse_min_event_ndims=None,
               validate_event_size=False,
               **kwargs):
    """Instantiates a Composition of bijectors.

    Args:
      bijectors: A nest-compatible structure of bijector instances or a
        `Composition` bijector. If `bijectors` is a nested structure, then
        `_walk_forward` and `_walk_inverse` must be implemented. If `bijectors`
        is a `Composition` bijector, `_walk_forward` and `_walk_inverse` call
        its corresponding methods.
      name: Name of this bijector.
      parameters: Dictionary of parameters used to initialize this bijector.
        These must be the exact values passed to `__init__`.
      forward_min_event_ndims: A (structure of) integers or `None` values.
        If all values are integers, then these specify the mininimum rank of
        each event part; their structure must match that of inputs to `forward`.
        If the minimum ranks are not known, inference may be triggered by
        passing a structure of `None` values instead. This structure must be
        such that calling `self._walk_forward(step_fn, forward_min_event_ndims)`
        passes exactly one `None` value to each toplevel invocation of
        `step_fn`---that is, it should correspond to the structure of
        *bijectors* that act directly on a user-passed input (excluding
        bijectors that act on downstream quantities), which may in general be
        shallower than the structure of inputs to `forward`. For example, the
        first step of a Chain applies a single bijector, so Chain would pass a
        single `None` (even though the initial bijector might itself accept a
        multipart input). On the other hand, the bijectors in a JointMap are
        all applied directly to user-passed input, so the
        appropriate structure would be that of `self.bijectors` (even if some
        of those bijectors might themselves accept multipart inputs).
      inverse_min_event_ndims: A (structure of) integers and/or `None` values,
        with semantics analogous to `forward_min_event_ndims`.
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

      # Copy the nested structure so we don't mutate arguments during tracking.
      self._bijectors = nest.map_structure(lambda b: b, bijectors)
      self._validate_event_size = validate_event_size
      self.__is_injective = is_injective
      self.__is_permutation = is_permutation

      if any(nd is None for nd in tf.nest.flatten(forward_min_event_ndims)):
        # Infer forward_min_event_ndims by walking backwards through the graph.
        forward_min_event_ndims = nest.map_structure_up_to(
            forward_min_event_ndims,
            lambda inferred: inferred.forward_min_event_ndims,
            self._walk_inverse(
                _update_forward_min_event_ndims,
                tf.nest.map_structure(lambda x: None, forward_min_event_ndims)))
      if any(nd is None for nd in tf.nest.flatten(inverse_min_event_ndims)):
        # Infer forward_min_event_ndims by walking forwards through the graph.
        inverse_min_event_ndims = nest.map_structure_up_to(
            inverse_min_event_ndims,
            lambda inferred: inferred.forward_min_event_ndims,
            self._walk_forward(
                _update_inverse_min_event_ndims,
                tf.nest.map_structure(lambda x: None, inverse_min_event_ndims)))

      super(Composition, self).__init__(
          forward_min_event_ndims=forward_min_event_ndims,
          inverse_min_event_ndims=inverse_min_event_ndims,
          is_constant_jacobian=is_constant_jacobian,
          parameters=parameters,
          name=name,
          **kwargs)

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict()

  @property
  def bijectors(self):
    if isinstance(self._bijectors, Composition):
      return self._bijectors.bijectors  # pylint: disable=protected-access
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

  @property
  def _parts_interact(self):
    # Parts of this Composition's inputs and outputs can interact if and only if
    # they interact at some component bijector. Note that this may need to be
    # overridden by subclasses such as `_DefaultJointBijector` that build
    # component bijectors on the fly during each invocation, since that process
    # can induce additional interactions.
    return any(
        b._parts_interact for b in tf.nest.flatten(self.bijectors))  # pylint: disable=protected-access

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
    Returns:
      bijectors_forward: The value returned by `self._bijectors._walk_forward`
        if `self._bijectors` is a `Composition` bijector.
    Raises:
      NotImplementedError, if `self._bijectors` is a nested structure.
    """
    if isinstance(self._bijectors, Composition):
      return self._bijectors._walk_forward(step_fn, argument, **kwargs)  # pylint: disable=protected-access
    raise NotImplementedError('{}._walk_forward is not implemented'.format(
        type(self).__name__))

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
    Returns:
      bijectors_inverse: The value returned by `self._bijectors._walk_inverse`
        if `self._bijectors` is a `Composition` bijector.
    Raises:
      NotImplementedError, if `self._bijectors` is a nested structure.
    """
    if isinstance(self._bijectors, Composition):
      return self._bijectors._walk_inverse(step_fn, argument, **kwargs)  # pylint: disable=protected-access
    raise NotImplementedError('{}._walk_inverse is not implemented'.format(
        type(self).__name__))

  ###
  ### Nontrivial Methods
  ###

  ## LDJ Methods

  def _call_forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
    """Compute forward_log_det_jacobian over the composition."""
    with self._name_and_control_scope(name):
      dtype = self.inverse_dtype(**kwargs)
      x = nest_util.convert_to_nested_tensor(
          x, name='x', dtype_hint=dtype,
          dtype=None if bijector.SKIP_DTYPE_CHECKS else dtype,
          allow_packing=True)
      if event_ndims is None:
        event_ndims = self.forward_min_event_ndims
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
        event_ndims = self.inverse_min_event_ndims
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

  def _batch_shape(self, x_event_ndims):
    """Broadcasts the batch shapes of component bijectors."""
    batch_shapes_at_components = []

    def _accumulate_batch_shapes_forward(bij, event_ndims):
      batch_shapes_at_components.append(
          bij.experimental_batch_shape(x_event_ndims=event_ndims))
      return bij.forward_event_ndims(event_ndims)

    # Populate 'batch_shapes_at_components' by walking forwards.
    self._walk_forward(_accumulate_batch_shapes_forward, x_event_ndims)
    return functools.reduce(tf.broadcast_static_shape,
                            batch_shapes_at_components,
                            tf.TensorShape([]))

  def _batch_shape_tensor(self, x_event_ndims):
    """Broadcasts the batch shapes of component bijectors."""
    batch_shapes_at_components = []

    def _accumulate_batch_shapes_forward(bij, event_ndims):
      batch_shapes_at_components.append(
          bij.experimental_batch_shape_tensor(x_event_ndims=event_ndims))
      return bij.forward_event_ndims(event_ndims)

    # Populate 'batch_shapes_at_components' by walking forwards.
    self._walk_forward(_accumulate_batch_shapes_forward, x_event_ndims)
    return functools.reduce(ps.broadcast_shape,
                            batch_shapes_at_components,
                            [])

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
    if tf.nest.is_nested(event_ndims) and not self._parts_interact:
      return self._call_walk_forward(
          lambda b, nd, **kwds: b.forward_event_ndims(nd, **kwds),
          event_ndims, **kwargs)
    return super(Composition, self).forward_event_ndims(event_ndims, **kwargs)

  def inverse_event_ndims(self, event_ndims, **kwargs):
    if tf.nest.is_nested(event_ndims) and not self._parts_interact:
      return self._call_walk_inverse(
          lambda b, nd, **kwds: b.inverse_event_ndims(nd, **kwds),
          event_ndims, **kwargs)
    return super(Composition, self).inverse_event_ndims(event_ndims, **kwargs)


def _update_forward_min_event_ndims(
    bij,
    downstream_quantities,
    get_forward_min_event_ndims=lambda b: b.forward_min_event_ndims,
    get_inverse_min_event_ndims=lambda b: b.inverse_min_event_ndims,
    inverse_event_ndims_fn=lambda b, nd: b.inverse_event_ndims(nd)):
  """Step backwards through the graph to infer `forward_min_event_ndims`.

  Args:
    bij: local tfb.Bijector instance at the current graph node.
    downstream_quantities: Instance of `MinEventNdimsDownstreamQuantities`
      namedtuple, containing event_ndims that satisfy the bijector(s)
      downstream from `bij` in the graph. May be `None` if there are no such
      bijectors.
    get_forward_min_event_ndims: callable; may be overridden to swap
      forward/inverse direction.
    get_inverse_min_event_ndims: callable; may be overridden to swap
      forward/inverse direction.
    inverse_event_ndims_fn: callable; may be overridden to swap
      forward/inverse direction.
  Returns:
    downstream_quantities: Instance of `MinEventNdimsDownstreamQuantities`
      namedtuple containing event_ndims that satisfy `bij` and all downstream
      bijectors.
  """
  if downstream_quantities is None:  # This is a leaf bijector.
    return MinEventNdimsInferenceDownstreamQuantities(
        forward_min_event_ndims=get_forward_min_event_ndims(bij),
        parts_interact=bij._parts_interact)  # pylint: disable=protected-access

  inverse_min_event_ndims = get_inverse_min_event_ndims(bij)
  downstream_min_event_ndims = nest_util.coerce_structure(
      inverse_min_event_ndims,
      downstream_quantities.forward_min_event_ndims)

  # Update the min_event_ndims that is a valid input to downstream bijectors
  # to also be a valid *output* of this bijector, or equivalently, a valid
  # input to `bij.inverse`.
  rank_mismatches = tf.nest.flatten(
      tf.nest.map_structure(
          lambda dim, min_dim: dim - min_dim,
          downstream_min_event_ndims,
          inverse_min_event_ndims))
  if downstream_quantities.parts_interact:
    # If downstream bijectors involve interaction between parts,
    # then a valid input to the downstream bijectors must augment the
    # `downstream_min_event_ndims` by the
    # same rank for every part (otherwise we would induce event shape
    # broadcasting). Hopefully, this will also avoid event-shape broadcasting
    # at the current bijector---if not, the composition is invalid, and the call
    # to `bij.inverse_event_ndims(valid_inverse_min_event_ndims)` below will
    # raise an exception.
    maximum_rank_deficiency = -ps.reduce_min([0] + rank_mismatches)
    valid_inverse_min_event_ndims = tf.nest.map_structure(
        lambda ndims: maximum_rank_deficiency + ndims,
        downstream_min_event_ndims)
  else:
    if bij._parts_interact:  # pylint: disable=protected-access
      # If this bijector does *not* operate independently on its parts, then a
      # valid input to `inverse` cannot require event shape broadcasting. That
      # is, each part must have the same 'excess rank' above the local
      # inverse_min_event_ndims; we ensure this by construction.
      maximum_excess_rank = ps.reduce_max([0] + rank_mismatches)
      valid_inverse_min_event_ndims = tf.nest.map_structure(
          lambda ndims: maximum_excess_rank + ndims,
          inverse_min_event_ndims)
    else:
      # If all parts are independent, can take the pointwise max event_ndims.
      valid_inverse_min_event_ndims = tf.nest.map_structure(
          ps.maximum, downstream_min_event_ndims, inverse_min_event_ndims)

  return MinEventNdimsInferenceDownstreamQuantities(
      # Pull the desired output ndims back through the bijector, to get
      # the ndims of a valid *input*.
      forward_min_event_ndims=inverse_event_ndims_fn(
          bij, valid_inverse_min_event_ndims),
      parts_interact=(
          downstream_quantities.parts_interact or
          bij._parts_interact))  # pylint: disable=protected-access

_update_inverse_min_event_ndims = functools.partial(
    _update_forward_min_event_ndims,
    get_forward_min_event_ndims=lambda b: b.inverse_min_event_ndims,
    get_inverse_min_event_ndims=lambda b: b.forward_min_event_ndims,
    inverse_event_ndims_fn=lambda b, nd: b.forward_event_ndims(nd))
