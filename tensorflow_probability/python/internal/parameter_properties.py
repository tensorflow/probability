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
"""Properties of parameters to distributions and bijectors."""

import collections

import tensorflow.compat.v2 as tf

__all__ = [
    'BatchedComponentProperties',
    'BIJECTOR_NOT_IMPLEMENTED',
    'ParameterProperties',
    'SHAPE_FN_NOT_IMPLEMENTED',
]


NO_EVENT_NDIMS = 'INTERNAL_NO_EVENT_NDIMS'


def BIJECTOR_NOT_IMPLEMENTED():  # pylint: disable=invalid-name
  raise NotImplementedError('No constraining bijector is implemented for this '
                            'parameter.')


def SHAPE_FN_NOT_IMPLEMENTED(sample_shape):  # pylint: disable=invalid-name
  del sample_shape  # Unused.
  raise NotImplementedError('No shape function is implemented for this '
                            'parameter.')


def _default_constraining_bijector_fn():
  from tensorflow_probability.python.bijectors import identity as identity_bijector  # pylint:disable=g-import-not-at-top
  return identity_bijector.Identity()


class ParameterProperties(
    collections.namedtuple('ParameterProperties', [
        'event_ndims', 'event_ndims_tensor',
        'shape_fn', 'default_constraining_bijector_fn',
        'is_preferred', 'is_tensor', 'specifies_shape'
    ])):
  """Annotates expected properties of a `Tensor`-valued distribution parameter.

  Distributions and Bijectors implementing `._parameter_properties` specify a
  `ParameterProperties` annotation for each of their `Tensor`-valued
  parameters.

  Elements:
    event_ndims: Python `int`, structure of `int`s, or `callable`, specifying
      the minimum effective Tensor rank of this parameter. See description
      below. May also be `None` or NO_EVENT_NDIMS, indicating that this
      parameter does not follow a batch-shape-vs-event-shape distinction (for
      example, if the parameter cannot have batch dimensions, or if its value
      itself specifies a shape). When `event_ndims` is callable, it should
      return `None` to indicate that the instance-dependent event_ndims value is
      unknown. Alternatively, it should return NO_EVENT_NDIMS to indicate that
      this parameter does not participate in batch semantics in an
      instance-dependent manner.
      Default value: `0`.
    event_ndims_tensor: optional value like `event_ndims`, except that callables
      are allowed to perform Tensor operations. Defaults to `event_ndims` if not
      provided.
      Default value: `None`.
    shape_fn: Python `callable` with signature
      `parameter_shape = shape_fn(shape)`. Given the desired shape of
      an 'output' from this instance, returns the expected shape of the
      parameter. For `Distribution`s, an output is a value returned by
      `self.sample()` (whose shape is the concatenation of `self.batch_shape`
      with `self.event_shape`). For `Bijector`s, an output is a value
      `y = self.forward(x)` where `x` is an input `Tensor` of rank
      `self.forward_min_event_ndims` (and the shape of `y` is the concatenation
      of `self.batch_shape` with an 'event' shape of rank
      `self.inverse_min_event_ndims`). May raise an exception if the shape
      cannot be inferred.
      Default value: `lambda shape: shape`.
    default_constraining_bijector_fn: Optional Python `callable` with signature
      `bijector = default_constraining_bijector_fn()`. The return value is a
      `tfb.Bijector` instance that maps from an unconstrained real-valued vector
      to the support of the parameter.
      Default value: `tfb.Identity`.
    is_preferred: Python bool value specifying whether this
      parameter should be passed when this distribution or bijector is
      automatically instantiated. Only one of a set of mutually-exclusive
      parameters, such as `logits` and `probs`, may set `is_preferred=True`;
      as a guideline, this is generally the parameterization that allows for
      more stable computation.
      Default value: `True`.
    is_tensor: Python bool specifying whether this parameter is (or may be) a
      Tensor. This should be `False` for non-numeric parameters such as
      other distributions or bijectors.
      Default value: `True`.
    specifies_shape: Python `bool` indicating whether this parameter is a shape,
      index, axis, or other quantity such that Tensor shapes
      may depend on the *value* (rather than just the shape) of this parameter.
      Default value: `False`.

  #### Batch shapes and parameter `event_ndims`

  The `batch_shape` of a distribution/bijector/linear operator/PSD kernel/etc.
  instance is the shape of distinct parameterizations represented by that
  instance. It is computed by broadcasting the batch shapes of that instance's
  parameters, where an individual parameter's 'batch shape' is the shape of
  values specified for that parameter.

  To compute the batch shape of a given parameter, we need to know what counts
  as a 'single value' of that parameter. For example, the `scale` parameter of
  `tfd.Normal` is semantically scalar-valued, so a value of shape `[d]`
  would have batch shape `[d]`. On the other hand, the `scale_diag` parameter
  of `tfd.MultivariateNormalDiag` is semantically vector-valued, so in this
  context a value of shaped `[d]` would have batch shape `[]`. TFP formalizes
  this by annotating the `scale` parameter with `event_ndims=0`, and the
  `scale_diag` parameter with `event_ndims=1`.

  In general, the `event_ndims` of a `Tensor`-valued parameter is the number of
  rightmost dimensions of its shape used to describe a single event of the
  parameterized instance. Equivalently, it is the minimal Tensor rank of a valid
  value for that parameter. The portion of each Tensor parameter's shape that
  remains after slicing off the rightmost `event_ndims` is its 'parameter
  batch shape'. The batch shape(s) of all parameters must broadcast with each
  other. For example, in a `tfd.MultivariateNormalDiag(loc, scale_diag)`
  distribution, where `loc.shape == [3, 2]` and `scale_diag.shape == [4, 1, 2]`,
  the parameter batch shapes are `[3]` and `[4, 1]` respectively, and these
  broadcast to an overall batch shape of `[4, 3]`.

  #### Instance-dependent (callable) `event_ndims`

  A parameter's `event_ndims` may be specified as a *callable* that returns an
  integer and takes as its argument an instance `self` of the class being
  parameterized. This allows parameters whose interpretation depends on other
  parameters. Callables for `Bijector` parameters must also accept
  a second argument `x_event_ndims`, described below.

  For example, for the `distribution` parameter of `tfd.Independent`, we
  would specify `event_ndims=lambda self: self.reinterpreted_batch_ndims`,
  indicating that the outer class's relationship to the inner distribution
  depends on another instance parameter (`reinterpreted_batch_ndims`). The
  value returned from an `event_ndims` callable may be a Python `int` or an
  integer `Tensor`, but the callable itself may not cause graph side effects
  (e.g., create new Tensors). In cases where graph ops can't be avoided,
  the `event_ndims` callable should return `None`, and a separate callable
  `event_ndims_tensor` must be provided.

  #### Parameters of non-`Distribution` objects

  The notion of an 'event' generalizes beyond distributions. In general, an
  `event` refers to an instance of an object with `batch_shape==[]`, and the
  `event_ndims` of a parameter describes the parameter value that would define
  such an instance. For example:

  * `tf.linalg.LinearOperator`s: an 'event' of a linear operator is a single
    linear transformation. For example, the `diag` parameter
    to `tf.linalg.LinearOperatorDiag` has `event_ndims=1`, because a
    diagonal matrix is defined by the vector of values along the diagonal.

  * `tfp.math.psd_kernels.PositiveSemidefiniteKernel`s: an event of a PSD kernel
    defines a single kernel function. For example, the `amplitude`
    parameter to `tf.math.psd_kernels.ExponentiatedQuadratic` has
    `event_ndims=0`, since a scalar amplitude is sufficient to specify the
    kernel (more precisely, because dimensions *above* a scalar will induce
    batch shape, describing a batch of kernels).

  * `tfb.Bijector`s: the notion of an 'event' for bijectors varies according to
    the `event_ndims` of the value being transformed. TFP supports two
    approaches to annotating the `event_ndims` of Bijector parameters:

    - Using `min_event_ndims` (static): the parameter `event_ndims` is
      a static integer corresponding to the parameter's rank
      when transforming an event of rank `forward_min_event_ndims`.
      For example, `tfb.ScaleMatvecTriL` has `forward_min_event_ndims==1`,
      indicating that it can transform vector events, so we would annotate its
      `scale_tril` parameter with `event_ndims=2` to indicate that such a
      transformation is parameterized by a matrix-valued `scale_tril`. This
      implies that transformations of matrix events would in general
      be parameterized by a rank-3 `scale_tril` parameter (with lower-rank
      parameter values implicitly broadcasting to rank 3), and so on.

    - Callable `event_ndims`: Alternately, a parameter's `event_ndims` may be
      specified as callable `event_ndims(bijector_instance, x_event_ndims)` that
      returns the rank of the parameter used to transform an event of rank
      `x_event_ndims`. This more general annotation strategy is
      required for multipart bijectors that define `_parts_interact=False`,
      since their parameters may interact with only some parts of the event.
      For example, the bijector `tfb.JointMap([tfb.Scale(scale=tf.ones([2])),
      tfb.Scale(scale=tf.ones([3]))])` is parameterized by two `Scale` bijectors
      (themselves each parameterized by a `scale` `Tensor`), each
      of which applies separately to the corresponding event part. When
      transforming events with `event_ndims=[0, 1]`, `[1, 0]`, or `[1, 1]`,
      the `bijectors` parameter to the `JointMap` may therefore have
      `event_ndims` of `[0, 1]`, `[1, 0]`, or `[1, 1]`,  respectively (implying
      contextual batch shape of `[2]`, `[3]`, or `[]` respectively). We could
      annotate this as a callable parameter `event_ndims` given by
      `lambda self, x_event_ndims: x_event_ndims` (the actual generic `JointMap`
      annotation is more complex, but will ground out to this in this case).
      Note that this bijector *cannot* transform an event with
      `event_ndims=[0, 0]`, since this would imply a contextual batch shape of
      `broadcast_shape([2], [3])`, which is not defined.

  ### Non-Tensor-valued parameters (Distributions, Bijectors, etc).

  The previous section discussed annotating parameters of
  non-Distribution objects. We'll now consider the orthogonal generalization:
  parameters that *themselves* take non-Tensor values. For example, the
  `distribution` and `bijector` parameters of `tfd.TransformedDistribution` are
  themselves a distribution and a bijector, respectively.

  * `Distribution`, `LinearOperator`, `PositiveSemidefiniteKernel`, and other
  batchable parameters: the `event_ndims` annotation for a parameter that
  has a (context-independent) batch shape is the number of rightmost dimensions
  of that batch shape required to describe an event of the
  parameterized object. For example, in `tfd.Independent(inner_dist,
  reinterpreted_batch_ndims=nd)`, a single event of the outer Independent
  distribution consumes a batch of events of shape
  `inner_dist.batch_shape[-nd:]` from the inner `distribution`. Here, we
  would take `event_ndims=nd` for the `distribution` parameter of
  `tfd.Independent`. This is analogous to the definition for
  Tensor parameters, simply replacing Tensor shape with `batch_shape`.

  * `Bijector`-valued parameters: the `event_ndims` annotation for a
  bijector-valued parameter is the rank of the `x` values *with which the
  bijector will be invoked* during an event of the outer object . For example,
  an event of `TransformedDistribution(distribution, bijector)` invokes the
  bijector with events of rank `rank_from_shape(distribution.event_shape)`.

  #### Structured parameter `event_ndims`

  A parameter's `event_ndims` will be a nested structure of integers
  (list, dict, etc.) if either of the following applies:

  1. The parameter value itself is a nested structure. For example, in the
     joint bijector `tfb.JointMap(bijectors=[tfb.Softplus(), tfb.Exp()])`,
     the `event_ndims` of the `bijectors` parameter would be `[0, 0]`, matching
     the structure of the `bijectors` value (note that since this structure is
     instance-dependent, the `event_ndims` would need to be specified using a
     callable, as detailed above).

  2. The parameter is a Bijector with structured `forward_min_event_ndims`.
     For example, in
     `tfb.JointMap(bijectors=[tfb.Softplus(), tfb.Invert(tfb.Split(2))])`,
     the `event_ndims` of the `bijectors` parameter would be `[0, [1, 1]]`,
     since the inverse of the Split bijector has
     `forward_min_event_ndims=[1, 1]`.

  Any ambiguity between these two uses for structured `event_ndims` can
  be resolved by examining the parameter value. For example,
  `event_ndims = [[2, 1], [1, 0]]` could describe a nested structure containing
  four Tensors (or distributions, single-part bijectors, etc.), a list
  containing two structured bijectors, or a single bijector operating on nested
  lists of Tensors, but we can always tell which of these is the case by
  examining the actually instantiated parameter.

  Note that `JointDistribution`-valued parameters never have structured
  `event_ndims`, despite having structured event shapes, because the
  `event_ndims` annotation of a Distribution parameter describes the
  number of that distribution's *batch* dimensions that contribute to an event
  of the outer parameterized object. Bijectors require additional annotation
  not because they operate on structured events, but rather because they operate
  in a context-specific manner depending on the event being transformed.

  #### Choice of constraining bijectors

  The practical support of a parameter---defined as the regime in
  which the distribution may be expected to produce numerically
  valid samples and (log-)densities---may differ slightly from the
  mathematical support. For example, Normal `scale` is mathematically supported
  on positive real numbers, but in practice, dividing by very small scales may
  cause overflow. We might therefore prefer a bijector such as
  `tfb.Softplus(low=eps)` that excludes very small values.

  **In general, default constraining bijectors should attempt to
  implement a *practical* rather than mathematical support, and users of
  default bijectors should be aware that extreme elements of the mathematical
  support may not be attainable.** The notion of 'practical support' is
  inherently fuzzy, and defining it may require arbitrary choices. However,
  this is preferred to the alternative of allowing the default behavior to be
  numerically unstable in common settings. As a general guide, any
  restrictions on the mathematical support should be 'conceptually
  infinitesimal': it may be appropriate to constrain a Beta concentration
  parameter to be greater than `eps`, but not to be greater than `1 + eps`,
  since the latter is a non-infinitesimal restriction of the mathematical
  support.
  """

  __slots__ = ()

  NO_EVENT_NDIMS = NO_EVENT_NDIMS

  # Specify default properties.
  def __new__(cls,
              event_ndims=0,
              event_ndims_tensor=None,
              shape_fn=lambda sample_shape: sample_shape,
              default_constraining_bijector_fn=(
                  _default_constraining_bijector_fn),
              is_preferred=True,
              is_tensor=True,
              specifies_shape=False):

    if event_ndims_tensor is None:
      event_ndims_tensor = event_ndims

    return super(ParameterProperties, cls).__new__(
        cls,
        event_ndims=event_ndims,
        event_ndims_tensor=event_ndims_tensor,
        shape_fn=shape_fn,
        default_constraining_bijector_fn=default_constraining_bijector_fn,
        is_preferred=is_preferred,
        is_tensor=is_tensor,
        specifies_shape=specifies_shape)

  def instance_event_ndims(self, instance, require_static=False):
    event_ndims = (self.event_ndims if require_static
                   else self.event_ndims_tensor)
    if callable(event_ndims):
      if instance is None:
        raise ValueError('Attempting to get the per-event rank of a parameter '
                         'for which this is instance-dependent, but no '
                         'instance was provided.')
      return event_ndims(instance)
    return event_ndims

  def bijector_instance_event_ndims(self,
                                    bijector,
                                    x_event_ndims,
                                    require_static=False):
    """Computes parameter event_ndims when parameterizing a bijector."""
    event_ndims = (self.event_ndims if require_static
                   else self.event_ndims_tensor)
    # Multipart bijectors with `_parts_interact=False` should
    # annotate parameter `event_ndims` callables that take the structured
    # `x_event_ndims` of the event being transformed and return the
    # (potentially structured) event_ndims of the corresponding parameter.
    if callable(event_ndims):
      if bijector is None:
        raise ValueError('Attempting to get the per-event rank of a parameter '
                         'for which this is instance-dependent, but no '
                         'instance was provided.')
      return event_ndims(bijector, x_event_ndims)
    if event_ndims is None:
      return None
    # Multipart bijectors with `_parts_interact=True` (e.g.,
    # `_DefaultJointBijector`, or compositions that include Split/Concat
    # bijectors) reduce a single joint LDJ, as opposed to individual LDJs for
    # each part, so they require that the number of reduction dimensions is the
    # same across all parts. That is, there should be a unique value for the
    # difference between the actual `x_event_ndims` and the minimum event ndims.
    additional_event_ndims = _unique_difference(
        x_event_ndims, bijector.forward_min_event_ndims)
    return tf.nest.map_structure(lambda nd: nd + additional_event_ndims,
                                 event_ndims)


class BatchedComponentProperties(ParameterProperties):
  """Alias to assist in defining properties of non-Tensor parameters."""

  def __new__(cls,
              event_ndims=0,
              event_ndims_tensor=None,
              default_constraining_bijector_fn=None,
              is_preferred=True):
    return super(BatchedComponentProperties, cls).__new__(  # pylint: disable=redundant-keyword-arg
        cls=cls,
        event_ndims=event_ndims,
        event_ndims_tensor=event_ndims_tensor,
        # TODO(davmre): do we need/want shape annotations for non-Tensor params?
        shape_fn=SHAPE_FN_NOT_IMPLEMENTED,
        default_constraining_bijector_fn=default_constraining_bijector_fn,
        is_preferred=is_preferred,
        is_tensor=False,
        specifies_shape=False)


class ShapeParameterProperties(ParameterProperties):
  """Convenience alias for annotating shape parameters."""

  def __new__(cls, is_preferred=True):
    return super(ShapeParameterProperties, cls).__new__(  # pylint: disable=redundant-keyword-arg
        cls=cls,
        event_ndims=NO_EVENT_NDIMS,
        shape_fn=SHAPE_FN_NOT_IMPLEMENTED,
        default_constraining_bijector_fn=BIJECTOR_NOT_IMPLEMENTED,
        is_preferred=is_preferred,
        is_tensor=True,
        specifies_shape=True)


def _unique_difference(structure1, structure2):
  differences = [a - b
                 for a, b in
                 zip(tf.nest.flatten(structure1), tf.nest.flatten(structure2))]
  if all([d == differences[0] for d in differences]):
    return differences[0]
  raise ValueError('Could not find unique difference between {} and {}'
                   .format(structure1, structure2))
