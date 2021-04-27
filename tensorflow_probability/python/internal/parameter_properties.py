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

from tensorflow_probability.python.bijectors import identity as identity_bijector

__all__ = [
    'BIJECTOR_NOT_IMPLEMENTED',
    'BatchedComponentProperties',
    'ParameterProperties',
]


def BIJECTOR_NOT_IMPLEMENTED():  # pylint: disable=invalid-name
  raise NotImplementedError('No constraining bijector is implemented for this '
                            'parameter.')


def SHAPE_FN_NOT_IMPLEMENTED(sample_shape):  # pylint: disable=invalid-name
  del sample_shape  # Unused.
  raise NotImplementedError('No shape function is implemented for this '
                            'parameter.')


class ParameterProperties(
    collections.namedtuple('ParameterProperties', [
        'event_ndims', 'shape_fn', 'default_constraining_bijector_fn',
        'is_preferred', 'is_tensor'
    ])):
  """Annotates expected properties of a `Tensor`-valued distribution parameter.

  Distributions and Bijectors implementing `._parameter_properties` specify a
  `ParameterProperties` annotation for each of their `Tensor`-valued
  parameters.

  Elements:
    event_ndims: Python `int`, structure of `int`s, or `callable`, specifying
      the minimum effective Tensor rank of this parameter. See description
      below.
      Default value: `0`.
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

  #### Definition of `event_ndims`

  The `event_ndims` of a parameter is the number of rightmost dimensions of
  its Tensor shape required to describe
  a single event (when parameterizing a distribution) or a single transformation
  (when parameterizing a bijector; this is the action of `self.forward(x)`
  when `rank(x) == self.forward_min_event_ndims`). Equivalently, the
  `event_ndims` of a parameter is the minimal Tensor rank of a valid value for
  that parameter. For non-Tensor parameters such as Distributions and
  Bijectors, the concepts of Tensor shape and Tensor rank naturally
  generalize to the parameter's `batch_shape` and batch rank, as discussed
  below.

  Since the `tfd.Normal` distribution can draw a (scalar) event
  given scalar `loc` and `scale` parameters, it would set `event_ndims=0`
  for both of those parameters, indicating that scalar values are valid. On
  the other hand, `tfd.MultivariateNormalTriL` requires a `loc` vector and a
  `scale_tril` matrix, so it would set `event_ndims=1` for `loc` and
  `event_ndims=2` for `scale_tril`.

  Similarly, the `scale` parameter of a `tfb.Scale` bijector may be a scalar
  (this is sufficient to transform a minimal input to the bijector---which
  itself is also a scalar because `tfb.Scale.forward_min_event_ndims==0`), so
  it has `event_dims=0`. By contrast, the `scale_tril` parameter of a
  `tfb.ScaleMatvecTriL` must be a matrix, so it would set `event_ndims=2`.

  The portion of each parameter's shape that remains after slicing off the
  rightmost `event_ndims` is its 'batch shape'. The batch shape(s) of all
  parameters must broadcast with each other, and this broadcast shape is the
  `batch_shape` of the distribution (or bijector, etc) instance. For
  example, in a `tfd.MultivariateNormalTriL(loc, scale_tril)` distribution,
  where `loc.shape == [3, 2]` and `scale_tril.shape == [4, 1, 2, 2]`, the
  parameter batch shapes are `[3]` and `[4, 1]` respectively, and these
  broadcast to an overall batch shape of `[4, 3]`.

  **Non-Tensor parameters (nested Distributions, Bijectors, etc):** When a
  distribution takes another distribution as a parameter, an event of the
  outer distribution may sample multiple events from the inner
  distribution (the reverse, in which a single inner event generates multiple
  outer events, is not possible; events are taken to be indivisible). For
  example, in `tfd.Independent(inner_dist, reinterpreted_batch_ndims=nd)`,
  a single event of the outer Independent distribution consumes a batch of
  events of shape `inner_dist.batch_shape[-nd:]` from the inner
  `distribution`. In general, the
  `event_ndims` of a non-Tensor parameter is the number of rightmost
  dimensions of its *batch* shape required to describe an event of the outer
  distribution: here, we would take `event_ndims=nd` for the `distribution`
  parameter of `tfd.Independent`. This is analogous to the definition for Tensor
  parameters, simply replacing Tensor shape with `batch_shape`.

  It is important to distinguish between the properties of the inner
  distribution (which will itself have an `event_shape` whose rank we
  might colloquially refer to as its 'event ndims'), versus the parameter
  `event_ndims` we are discussing here, which is a property solely of the outer
  distribution. The latter describes an index into the
  *batch* (rather than event) shape of the inner distribution instance.

  **Instance-dependent `event_ndims`**. A parameter's `event_ndims` may be a
  callable that returns an integer (or a structure of integers; see below),
  rather than just a static integer. This callable must take as its sole
  argument an instance of the class being parameterized (`self`). For example,
  for the `distribution` parameter of the `Independent` distribution given
  above we would specify
  `event_ndims=lambda self: self.reinterpreted_batch_ndims`, indicating that the
  outer class's relationship to the inner distribution depends on another
  instance parameter (`reinterpreted_batch_ndims`). The returned value may be
  a Python `int` or an integer `Tensor`, but the callable itself may not cause
  graph side effects (e.g., creating new Tensors).

  **Structured parameters.** A parameter's `event_ndims` may be a nested
  structure (list, dict, etc.) of integers, if and only if the parameter value
  *itself* is a nested structure of Tensors or non-Tensor objects. For
  example, in the joint bijector
  `tfb.JointMap(bijectors=[tfb.Softplus(), tfb.Exp()])`,
  the `event_ndims` of the `bijectors` parameter would be `[0, 0]`, matching
  the structure of the `bijectors` value (note that since this structure is
  instance-dependent, the `event_ndims` would need to be specified using a
  calalble, as detailed above).

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

  # Specify default properties.
  def __new__(cls,
              event_ndims=0,
              shape_fn=lambda sample_shape: sample_shape,
              default_constraining_bijector_fn=identity_bijector.Identity,
              is_preferred=True,
              is_tensor=True):
    return super(ParameterProperties, cls).__new__(
        cls,
        event_ndims=event_ndims,
        shape_fn=shape_fn,
        default_constraining_bijector_fn=default_constraining_bijector_fn,
        is_preferred=is_preferred,
        is_tensor=is_tensor)

  def instance_event_ndims(self, instance):
    if callable(self.event_ndims):
      if instance is None:
        raise ValueError('Attempting to get the per-event rank of a parameter '
                         'for which this is instance-dependent, but no '
                         'instance was provided.')
      return self.event_ndims(instance)  # Must not have graph side effects.
    return self.event_ndims


class BatchedComponentProperties(ParameterProperties):
  """Alias to assist in defining properties of non-Tensor parameters."""

  def __new__(cls,
              event_ndims=0,
              shape_fn=None,
              default_constraining_bijector_fn=None,
              is_preferred=True):
    return super(BatchedComponentProperties, cls).__new__(  # pylint: disable=redundant-keyword-arg
        cls=cls,
        event_ndims=event_ndims,
        shape_fn=shape_fn,
        default_constraining_bijector_fn=default_constraining_bijector_fn,
        is_preferred=is_preferred,
        is_tensor=False)

