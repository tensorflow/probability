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
        'is_preferred'
    ])):
  """Annotates expected properties of a `Tensor`-valued distribution parameter.

  Distributions and Bijectors implementing `._parameter_properties` specify a
  `ParameterProperties` annotation for each of their `Tensor`-valued
  parameters.

  Elements:
    event_ndims: Python `int` rank of the parameter required to describe a
      single event (for distributions) or a single transformation (for
      bijectors, this is the action on a Tensor of rank
      `self.forward_min_event_ndims`). For example, `tfd.Normal` has scalar
      parameters, so would set `event_ndims=0` for both its `loc` and `scale`
      parameters. On the other hand, `tfd.MultivariateNormalTriL` has vector
      loc and matrix scale, so it would set `event_ndims=1` for `loc` and
      `event_ndims=2` for `scale_tril`. Similarly, `tfb.Scale` would set
      `event_dims=0` for its `scale` parameter, while `tfb.ScaleMatvecTriL`
      would set `event_ndims=2` for its `scale_tril` parameter.
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
              is_preferred=True):
    return super(ParameterProperties, cls).__new__(
        cls,
        event_ndims=event_ndims,
        shape_fn=shape_fn,
        default_constraining_bijector_fn=default_constraining_bijector_fn,
        is_preferred=is_preferred)
