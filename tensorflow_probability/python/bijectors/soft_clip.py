# Copyright 2019 The TensorFlow Probability Authors.
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
"""SoftClip bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow.compat.v2 as tf

from tensorflow_probability.python import util as tfp_util

from tensorflow_probability.python.bijectors import bijector
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import scale
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import softplus

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util


__all__ = [
    'SoftClip',
]


class SoftClip(bijector.Bijector):
  """Bijector that approximates clipping as a continuous, differentiable map.

  The `forward` method takes unconstrained scalar `x` to a value `y` in
  `[low, high]`. For values within the interval and far from the bounds
  (`low << x << high`), this mapping is approximately the identity mapping.

  ```python
  b = tfb.SoftClip(low=-10., high=10.)
  b.forward([-15., -7., 1., 9., 20.])
    # => [-9.993284, -6.951412,  0.9998932,  8.686738,  9.999954 ]
  ```

  The softness of the clipping can be adjusted via the `hinge_softness`
  parameter. A sharp constraint (`hinge_softness < 1.0`) will approximate
  the identity mapping very well across almost all of its range, but may
  be numerically ill-conditioned at the boundaries. A soft constraint
  (`hinge_softness > 1.0`) corresponds to a smoother, better-conditioned
  mapping, but creates a larger distortion of its inputs.

  ```python
  b_hard = SoftClip(low=-5, high=5., hinge_softness=0.1)
  b_soft.forward([-15., -7., 1., 9., 20.])
    # => [-10., -7., 1., 8.999995,  10.]

  b_soft = SoftClip(low=-5, high=5., hinge_softness=10.0)
  b_soft.forward([-15., -7., 1., 9., 20.])
    # => [-6.1985435, -3.369276,  0.16719627,  3.6655345,  7.1750355]
  ```

  Note that the outputs are always in the interval `[low, high]`, regardless
  of the `hinge_softness`.

  #### Example use

  A trivial application of this bijector is to constrain the values sampled
  from a distribution:

  ```python
  dist = tfd.TransformedDistribution(
    distribution=tfd.Normal(loc=0., scale=1.),
    bijector=tfb.SoftClip(low=-5., high=5.))
  samples = dist.sample(100)  # => samples guaranteed in [-10., 10.]
  ```

  A more useful application is to constrain the values considered
  during inference, preventing an inference algorithm from proposing values
  that cause numerical issues. For example, this model will return a `log_prob`
  of `NaN` when `z` is outside of the range `[-5., 5.]`:

  ```python
  dist = tfd.JointDistributionNamed({
    'z': tfd.Normal(0., 1.0)
    'x': lambda z: tfd.Normal(
                     loc=tf.log(25 - z**2), # Breaks if z >= 5 or z <= -5.
                     scale=1.)})
  ```

  Using SoftClip allows us to keep an inference algorithm in the feasible
  region without distorting the inference geometry by very much:

  ```python
  target_log_prob_fn = lambda z: dist.log_prob(z=z, x=3.)  # Condition on x==3.

  # Use SoftClip to ensure sampler stays within the numerically valid region.
  mcmc_samples = tfp.mcmc.sample_chain(
    kernel=tfp.mcmc.TransformedTransitionKernel(
      tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=0.1),
      bijector=tfb.SoftClip(-5., 5.)),
    trace_fn=None,
    current_state=0.,
    num_results=100)
  ```

  #### Mathematical Details

  The constraint is built by using `softplus(x) = log(1 + exp(x))` as a smooth
  approximation to `max(x, 0)`. In combination with affine transformations, this
  can implement a constraint to any scalar interval.

  In particular, translating `softplus` gives a generic lower bound constraint:

  ```
  max(x, low) =  max(x - low, 0) + low
              ~= softplus(x - low) + low
              := softlower(x)
  ```

  Note that this quantity is always greater than `low` because `softplus` is
  positive-valued. We can also implement a soft upper bound:

  ```
  min(x, high) =  min(x - high, 0) + high
               = -max(high - x, 0) + high
              ~= -softplus(high - x) + high
              := softupper(x)
  ```

  which, similarly, is always less than `high`.

  Composing these bounds as `softupper(softlower(x))` gives a quantity bounded
  above by `high`, and bounded below by `softupper(low)` (because `softupper`
  is monotonic and its input is bounded below by `low`). In general, we will
  have `softupper(low) < low`, so we need to shrink the interval slightly
  (by `(high - low) / (high - softupper(low))`) to preserve the lower bound.
  The two-sided constraint is therefore:

  ```python
  softclip(x) := (softupper(softlower(x)) - high) *
                   (high - low) / (high - softupper(low)) + high
               = -softplus(high - low - softplus(x - low)) *
                   (high - low) / (softplus(high-low)) + high
  ```

  Due to this rescaling, the bijector can be mildly asymmetric. Values
  of equal distance from the endpoints are mapped to values with slightly
  unequal distance from the endpoints; for example,

  ```python
  b = SoftConstrain(-1., 1.)
  b.forward([-0.5., 0.5.])
    # => [-0.2527727 ,  0.19739306]
  ```

  The degree of the asymmetry is proportional to the size of the rescaling
  correction, i.e., the extent to which `softupper` fails to be the identity
  map at the lower end of the interval. This is maximized when the upper and
  lower bounds are very close together relative to the hinge softness, as in
  the example above. Conversely, when the interval is wide, the required
  correction and asymmetry are very small.

  """

  def __init__(self,
               low=None,
               high=None,
               hinge_softness=None,
               validate_args=False,
               name='soft_clip'):
    """Instantiates the SoftClip bijector.

    Args:
      low: Optional float `Tensor` lower bound. If `None`, the lower-bound
        constraint is omitted.
        Default value: `None`.
      high: Optional float `Tensor` upper bound. If `None`, the upper-bound
        constraint is omitted.
        Default value: `None`.
      hinge_softness: Optional nonzero float `Tensor`. Controls the softness
        of the constraint at the boundaries; values outside of the constraint
        set are mapped into intervals of width approximately
        `log(2) * hinge_softness` on the interior of each boundary. High
        softness reserves more space for values outside of the constraint set,
        leading to greater distortion of inputs *within* the constraint set,
        but improved numerical stability near the boundaries.
        Default value: `None` (`1.0`).
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    """
    with tf.name_scope(name):
      dtype = dtype_util.common_dtype(
          [low, high, hinge_softness], dtype_hint=tf.float32)
      low = tensor_util.convert_nonref_to_tensor(
          low, name='low', dtype=dtype)
      high = tensor_util.convert_nonref_to_tensor(
          high, name='high', dtype=dtype)
      hinge_softness = tensor_util.convert_nonref_to_tensor(
          hinge_softness, name='hinge_softness', dtype=dtype)

      softplus_bijector = softplus.Softplus(hinge_softness=hinge_softness)
      negate = tf.convert_to_tensor(-1., dtype=dtype)

      components = []
      if low is not None and high is not None:
        # Support reference tensors (eg Variables) for `high` and `low` by
        # deferring all computation on them until needed.
        width = tfp_util.DeferredTensor(
            pretransformed_input=high, transform_fn=lambda high: high - low)
        negated_shrinkage_factor = tfp_util.DeferredTensor(
            pretransformed_input=width,
            transform_fn=lambda w: tf.cast(  # pylint: disable=g-long-lambda
                negate * w / softplus_bijector.forward(w), dtype=dtype))

        # Implement the soft constraint from 'Mathematical Details' above:
        #  softclip(x) := -softplus(width - softplus(x - low)) *
        #                        (width) / (softplus(width)) + high
        components = [
            shift.Shift(high),
            scale.Scale(negated_shrinkage_factor),
            softplus_bijector,
            shift.Shift(width),
            scale.Scale(negate),
            softplus_bijector,
            shift.Shift(tfp_util.DeferredTensor(low, lambda x: -x))]
      elif low is not None:
        # Implement a soft lower bound:
        #  softlower(x) := softplus(x - low) + low
        components = [
            shift.Shift(low),
            softplus_bijector,
            shift.Shift(tfp_util.DeferredTensor(low, lambda x: -x))]
      elif high is not None:
        # Implement a soft upper bound:
        #  softupper(x) := -softplus(high - x) + high
        components = [shift.Shift(high),
                      scale.Scale(negate),
                      softplus_bijector,
                      scale.Scale(negate),
                      shift.Shift(high)]

      self._low = low
      self._high = high
      self._hinge_softness = hinge_softness
      self._chain = chain.Chain(components, validate_args=validate_args)

    super(SoftClip, self).__init__(
        forward_min_event_ndims=0,
        dtype=dtype,
        validate_args=validate_args,
        is_constant_jacobian=not components,
        name=name)

  @property
  def low(self):
    return self._low

  @property
  def high(self):
    return self._high

  @property
  def hinge_softness(self):
    return self._hinge_softness

  @classmethod
  def _is_increasing(cls):
    return True

  def _forward(self, x):
    return self._chain.forward(x)

  def _forward_log_det_jacobian(self, x):
    return self._chain._forward_log_det_jacobian(x)  # pylint: disable=protected-access

  def _inverse(self, y):
    with tf.control_dependencies(self._assert_valid_inverse_input(y)):
      return self._chain._inverse(y)  # pylint: disable=protected-access

  def _inverse_log_det_jacobian(self, y):
    with tf.control_dependencies(self._assert_valid_inverse_input(y)):
      return self._chain._inverse_log_det_jacobian(y)  # pylint: disable=protected-access

  def _assert_valid_inverse_input(self, y):
    assertions = []
    if self.validate_args and self.low is not None:
      assertions += [assert_util.assert_greater(
          y, self.low,
          message='Input must be greater than `low`.')]
    if self.validate_args and self.high is not None:
      assertions += [assert_util.assert_less(
          y, self.high,
          message='Input must be less than `high`.')]
    return assertions

  def _parameter_control_dependencies(self, is_init):
    if not self.validate_args or self.low is None or self.high is None:
      return []
    assertions = []
    if is_init != (tensor_util.is_ref(self.low) or
                   tensor_util.is_ref(self.high)):
      assertions.append(assert_util.assert_greater(
          self.high, self.low,
          message='Argument `high` must be greater than `low`.'))
    return assertions
