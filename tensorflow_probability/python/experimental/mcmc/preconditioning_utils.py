# Copyright 2021 The TensorFlow Probability Authors.
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
"""DiagonalMassMatrixAdaptation TransitionKernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.distributions import batch_broadcast
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental.distributions import mvn_precision_factor_linop as mvn_pfl
from tensorflow_probability.python.internal import broadcast_util as bu
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

__all__ = [
    'make_momentum_distribution',
]

JAX_MODE = False
NUMPY_MODE = False

# TODO(b/182603117): Remove this block once distributions are auto-composite.
if JAX_MODE or NUMPY_MODE:
  _CompositeJointDistributionSequential = jds.JointDistributionSequential
  _CompositeLinearOperatorDiag = tf.linalg.LinearOperatorDiag
  _CompositeMultivariateNormalPrecisionFactorLinearOperator = mvn_pfl.MultivariateNormalPrecisionFactorLinearOperator
  _CompositeIndependent = independent.Independent
  _CompositeReshape = reshape.Reshape
  _CompositeTransformedDistribution = transformed_distribution.TransformedDistribution
  _CompositeBatchBroadcast = batch_broadcast.BatchBroadcast

else:
  from tensorflow_probability.python.internal import auto_composite_tensor  # pylint: disable=g-import-not-at-top

  # Add auto-composite tensors to the global namespace to avoid creating new
  # classes inside functions.
  _CompositeJointDistributionSequential = auto_composite_tensor.auto_composite_tensor(
      jds.JointDistributionSequential, omit_kwargs=('name',))
  _CompositeLinearOperatorDiag = auto_composite_tensor.auto_composite_tensor(
      tf.linalg.LinearOperatorDiag, omit_kwargs=('name',))
  _CompositeMultivariateNormalPrecisionFactorLinearOperator = auto_composite_tensor.auto_composite_tensor(
      mvn_pfl.MultivariateNormalPrecisionFactorLinearOperator,
      omit_kwargs=('name',))
  _CompositeIndependent = auto_composite_tensor.auto_composite_tensor(
      independent.Independent, omit_kwargs=('name',))
  _CompositeReshape = auto_composite_tensor.auto_composite_tensor(
      reshape.Reshape, omit_kwargs=('name',))
  _CompositeTransformedDistribution = auto_composite_tensor.auto_composite_tensor(
      transformed_distribution.TransformedDistribution,
      omit_kwargs=('name', 'kwargs_split_fn', 'parameters'))
  _CompositeBatchBroadcast = auto_composite_tensor.auto_composite_tensor(
      batch_broadcast.BatchBroadcast, omit_kwargs=('name',))


def make_momentum_distribution(state_parts, batch_ndims,
                               running_variance_parts=None):
  """Construct a momentum distribution from the running variance.

  This uses a running variance to construct a momentum distribution with the
  correct batch_shape and event_shape.

  Args:
    state_parts: List of `Tensor`.
    batch_ndims: Scalar, for leading batch dimensions.
    running_variance_parts: Optional, list of `Tensor`
       outputs of `tfp.experimental.stats.RunningVariance.variance()`. Defaults
       to ones with the same shape as state_parts.

  Returns:
    `tfd.Distribution` where `.sample` has the same structure as `state_parts`,
    and `.log_prob` of the sample will have the rank of `batch_ndims`
  """
  if running_variance_parts is None:
    running_variance_parts = tf.nest.map_structure(tf.ones_like, state_parts)
  distributions = []
  for variance_part, state_part in zip(running_variance_parts, state_parts):
    running_variance_rank = ps.rank(variance_part)
    state_rank = ps.rank(state_part)
    event_shape = state_part.shape[batch_ndims:]
    if not tensorshape_util.is_fully_defined(event_shape):
      event_shape = ps.shape(state_part)[batch_ndims:]
    batch_shape = state_part.shape[:batch_ndims]
    if not tensorshape_util.is_fully_defined(batch_shape):
      batch_shape = ps.shape(state_part)[:batch_ndims]
    nevt = ps.cast(ps.reduce_prod(event_shape), tf.int32)
    # Pad dimensions and tile by multiplying by tf.ones to add a batch shape
    ones = tf.ones(ps.shape(state_part)[:-(state_rank - running_variance_rank)],
                   dtype=variance_part.dtype)
    ones = bu.left_justified_expand_dims_like(ones, state_part)
    variance_tiled = ones * variance_part
    variance_flattened = tf.reshape(
        variance_tiled,
        ps.concat([ps.shape(variance_tiled)[:batch_ndims],
                   [nevt]], axis=0))

    distribution = _CompositeTransformedDistribution(
        bijector=_CompositeReshape(
            event_shape_out=event_shape, event_shape_in=[nevt]),
        distribution=(_CompositeMultivariateNormalPrecisionFactorLinearOperator(
            precision_factor=_CompositeLinearOperatorDiag(
                tf.math.sqrt(variance_flattened)),
            precision=_CompositeLinearOperatorDiag(variance_flattened))))
    distributions.append(distribution)
  return maybe_make_list_and_batch_broadcast(
      _CompositeJointDistributionSequential(distributions), batch_shape)


def maybe_make_list_and_batch_broadcast(momentum_distribution, batch_shape):
  """Makes the distribution list-like and batched, if possible."""
  if not mcmc_util.is_list_like(momentum_distribution.dtype):
    momentum_distribution = _CompositeJointDistributionSequential(
        [momentum_distribution])
  if (isinstance(momentum_distribution, jds.JointDistributionSequential) and
      not isinstance(momentum_distribution, jdn.JointDistributionNamed) and
      not any(callable(dist_fn) for dist_fn in momentum_distribution.model)):
    momentum_distribution = momentum_distribution.copy(model=[
        _CompositeBatchBroadcast(md, with_shape=batch_shape)
        for md in momentum_distribution.model
    ])
  return momentum_distribution
