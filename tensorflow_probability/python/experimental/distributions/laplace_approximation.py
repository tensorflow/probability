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
"""A Laplace approximation over a JointDistribution object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import mvn_tril
from tensorflow_probability.python.math import value_and_gradient
from tensorflow_probability.python.optimizer import bfgs_utils
from tensorflow_probability.python.optimizer import bfgs


def _common_dtype(dtypes, dtype_hint=None):
  """The `tf.Dtype` common to all elements in `dtypes`."""

  dtype = None
  seen = []

  for dt in dtypes:
    seen.append(dt)
    if dtype is None:
      dtype = dt
    elif dtype != dt:
      raise TypeError(
          "Found incompatible dtypes, {} and {}. Seen so far: {}".format(
              dtype, dt, seen))

  if dtype is None:
    return dtype_hint

  return dtype


def _bfgs_minimize(
    pinned_joint,
    bijector,
    initial_values,
    initial_inverse_hessian_estimate,
    validate_convergence,
):
  """Minimize `-pinned_joint.unnormalized_log_prob` using BFGS."""

  def loss(unconstrained_values):
    constrained_values = bijector.forward(unconstrained_values)
    return -pinned_joint.unnormalized_log_prob(constrained_values)

  @tf.function(autograph=False)
  def loss_and_gradient(unconstrained_values):
    return value_and_gradient(loss, unconstrained_values)

  dtype = _common_dtype(pinned_joint.dtype)
  dim = bijector.inverse_event_shape(pinned_joint.event_shape)

  if initial_values is None:
    initial_values = pinned_joint.sample_unpinned(10, seed=42)

  initial_position = bijector.inverse(initial_values)

  if initial_inverse_hessian_estimate is None:
    initial_inverse_hessian_estimate = tf.linalg.diag(
        tf.constant(1e-3, dtype=dtype, shape=dim))

  bfgs_results = bfgs.minimize(
      loss_and_gradient,
      initial_position=initial_position,
      initial_inverse_hessian_estimate=initial_inverse_hessian_estimate,
      stopping_condition=bfgs_utils.converged_any,
  )

  if not validate_convergence:
    return bfgs_results

  converged, failed = bfgs_results.converged, bfgs_results.failed

  assert_any_converged = tf.debugging.Assert(
        condition=bfgs_utils.converged_any(converged, failed),
        data=bfgs_results,
    )

  with tf.control_dependencies([assert_any_converged]):
    return bfgs_results


def laplace_approximation(
    joint_dist,
    bijectors=None,
    data=None,
    initial_values=None,
    initial_inverse_hessian_estimate=None,
    validate_convergence=True,
):
  """A Laplace approximation over a joint distribution.

  Args:
    joint_dist:

    bijectors:

    data:

    initial_values:

    initial_inverse_hessian_estimate:

    validate_convergence:

  Returns:


  #### Examples

  #### References

"""

  pinned_joint = joint_dist.experimental_pin(data)

  if bijectors is None:
    pinned_bijector = pinned_joint.experimental_default_event_space_bijector()
    bijectors = pinned_bijector.bijectors

  unconstrained_shapes = [
      x.inverse_event_shape(y)
      for x, y in zip(bijectors, pinned_joint.event_shape)
  ]

  size_splits = [x.num_elements() for x in unconstrained_shapes]

  # this is required as as split can produce a tensor of shape [1] when the
  # distribution event shape is []
  reshapers = [
      reshape.Reshape(event_shape_out=x, event_shape_in=[x.num_elements()])
      for x in unconstrained_shapes
  ]

  bijector = chain.Chain([
      joint_map.JointMap(bijectors=bijectors),
      joint_map.JointMap(bijectors=reshapers),
      split.Split(num_or_size_splits=size_splits),
  ])

  bfgs_results = _bfgs_minimize(
      pinned_joint=pinned_joint,
      bijector=bijector,
      initial_values=initial_values,
      initial_inverse_hessian_estimate=initial_inverse_hessian_estimate,
      validate_convergence=validate_convergence,
  )

  # there is also the option of using multiple solutions and returning a batch
  # of distributions. For example we could use all the solutions which have
  # "converged" and the line search has not "failed"
  best_soln = tf.argmin(tf.reshape(bfgs_results.objective_value, [-1]))

  dim = sum(size_splits)

  unconstrained_mean = tf.reshape(bfgs_results.position, [-1, dim])[best_soln]

  # this is an approximation of the inverse hessian, we could be more accurate
  # using e.g. `tf.hessians` at the cost of some compute
  unconstrained_covariance_matrix = tf.reshape(
      bfgs_results.inverse_hessian_estimate, [-1, dim, dim])[best_soln]

  underlying_mvn = mvn_tril.MultivariateNormalTriL(
      loc=unconstrained_mean,
      scale_tril=tf.linalg.cholesky(unconstrained_covariance_matrix))

  return transformed_distribution.TransformedDistribution(
      distribution=underlying_mvn, bijector=bijector)
