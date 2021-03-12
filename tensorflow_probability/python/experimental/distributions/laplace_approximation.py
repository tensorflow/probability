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
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.math import value_and_gradient
from tensorflow_probability.python.optimizer import bfgs_utils
from tensorflow_probability.python.optimizer import bfgs


def _call(obj, fns, *args, **kwargs):
  """attempt to call the first of `obj.fn` for `fn` in `fns`"""
  for fn in fns:
    call = getattr(obj, fn, None)
    if call is not None:
      return call(*args, **kwargs)
  raise ValueError("Couldn't find any of {} in {}".format(fns, obj))


def _sample(joint_dist, *args, **kwargs):
  """Take samples regardless of joint distribution type."""
  fns = ["sample", "sample_unpinned"]
  return _call(joint_dist, fns, *args, **kwargs)


def _log_prob(joint_dist, *args, **kwargs):
  """Log prob regardless of joint distribution type."""
  fns = ["log_prob", "unnormalized_log_prob"]
  return _call(joint_dist, fns, *args, **kwargs)


def _bfgs_dimension(joint_dist, bijector):
  """The dimension which BFGS works on"""
  flat_event_shape = tf.nest.flatten(joint_dist.event_shape)
  return bijector.inverse_event_shape(flat_event_shape)[0]


def _bfgs_minimize(
    joint_dist,
    bijector,
    initial_values,
    initial_inverse_hessian_estimate,
    validate_convergence,
    add_log_det_jacobian,
):
  """Minimize the negative log prob of a joint distribution using BFGS."""

  names = joint_dist._flat_resolve_names()

  def loss(unconstrained_values):
    constrained_values = bijector.forward(unconstrained_values)
    lp = _log_prob(joint_dist, **dict(zip(names, constrained_values)))
    if not add_log_det_jacobian:
      return -lp
    lp_rank = ps.rank(lp)
    event_ndims = ps.rank(unconstrained_values) - lp_rank
    ldj = bijector.forward_log_det_jacobian(unconstrained_values, event_ndims)
    return -(lp + ldj)

  @tf.function(autograph=False)
  def loss_and_gradient(unconstrained_values):
    return value_and_gradient(loss, unconstrained_values)

  if initial_values is None:
    initial_values = _sample(joint_dist, sample_shape=10, seed=(42, 666))

  initial_position = bijector.inverse(tf.nest.flatten(initial_values))

  if initial_inverse_hessian_estimate is None:
    dtype = dtype_util.common_dtype(
        joint_dist._get_single_sample_distributions())
    dim = _bfgs_dimension(joint_dist, bijector)
    initial_inverse_hessian_estimate = tf.linalg.diag(
        tf.constant(1e-3, dtype=dtype, shape=[dim]))

  bfgs_results = bfgs.minimize(
      loss_and_gradient,
      initial_position=initial_position,
      initial_inverse_hessian_estimate=initial_inverse_hessian_estimate,
      stopping_condition=bfgs_utils.converged_any)

  if not validate_convergence:
    return bfgs_results

  converged, failed = bfgs_results.converged, bfgs_results.failed

  assert_any_converged = tf.debugging.Assert(
        condition=bfgs_utils.converged_any(converged, failed),
        data=bfgs_results)

  with tf.control_dependencies([assert_any_converged]):
    return bfgs_results


def _transform_reshape_split_bijector(joint_dist, bijectors):
  """A bijector from the BFGS compatible `Tensor` to joint dist event shape"""

  unconstrained_shapes = [
      x.inverse_event_shape(y)
      for x, y in zip(bijectors, tf.nest.flatten(joint_dist.event_shape))
  ]

  # this reshaping is required as as split can produce a tensor of shape [1]
  # when the distribution event shape is []
  reshapers = [
      reshape.Reshape(event_shape_out=x, event_shape_in=[x.num_elements()])
      for x in unconstrained_shapes]

  size_splits = [x.num_elements() for x in unconstrained_shapes]

  return chain.Chain([joint_map.JointMap(bijectors=bijectors),
                      joint_map.JointMap(bijectors=reshapers),
                      split.Split(num_or_size_splits=size_splits)])


def laplace_approximation(
    joint_dist,
    data=None,
    bijectors=None,
    initial_values=None,
    initial_inverse_hessian_estimate=None,
    validate_convergence=True,
    add_log_det_jacobian=False,
):
  """A Laplace approximation over a joint distribution.

  Args:
    joint_dist:

    data:

    bijectors:

    initial_values:

    initial_inverse_hessian_estimate:

    validate_convergence:

    add_log_det_jacobian:

  Returns:


  #### Examples

  #### References

  """

  if data is not None:
    joint_dist = joint_dist.experimental_pin(data)

  if bijectors is None:
    joint_bijector = joint_dist.experimental_default_event_space_bijector()
    bijectors = joint_bijector.bijectors

  bijector = _transform_reshape_split_bijector(joint_dist, bijectors)

  bfgs_results = _bfgs_minimize(
      joint_dist=joint_dist,
      bijector=bijector,
      initial_values=initial_values,
      initial_inverse_hessian_estimate=initial_inverse_hessian_estimate,
      validate_convergence=validate_convergence,
      add_log_det_jacobian=add_log_det_jacobian)

  # there is also the option of using multiple solutions and returning a batch
  # of distributions. For example we could use all the solutions which have
  # "converged" and the line search has not "failed"
  best_soln = tf.argmin(tf.reshape(bfgs_results.objective_value, [-1]))

  dim = _bfgs_dimension(joint_dist, bijector)

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
