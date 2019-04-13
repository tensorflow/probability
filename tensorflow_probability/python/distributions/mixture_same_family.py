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
"""The same-family Mixture distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util as distribution_utils
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util

from tensorflow.python.ops import array_ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.parallel_for import gradients  # pylint: disable=g-direct-tensorflow-import


class MixtureSameFamily(distribution.Distribution):
  """Mixture (same-family) distribution.

  The `MixtureSameFamily` distribution implements a (batch of) mixture
  distribution where all components are from different parameterizations of the
  same distribution type. It is parameterized by a `Categorical` "selecting
  distribution" (over `k` components) and a components distribution, i.e., a
  `Distribution` with a rightmost batch shape (equal to `[k]`) which indexes
  each (batch of) component.

  #### Examples

  ```python
  tfd = tfp.distributions

  ### Create a mixture of two scalar Gaussians:

  gm = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=[0.3, 0.7]),
      components_distribution=tfd.Normal(
        loc=[-1., 1],       # One for each component.
        scale=[0.1, 0.5]))  # And same here.

  gm.mean()
  # ==> 0.4

  gm.variance()
  # ==> 1.018

  # Plot PDF.
  x = np.linspace(-2., 3., int(1e4), dtype=np.float32)
  import matplotlib.pyplot as plt
  plt.plot(x, gm.prob(x).eval());

  ### Create a mixture of two Bivariate Gaussians:

  gm = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=[0.3, 0.7]),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=[[-1., 1],  # component 1
               [1, -1]],  # component 2
          scale_identity_multiplier=[.3, .6]))

  gm.mean()
  # ==> array([ 0.4, -0.4], dtype=float32)

  gm.covariance()
  # ==> array([[ 1.119, -0.84],
  #            [-0.84,  1.119]], dtype=float32)

  # Plot PDF contours.
  def meshgrid(x, y=x):
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
  grid = meshgrid(np.linspace(-2, 2, 100, dtype=np.float32))
  plt.contour(grid[..., 0], grid[..., 1], gm.prob(grid).eval());

  ```

  """

  def __init__(self,
               mixture_distribution,
               components_distribution,
               reparameterize=False,
               validate_args=False,
               allow_nan_stats=True,
               name="MixtureSameFamily"):
    """Construct a `MixtureSameFamily` distribution.

    Args:
      mixture_distribution: `tfp.distributions.Categorical`-like instance.
        Manages the probability of selecting components. The number of
        categories must match the rightmost batch dimension of the
        `components_distribution`. Must have either scalar `batch_shape` or
        `batch_shape` matching `components_distribution.batch_shape[:-1]`.
      components_distribution: `tfp.distributions.Distribution`-like instance.
        Right-most batch dimension indexes components.
      reparameterize: Python `bool`, default `False`. Whether to reparameterize
        samples of the distribution using implicit reparameterization gradients
        [(Figurnov et al., 2018)][1]. The gradients for the mixture logits are
        equivalent to the ones described by [(Graves, 2016)][2]. The gradients
        for the components parameters are also computed using implicit
        reparameterization (as opposed to ancestral sampling), meaning that
        all components are updated every step.
        Only works when:
          (1) components_distribution is fully reparameterized;
          (2) components_distribution is either a scalar distribution or
          fully factorized (tfd.Independent applied to a scalar distribution);
          (3) batch shape has a known rank.
        Experimental, may be slow and produce infs/NaNs.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.

    Raises:
      ValueError: `if not dtype_util.is_integer(mixture_distribution.dtype)`.
      ValueError: if mixture_distribution does not have scalar `event_shape`.
      ValueError: if `mixture_distribution.batch_shape` and
        `components_distribution.batch_shape[:-1]` are both fully defined and
        the former is neither scalar nor equal to the latter.
      ValueError: if `mixture_distribution` categories does not equal
        `components_distribution` rightmost batch shape.

    #### References

    [1]: Michael Figurnov, Shakir Mohamed and Andriy Mnih. Implicit
         reparameterization gradients. In _Neural Information Processing
         Systems_, 2018. https://arxiv.org/abs/1805.08498

    [2]: Alex Graves. Stochastic Backpropagation through Mixture Density
         Distributions. _arXiv_, 2016. https://arxiv.org/abs/1607.05690
    """
    parameters = dict(locals())
    with tf.name_scope(name) as name:
      self._mixture_distribution = mixture_distribution
      self._components_distribution = components_distribution
      self._runtime_assertions = []

      s = components_distribution.event_shape_tensor()
      self._event_ndims = tf.compat.dimension_value(s.shape[0])
      if self._event_ndims is None:
        self._event_ndims = tf.size(input=s)
      self._event_size = tf.reduce_prod(input_tensor=s)

      if not dtype_util.is_integer(mixture_distribution.dtype):
        raise ValueError(
            "`mixture_distribution.dtype` ({}) is not over integers".format(
                dtype_util.name(mixture_distribution.dtype)))

      if (tensorshape_util.rank(mixture_distribution.event_shape) is not None
          and tensorshape_util.rank(mixture_distribution.event_shape) != 0):
        raise ValueError("`mixture_distribution` must have scalar `event_dim`s")
      elif validate_args:
        self._runtime_assertions += [
            assert_util.assert_equal(
                tf.size(input=mixture_distribution.event_shape_tensor()),
                0,
                message="`mixture_distribution` must have scalar `event_dim`s"),
        ]

      mdbs = mixture_distribution.batch_shape
      cdbs = tensorshape_util.with_rank_at_least(
          components_distribution.batch_shape, 1)[:-1]
      if tensorshape_util.is_fully_defined(
          mdbs) and tensorshape_util.is_fully_defined(cdbs):
        if tensorshape_util.rank(mdbs) != 0 and mdbs != cdbs:
          raise ValueError(
              "`mixture_distribution.batch_shape` (`{}`) is not "
              "compatible with `components_distribution.batch_shape` "
              "(`{}`)".format(
                  tensorshape_util.as_list(mdbs),
                  tensorshape_util.as_list(cdbs)))
      elif validate_args:
        mdbs = mixture_distribution.batch_shape_tensor()
        cdbs = components_distribution.batch_shape_tensor()[:-1]
        self._runtime_assertions += [
            assert_util.assert_equal(
                distribution_utils.pick_vector(
                    mixture_distribution.is_scalar_batch(), cdbs, mdbs),
                cdbs,
                message=(
                    "`mixture_distribution.batch_shape` is not "
                    "compatible with `components_distribution.batch_shape`"))
        ]

      km = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(mixture_distribution.logits.shape,
                                              1)[-1])
      kc = tf.compat.dimension_value(
          tensorshape_util.with_rank_at_least(
              components_distribution.batch_shape, 1)[-1])
      if km is not None and kc is not None and km != kc:
        raise ValueError("`mixture_distribution components` ({}) does not "
                         "equal `components_distribution.batch_shape[-1]` "
                         "({})".format(km, kc))
      elif validate_args:
        km = tf.shape(input=mixture_distribution.logits)[-1]
        kc = components_distribution.batch_shape_tensor()[-1]
        self._runtime_assertions += [
            assert_util.assert_equal(
                km,
                kc,
                message=("`mixture_distribution components` does not equal "
                         "`components_distribution.batch_shape[-1:]`")),
        ]
      elif km is None:
        km = tf.shape(input=mixture_distribution.logits)[-1]

      self._num_components = km

      self._reparameterize = reparameterize
      if reparameterize:
        # Note: tfd.Independent passes through the reparameterization type hence
        # we do not need separate logic for Independent.
        if (self._components_distribution.reparameterization_type !=
            reparameterization.FULLY_REPARAMETERIZED):
          raise ValueError("Cannot reparameterize a mixture of "
                           "non-reparameterized components.")
        reparameterization_type = reparameterization.FULLY_REPARAMETERIZED
      else:
        reparameterization_type = reparameterization.NOT_REPARAMETERIZED

      super(MixtureSameFamily, self).__init__(
          dtype=self._components_distribution.dtype,
          reparameterization_type=reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          graph_parents=(
              self._mixture_distribution._graph_parents  # pylint: disable=protected-access
              + self._components_distribution._graph_parents),  # pylint: disable=protected-access
          name=name)

  @property
  def mixture_distribution(self):
    return self._mixture_distribution

  @property
  def components_distribution(self):
    return self._components_distribution

  def __getitem__(self, slices):
    # Because slicing is parameterization-dependent, we only implement slicing
    # for instances of MSF, not subclasses thereof.
    if type(self) is not MixtureSameFamily:  # pylint: disable=unidiomatic-typecheck
      return super(MixtureSameFamily, self).__getitem__(slices)

    slices = (
        list(slices) if isinstance(slices, collections.Sequence) else [slices])
    mixture_rank = tensorshape_util.rank(self.mixture_distribution.batch_shape)
    if mixture_rank is None:
      raise NotImplementedError("Cannot slice MixtureSameFamily with unknown "
                                "mixture_distribution rank")
    elif mixture_rank > 0:
      sliced_mixture_dist = self.mixture_distribution.__getitem__(slices)
    else:  # must be scalar
      sliced_mixture_dist = self.mixture_distribution

    # The components distribution has the component axis as the last batch dim,
    # and this must be preserved.
    if Ellipsis not in slices:
      slices.append(Ellipsis)
    slices.append(slice(None))
    sliced_components_dist = self.components_distribution.__getitem__(slices)
    return self.copy(
        mixture_distribution=sliced_mixture_dist,
        components_distribution=sliced_components_dist)

  def _batch_shape_tensor(self):
    with tf.control_dependencies(self._runtime_assertions):
      return self.components_distribution.batch_shape_tensor()[:-1]

  def _batch_shape(self):
    return tensorshape_util.with_rank_at_least(
        self.components_distribution.batch_shape, 1)[:-1]

  def _event_shape_tensor(self):
    with tf.control_dependencies(self._runtime_assertions):
      return self.components_distribution.event_shape_tensor()

  def _event_shape(self):
    return self.components_distribution.event_shape

  def _sample_n(self, n, seed):
    with tf.control_dependencies(self._runtime_assertions):
      seed = seed_stream.SeedStream(seed, salt="MixtureSameFamily")
      x = self.components_distribution.sample(n, seed=seed())  # [n, B, k, E]
      # TODO(jvdillon): Consider using tf.gather (by way of index unrolling).
      npdt = dtype_util.as_numpy_dtype(x.dtype)
      mask = tf.one_hot(
          indices=self.mixture_distribution.sample(n, seed=seed()),  # [n, B]
          depth=self._num_components,  # == k
          on_value=npdt(1),
          off_value=npdt(0))  # [n, B, k]
      mask = distribution_utils.pad_mixture_dimensions(
          mask, self, self.mixture_distribution,
          self._event_ndims)                         # [n, B, k, [1]*e]
      x = tf.reduce_sum(
          input_tensor=x * mask, axis=-1 - self._event_ndims)  # [n, B, E]
      if self._reparameterize:
        x = self._reparameterize_sample(x)
      return x

  def _log_prob(self, x):
    with tf.control_dependencies(self._runtime_assertions):
      x = self._pad_sample_dims(x)
      log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
      log_mix_prob = tf.nn.log_softmax(
          self.mixture_distribution.logits, axis=-1)  # [B, k]
      return tf.reduce_logsumexp(
          input_tensor=log_prob_x + log_mix_prob, axis=-1)  # [S, B]

  def _mean(self):
    with tf.control_dependencies(self._runtime_assertions):
      probs = distribution_utils.pad_mixture_dimensions(
          self.mixture_distribution.probs, self, self.mixture_distribution,
          self._event_ndims)                         # [B, k, [1]*e]
      return tf.reduce_sum(
          input_tensor=probs * self.components_distribution.mean(),
          axis=-1 - self._event_ndims)  # [B, E]

  def _log_cdf(self, x):
    x = self._pad_sample_dims(x)
    log_cdf_x = self.components_distribution.log_cdf(x)      # [S, B, k]
    log_mix_prob = tf.nn.log_softmax(
        self.mixture_distribution.logits, axis=-1)  # [B, k]
    return tf.reduce_logsumexp(
        input_tensor=log_cdf_x + log_mix_prob, axis=-1)  # [S, B]

  def _variance(self):
    with tf.control_dependencies(self._runtime_assertions):
      # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
      probs = distribution_utils.pad_mixture_dimensions(
          self.mixture_distribution.probs, self, self.mixture_distribution,
          self._event_ndims)                         # [B, k, [1]*e]
      mean_cond_var = tf.reduce_sum(
          input_tensor=probs * self.components_distribution.variance(),
          axis=-1 - self._event_ndims)  # [B, E]
      var_cond_mean = tf.reduce_sum(
          input_tensor=probs *
          tf.math.squared_difference(self.components_distribution.mean(),
                                     self._pad_sample_dims(self._mean())),
          axis=-1 - self._event_ndims)  # [B, E]
      return mean_cond_var + var_cond_mean                   # [B, E]

  def _covariance(self):
    static_event_ndims = tensorshape_util.rank(self.event_shape)
    if static_event_ndims is not None and static_event_ndims != 1:
      # Covariance is defined only for vector distributions.
      raise NotImplementedError("covariance is not implemented")

    with tf.control_dependencies(self._runtime_assertions):
      # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
      probs = distribution_utils.pad_mixture_dimensions(
          distribution_utils.pad_mixture_dimensions(
              self.mixture_distribution.probs, self, self.mixture_distribution,
              self._event_ndims),
          self, self.mixture_distribution,
          self._event_ndims)                         # [B, k, 1, 1]
      mean_cond_var = tf.reduce_sum(
          input_tensor=probs * self.components_distribution.covariance(),
          axis=-3)  # [B, e, e]
      var_cond_mean = tf.reduce_sum(
          input_tensor=probs *
          _outer_squared_difference(self.components_distribution.mean(),
                                    self._pad_sample_dims(self._mean())),
          axis=-3)  # [B, e, e]
      return mean_cond_var + var_cond_mean                   # [B, e, e]

  def _pad_sample_dims(self, x):
    with tf.name_scope("pad_sample_dims"):
      ndims = tensorshape_util.rank(
          x.shape) if tensorshape_util.rank(x.shape) is not None else tf.rank(x)
      shape = tf.shape(input=x)
      d = ndims - self._event_ndims
      x = tf.reshape(x, shape=tf.concat([shape[:d], [1], shape[d:]], axis=0))
      return x

  def _reparameterize_sample(self, x):
    """Adds reparameterization (pathwise) gradients to samples of the mixture.

    Implicit reparameterization gradients are
       dx/dphi = -(d transform(x, phi) / dx)^-1 * d transform(x, phi) / dphi,
    where transform(x, phi) is distributional transform that removes all
    parameters from samples x.

    We implement them by replacing x with
      -stop_gradient(d transform(x, phi) / dx)^-1 * transform(x, phi)]
    for the backward pass (gradient computation).
    The derivative of this quantity w.r.t. phi is then the implicit
    reparameterization gradient.
    Note that this replaces the gradients w.r.t. both the mixture
    distribution parameters and components distributions parameters.

    Limitations:
      1. Fundamental: components must be fully reparameterized.
      2. Distributional transform is currently only implemented for
        factorized components.
      3. Distributional transform currently only works for known rank of the
        batch tensor.

    Arguments:
      x: Sample of mixture distribution

    Returns:
      Tensor with same value as x, but with reparameterization gradients
    """
    # Remove the existing gradients of x wrt parameters of the components.
    x = tf.stop_gradient(x)

    x_2d_shape = [-1, self._event_size]  # [S*prod(B), prod(E)]

    # Perform distributional transform of x in [S, B, E] shape,
    # but have Jacobian of size [S*prod(B), prod(E), prod(E)].
    def reshaped_distributional_transform(x_2d):
      return tf.reshape(
          self._distributional_transform(tf.reshape(x_2d, tf.shape(input=x))),
          x_2d_shape)

    # transform_2d: [S*prod(B), prod(E)]
    # jacobian: [S*prod(B), prod(E), prod(E)]
    transform_2d, jacobian = _value_and_batch_jacobian(
        reshaped_distributional_transform, tf.reshape(x, x_2d_shape))

    # We only provide the first derivative; the second derivative computed by
    # autodiff would be incorrect, so we raise an error if it is requested.
    transform_2d = _prevent_2nd_derivative(transform_2d)

    # Compute [- stop_gradient(jacobian)^-1 * transform] by solving a linear
    # system. The Jacobian is lower triangular because the distributional
    # transform for i-th event dimension does not depend on the next
    # dimensions.
    surrogate_x_2d = -tf.linalg.triangular_solve(
        tf.stop_gradient(jacobian), tf.expand_dims(transform_2d, axis=-1),
        lower=True)  # [S*prod(B), prod(E), 1]
    surrogate_x = tf.reshape(surrogate_x_2d, tf.shape(input=x))

    # Replace gradients of x with gradients of surrogate_x, but keep the value.
    return x + (surrogate_x - tf.stop_gradient(surrogate_x))

  def _distributional_transform(self, x):
    """Performs distributional transform of the mixture samples.

    Distributional transform removes the parameters from samples of a
    multivariate distribution by applying conditional CDFs:
      (F(x_1), F(x_2 | x1_), ..., F(x_d | x_1, ..., x_d-1))
    (the indexing is over the "flattened" event dimensions).
    The result is a sample of product of Uniform[0, 1] distributions.

    We assume that the components are factorized, so the conditional CDFs become
      F(x_i | x_1, ..., x_i-1) = sum_k w_i^k F_k (x_i),
    where w_i^k is the posterior mixture weight: for i > 0
      w_i^k = w_k prob_k(x_1, ..., x_i-1) / sum_k' w_k' prob_k'(x_1, ..., x_i-1)
    and w_0^k = w_k is the mixture probability of the k-th component.

    Arguments:
      x: Sample of mixture distribution

    Returns:
      Result of the distributional transform
    """

    if tensorshape_util.rank(x.shape) is None:
      # tf.nn.softmax raises an error when applied to inputs of undefined rank.
      raise ValueError("Distributional transform does not support inputs of "
                       "undefined rank.")

    # Obtain factorized components distribution and assert that it's
    # a scalar distribution.
    if isinstance(self._components_distribution, independent.Independent):
      univariate_components = self._components_distribution.distribution
    else:
      univariate_components = self._components_distribution

    with tf.control_dependencies([
        assert_util.assert_equal(
            univariate_components.is_scalar_event(),
            True,
            message="`univariate_components` must have scalar event")
    ]):
      x_padded = self._pad_sample_dims(x)  # [S, B, 1, E]
      log_prob_x = univariate_components.log_prob(x_padded)  # [S, B, k, E]
      cdf_x = univariate_components.cdf(x_padded)  # [S, B, k, E]

      # log prob_k (x_1, ..., x_i-1)
      cumsum_log_prob_x = tf.reshape(
          tf.math.cumsum(
              # [S*prod(B)*k, prod(E)]
              tf.reshape(log_prob_x, [-1, self._event_size]),
              exclusive=True,
              axis=-1),
          tf.shape(input=log_prob_x))  # [S, B, k, E]

      logits_mix_prob = distribution_utils.pad_mixture_dimensions(
          self.mixture_distribution.logits, self, self.mixture_distribution,
          self._event_ndims)  # [B, k, 1]

      # Logits of the posterior weights: log w_k + log prob_k (x_1, ..., x_i-1)
      log_posterior_weights_x = logits_mix_prob + cumsum_log_prob_x

      component_axis = tensorshape_util.rank(x.shape) - self._event_ndims
      posterior_weights_x = tf.nn.softmax(log_posterior_weights_x,
                                          axis=component_axis)
      return tf.reduce_sum(
          input_tensor=posterior_weights_x * cdf_x, axis=component_axis)


def _outer_squared_difference(x, y):
  """Convenience function analogous to tf.squared_difference."""
  z = x - y
  return z[..., tf.newaxis, :] * z[..., tf.newaxis]


def _value_and_batch_jacobian(f, x):
  """Enables uniform interface to value and batch jacobian calculation.

  Works in both eager and graph modes.

  Arguments:
    f: The scalar function to evaluate.
    x: The value at which to compute the value and the batch jacobian.

  Returns:
    A tuple (f(x), J(x)), where J(x) is the batch jacobian.
  """
  if tf.executing_eagerly():
    with tf.GradientTape() as tape:
      tape.watch(x)
      value = f(x)
    batch_jacobian = tape.batch_jacobian(value, x)
  else:
    value = f(x)
    batch_jacobian = gradients.batch_jacobian(value, x)
  return value, batch_jacobian


@tf.custom_gradient
def _prevent_2nd_derivative(x):
  """Disables computation of the second derivatives for a tensor.

  NB: you need to apply a non-identity function to the output tensor for the
  exception to be raised.

  Arguments:
    x: A tensor.

  Returns:
    A tensor with the same value and the same derivative as x, but that raises
    LookupError when trying to compute the second derivatives.
  """
  def grad(dy):
    return array_ops.prevent_gradient(
        dy, message="Second derivative is not implemented.")

  return tf.identity(x), grad
