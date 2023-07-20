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

import warnings

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import batch_broadcast
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.gradient import value_and_batch_jacobian


# Cause all warnings to always be triggered.
# Not having this means subsequent calls won't trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*mixture_same_family',
                        append=True)  # Don't override user-set filters.


class _MixtureSameFamily(distribution.Distribution):
  """Mixture (same-family) distribution.

  The `MixtureSameFamily` distribution implements a (batch of) mixture
  distribution where all components are from different parameterizations of the
  same distribution type. It is parameterized by a `Categorical` 'selecting
  distribution' (over `k` components) and a components distribution, i.e., a
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
  plt.plot(x, gm.prob(x));

  ### Create a mixture of three Bivariate Gaussians:

  gm = tfd.MixtureSameFamily(
      mixture_distribution=tfd.Categorical(
          probs=[0.2, 0.4, 0.4]),
      components_distribution=tfd.MultivariateNormalDiag(
          loc=[[-1., 1],  # component 1
               [1, -1],  # component 2
               [1, 1]],  # component 3
          scale_diag=tf.tile([[.3], [.6], [.7]], [1, 2]))

  gm.components_distribution.batch_shape
  # ==> (3,)

  gm.components_distribution.event_shape
  # ==> (2,)

  gm.mean()
  # ==> array([ 0.6, 0.2], dtype=float32)

  gm.covariance()
  # ==> array([[ 0.998    , -0.32     ],
  #            [-0.32     ,  1.3180001]], dtype=float32)

  # Plot PDF contours.
  def meshgrid(x):
    y = x
    [gx, gy] = np.meshgrid(x, y, indexing='ij')
    gx, gy = np.float32(gx), np.float32(gy)
    grid = np.concatenate([gx.ravel()[None, :], gy.ravel()[None, :]], axis=0)
    return grid.T.reshape(x.size, y.size, 2)
  grid = meshgrid(np.linspace(-2, 2, 100, dtype=np.float32))
  plt.contour(grid[..., 0], grid[..., 1], gm.prob(grid));
  ```

  Note that this distribution is *not* a joint distribution over categorical
  and continuous values, but rather a mixture of continuous distributions
  proportioned by the given categorical distribution. If you want a joint
  distribution, you might write it as:

  ```python
  @tfd.JointDistributionCoroutineAutoBatched
  def model():
    mus = tf.constant([[-1., 1], # component 1
                       [1, -1],  # component 2
                       [1, 1]])  # component 3
    scales = tf.constant([.3, .6, .7])
    idx = yield tfd.Categorical(probs=[.2, .4, .4], name='idx')
    val = yield tfd.MultivariateNormalDiag(
        loc=mus[idx], scale_diag=tf.ones(2) * scales[idx], name='val')

  model.sample()
  # ==> StructTuple(
  #       idx=2,
  #       val=array([1.0582672, 1.3583777], dtype=float32)
  #     )
  ```

  """

  def __init__(self,
               mixture_distribution,
               components_distribution,
               reparameterize=False,
               validate_args=False,
               allow_nan_stats=True,
               name='MixtureSameFamily'):
    """Construct a `MixtureSameFamily` distribution.

    Args:
      mixture_distribution: `tfd.Categorical`-like instance.
        Manages the probability of selecting components. The number of
        categories must match the rightmost batch dimension of the
        `components_distribution`. Must have `batch_shape` broadcastable
        with `components_distribution.batch_shape[:-1]`.
      components_distribution: `tfd.Distribution`-like instance.
        The right-most batch dimension indexes the mixture components.
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
        (e.g., mean, mode, variance) use the value '`NaN`' to indicate the
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

      self._reparameterize = reparameterize
      if reparameterize:
        # Note: tfd.Independent passes through the reparameterization type hence
        # we do not need separate logic for Independent.
        if (self._components_distribution.reparameterization_type !=
            reparameterization.FULLY_REPARAMETERIZED):
          raise ValueError('Cannot reparameterize a mixture of '
                           'non-reparameterized components.')
        reparameterization_type = reparameterization.FULLY_REPARAMETERIZED
      else:
        reparameterization_type = reparameterization.NOT_REPARAMETERIZED

      super(_MixtureSameFamily, self).__init__(
          dtype=self._components_distribution.dtype,
          reparameterization_type=reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def _event_ndims(self):
    return ps.rank_from_shape(
        self.components_distribution.event_shape_tensor,
        self.components_distribution.event_shape)

  @property
  def mixture_distribution(self):
    return self._mixture_distribution

  @property
  def components_distribution(self):
    return self._components_distribution

  @property
  def experimental_is_sharded(self):
    sharded = self.components_distribution.experimental_is_sharded
    if self.mixture_distribution.experimental_is_sharded != sharded:
      raise ValueError(
          '`MixtureSameFamily.mixture_distribution` sharding must match '
          '`MixtureSameFamily.components_distribution`.')
    return sharded

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        mixture_distribution=(
            parameter_properties.BatchedComponentProperties()),
        components_distribution=(
            parameter_properties.BatchedComponentProperties(
                event_ndims=1)))

  def _get_distributions_with_broadcast_batch_shape(self):
    """Broadcasts the mixture and component dists to have full batch shape."""
    overall_batch_shape = self.batch_shape
    if (tensorshape_util.is_fully_defined(overall_batch_shape) and
        self.components_distribution.batch_shape[:-1] == overall_batch_shape and
        self.mixture_distribution.batch_shape == overall_batch_shape):
      # No need to broadcast.
      return self.mixture_distribution, self.components_distribution

    if not tensorshape_util.is_fully_defined(overall_batch_shape):
      overall_batch_shape = self.batch_shape_tensor()
    # The mixture distribution is primarily accessed through its parameters
    # (e.g., logits), so broadcast those directly.
    mixture_distribution = (
        self.mixture_distribution._broadcast_parameters_with_batch_shape(
            overall_batch_shape))
    components_distribution = batch_broadcast.BatchBroadcast(
        self.components_distribution,
        with_shape=ps.concat([overall_batch_shape, [1]], axis=0))
    return mixture_distribution, components_distribution

  def _event_shape_tensor(self):
    return self.components_distribution.event_shape_tensor()

  def _event_shape(self):
    return self.components_distribution.event_shape

  def _sample_n(self, n, seed):
    components_seed, mix_seed = samplers.split_seed(seed,
                                                    salt='MixtureSameFamily')
    mixture_distribution, components_distribution = (
        self._get_distributions_with_broadcast_batch_shape())
    x = components_distribution.sample(  # [n, B, k, E]
        n, seed=components_seed)

    event_ndims = ps.rank_from_shape(self.event_shape_tensor, self.event_shape)
    # We could also check if num_components can be computed statically from
    # self.mixture_distribution's logits or probs.
    num_components = ps.dimension_size(x, idx=-1 - event_ndims)

    # TODO(jvdillon): Consider using tf.gather (by way of index unrolling).
    npdt = dtype_util.as_numpy_dtype(x.dtype)
    mix_sample = mixture_distribution.sample(
        n, seed=mix_seed)  # [n, B]
    mask = tf.one_hot(
        indices=mix_sample,  # [n, B]
        depth=num_components,
        on_value=npdt(1),
        off_value=npdt(0))    # [n, B, k]

    # Pad `mask` to [n, B, k, [1]*e].
    batch_ndims = ps.rank(x) - event_ndims - 1
    mask_batch_ndims = ps.rank(mask) - 1
    pad_ndims = batch_ndims - mask_batch_ndims
    mask_shape = ps.shape(mask)
    target_shape = ps.concat([
        mask_shape[:-1],
        ps.ones([pad_ndims], dtype=tf.int32),
        mask_shape[-1:],
        ps.ones([event_ndims], dtype=tf.int32),
    ], axis=0)
    mask = tf.reshape(mask, shape=target_shape)

    if dtype_util.is_floating(x.dtype) or dtype_util.is_complex(x.dtype):
      masked = tf.math.multiply_no_nan(x, mask)
    else:
      masked = x * mask
    ret = tf.reduce_sum(masked, axis=-1 - event_ndims)  # [n, B, E]

    if self._reparameterize:
      ret = self._reparameterize_sample(
          ret, event_shape=components_distribution.event_shape_tensor())

    return ret

  def _per_mixture_component_log_prob(self, x):
    """Per mixture component log probability.

    Args:
      x: A tensor representing observations from the mixture. Must
        be broadcastable with the mixture's batch shape.

    Returns:
      A Tensor representing, for each observation and for each mixture
      component, the log joint probability of that mixture component and
      the observation. The shape will be equal to the concatenation of (1) the
      broadcast shape of the observations and the batch shape, and (2) the
      number of mixture components.
    """
    x = self._pad_sample_dims(x)
    log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
    log_mix_prob = tf.math.log_softmax(
        self.mixture_distribution.logits_parameter(), axis=-1)  # [B, k]
    return log_prob_x + log_mix_prob  # [S, B, k]

  def _log_prob(self, x):
    return tf.reduce_logsumexp(
        self._per_mixture_component_log_prob(x), axis=-1)  # [S, B]

  def _mean(self):
    mixture_distribution, components_distribution = (
        self._get_distributions_with_broadcast_batch_shape())
    probs = mixture_distribution.probs_parameter()  # [B, k]
    component_means = components_distribution.mean()  # [B, k, E]
    event_ndims = self._event_ndims()

    # reshape probs to [B, k, [1]*e]
    probs = tf.reshape(probs, ps.concat([
        ps.shape(probs),
        ps.ones([event_ndims], dtype=tf.int32)
    ], axis=0))

    return tf.reduce_sum(probs * component_means,
                         axis=-1 - event_ndims)  # [B, E]

  def _log_cdf(self, x):
    x = self._pad_sample_dims(x)
    log_cdf_x = self.components_distribution.log_cdf(x)      # [S, B, k]
    log_mix_prob = tf.math.log_softmax(
        self.mixture_distribution.logits_parameter(), axis=-1)  # [B, k]
    return tf.reduce_logsumexp(log_cdf_x + log_mix_prob, axis=-1)  # [S, B]

  def _variance(self):
    mixture_distribution, components_distribution = (
        self._get_distributions_with_broadcast_batch_shape())
    probs = mixture_distribution.probs_parameter()  # [B, k]
    component_means = components_distribution.mean()  # [B, k, E]
    component_vars = components_distribution.variance()  # [B, k, E]
    event_ndims = self._event_ndims()

    # reshape probs to [B, k, [1]*e]
    probs = tf.reshape(probs, ps.concat([
        ps.shape(probs),
        ps.ones([event_ndims], dtype=tf.int32)
    ], axis=0))

    # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
    mean_cond_var = tf.reduce_sum(probs * component_vars,
                                  axis=-1 - event_ndims)  # [B, E]
    mean = tf.reduce_sum(probs * component_means,
                         axis=-1 - event_ndims, keepdims=True)  # [B, 1, E]
    var_cond_mean = tf.reduce_sum(
        probs * tf.math.squared_difference(component_means, mean),
        axis=-1 - event_ndims)  # [B, E]
    return mean_cond_var + var_cond_mean

  def _covariance(self):
    static_event_ndims = tensorshape_util.rank(self.event_shape)
    if static_event_ndims is not None and static_event_ndims != 1:
      # Covariance is defined only for vector distributions.
      raise NotImplementedError('covariance is not implemented')
    mixture_distribution, components_distribution = (
        self._get_distributions_with_broadcast_batch_shape())
    probs = mixture_distribution.probs_parameter()  # [B, k]
    component_means = components_distribution.mean()  # [B, k, E]
    component_covars = components_distribution.covariance()  # [B, k, E, E]

    # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
    probs = probs[..., tf.newaxis, tf.newaxis]  # [B, k, 1, 1]
    mean_cond_var = tf.reduce_sum(
        probs * component_covars, axis=-3)  # [B, E, E]
    mean = tf.reduce_sum(
        probs[..., 0] * component_means, axis=-2, keepdims=True)  # [B, 1, E]
    var_cond_mean = tf.reduce_sum(
        probs * _outer_squared_difference(component_means, mean),
        axis=-3)  # [B, E, E]
    return mean_cond_var + var_cond_mean  # [B, E, E]

  def posterior_marginal(self, observations, name='posterior_marginals'):
    """Compute the marginal posterior distribution for a batch of observations.

    Note: The behavior of this function is undefined if the `observations`
    argument represents impossible observations from the model.

    Args:
      observations: A tensor representing observations from the mixture. Must
        be broadcastable with the mixture's batch shape.
      name: A string naming a scope.

    Returns:
      posterior_marginals: A `Categorical` distribution object representing
        the marginal probability of the components of the mixture. The batch
        shape of the `Categorical` will be the broadcast shape of `observations`
        and the mixture batch shape; the number of classes will equal the
        number of mixture components.
    """
    with self._name_and_control_scope(name):
      return categorical.Categorical(
          logits=self._per_mixture_component_log_prob(observations))

  def posterior_mode(self, observations, name='posterior_mode'):
    """Compute the posterior mode for a batch of distributions.

    Note: The behavior of this function is undefined if the `observations`
    argument represents impossible observations from the mixture.

    Args:
      observations: A tensor representing observations from the mixture. Must
        be broadcastable with the mixture's batch shape.
      name: A string naming a scope.

    Returns:
      A Tensor representing the mode (most likely component) for each
      observation. The shape will be equal to the broadcast shape of the
      observations and the batch shape.
    """
    with self._name_and_control_scope(name):
      return tf.math.argmax(
          self._per_mixture_component_log_prob(observations), axis=-1)

  def _pad_sample_dims(self, x, event_ndims=None):
    with tf.name_scope('pad_sample_dims'):
      if event_ndims is None:
        event_ndims = self._event_ndims()
      ndims = ps.rank(x)
      # Must do the c_t_t in case ndims or event_ndims are Tensors and shape is
      # ndarray. Otherwise we get `TypeError: slice indices must be integers
      # or None or have an __index__ method`.
      shape = ps.convert_to_shape_tensor(ps.shape(x))
      d = ndims - event_ndims
      x = tf.reshape(
          x, shape=ps.concat([shape[:d], [1], shape[d:]], axis=0))
      return x

  def _reparameterize_sample(self, x, event_shape):
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

    Args:
      x: Sample of mixture distribution
      event_shape: The event shape of this distribution

    Returns:
      Tensor with same value as x, but with reparameterization gradients
    """
    # Remove the existing gradients of x wrt parameters of the components.
    x = tf.stop_gradient(x)

    event_size = ps.cast(
        ps.reduce_prod(event_shape), dtype=tf.int32)
    x_2d_shape = [-1, event_size]  # [S*prod(B), prod(E)]

    # Perform distributional transform of x in [S, B, E] shape,
    # but have Jacobian of size [S*prod(B), prod(E), prod(E)].
    def reshaped_distributional_transform(x_2d):
      return tf.reshape(
          self._distributional_transform(
              tf.reshape(x_2d, ps.shape(x)), event_shape),
          x_2d_shape)

    # transform_2d: [S*prod(B), prod(E)]
    # jacobian: [S*prod(B), prod(E), prod(E)]
    x_2d = tf.reshape(x, x_2d_shape)
    transform_2d, jacobian = value_and_batch_jacobian(
        reshaped_distributional_transform, x_2d)

    # We only provide the first derivative; the second derivative computed by
    # autodiff would be incorrect, so we raise an error if it is requested.
    transform_2d = _prevent_2nd_derivative(transform_2d)

    # Avoid short-circuiting from `tf.stop_gradient`, which might otherwise hide
    # the 2nd derivative prevention.
    soft_stop_gradient = tf.custom_gradient(lambda v: (v, lambda dv: None))
    # Compute [- stop_gradient(jacobian)^-1 * transform] by solving a linear
    # system. The Jacobian is lower triangular because the distributional
    # transform for i-th event dimension does not depend on the next
    # dimensions.
    surrogate_x_2d = -tf.linalg.triangular_solve(
        soft_stop_gradient(jacobian), transform_2d[..., tf.newaxis],
        lower=True)  # [S*prod(B), prod(E), 1]
    surrogate_x = tf.reshape(surrogate_x_2d, ps.shape(x))

    # Replace gradients of x with gradients of surrogate_x, but keep the value.
    return x + (surrogate_x - tf.stop_gradient(surrogate_x))

  def _distributional_transform(self, x, event_shape):
    """Performs distributional transform of the mixture samples.

    Distributional transform removes the parameters from samples of a
    multivariate distribution by applying conditional CDFs:
      (F(x_1), F(x_2 | x1_), ..., F(x_d | x_1, ..., x_d-1))
    (the indexing is over the 'flattened' event dimensions).
    The result is a sample of product of Uniform[0, 1] distributions.

    We assume that the components are factorized, so the conditional CDFs become
      F(x_i | x_1, ..., x_i-1) = sum_k w_i^k F_k (x_i),
    where w_i^k is the posterior mixture weight: for i > 0
      w_i^k = w_k prob_k(x_1, ..., x_i-1) / sum_k' w_k' prob_k'(x_1, ..., x_i-1)
    and w_0^k = w_k is the mixture probability of the k-th component.

    Args:
      x: Sample of mixture distribution
      event_shape: The event shape of this distribution

    Returns:
      Result of the distributional transform
    """

    if tensorshape_util.rank(x.shape) is None:
      # tf.math.softmax raises an error when applied to inputs of undefined
      # rank.
      raise ValueError('Distributional transform does not support inputs of '
                       'undefined rank.')

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
            message='`univariate_components` must have scalar event')
    ]):
      event_ndims = ps.rank_from_shape(event_shape)
      x_padded = self._pad_sample_dims(
          x, event_ndims=event_ndims)  # [S, B, 1, E]
      log_prob_x = univariate_components.log_prob(x_padded)  # [S, B, k, E]
      cdf_x = univariate_components.cdf(x_padded)  # [S, B, k, E]

      # log prob_k (x_1, ..., x_i-1)
      event_size = ps.cast(
          ps.reduce_prod(event_shape), dtype=tf.int32)
      cumsum_log_prob_x = tf.reshape(
          tf.math.cumsum(
              # [S*prod(B)*k, prod(E)]
              tf.reshape(log_prob_x, [-1, event_size]),
              exclusive=True,
              axis=-1),
          ps.shape(log_prob_x))  # [S, B, k, E]

      event_ndims = ps.rank_from_shape(event_shape)
      logits_mix_prob = self.mixture_distribution.logits_parameter()
      logits_mix_prob = tf.reshape(
          logits_mix_prob,  # [k] or [B, k]
          ps.concat([
              ps.shape(logits_mix_prob),
              ps.ones([event_ndims], dtype=tf.int32),
          ], axis=0))  # [k, [1]*e] or [B, k, [1]*e]

      # Logits of the posterior weights: log w_k + log prob_k (x_1, ..., x_i-1)
      log_posterior_weights_x = logits_mix_prob + cumsum_log_prob_x

      component_axis = tensorshape_util.rank(x.shape) - event_ndims
      posterior_weights_x = tf.math.softmax(
          log_posterior_weights_x, axis=component_axis)
      return tf.reduce_sum(posterior_weights_x * cdf_x, axis=component_axis)

  def _default_event_space_bijector(self):
    # TODO(b/146456627): Implement `default_event_space_bijector` for mixture
    # distributions.
    return

  def _parameter_control_dependencies(self, is_init):
    assertions = []

    if is_init and not dtype_util.is_integer(self.mixture_distribution.dtype):
      raise ValueError(
          '`mixture_distribution.dtype` ({}) is not over integers'.format(
              dtype_util.name(self.mixture_distribution.dtype)))

    if tensorshape_util.rank(self.mixture_distribution.event_shape) is not None:
      if tensorshape_util.rank(self.mixture_distribution.event_shape) != 0:
        raise ValueError('`mixture_distribution` must have scalar `event_dim`s')
    elif self.validate_args:
      assertions += [
          assert_util.assert_equal(
              tf.size(self.mixture_distribution.event_shape_tensor()),
              0,
              message='`mixture_distribution` must have scalar `event_dim`s'),
      ]

    # pylint: disable=protected-access
    mixture_dist_param = (self.mixture_distribution._probs
                          if self.mixture_distribution._logits is None
                          else self.mixture_distribution._logits)
    km = tf.compat.dimension_value(
        tensorshape_util.with_rank_at_least(mixture_dist_param.shape, 1)[-1])
    kc = tf.compat.dimension_value(
        tensorshape_util.with_rank_at_least(
            self.components_distribution.batch_shape, 1)[-1])
    component_bst = None
    if km is not None and kc is not None:
      if km != kc:
        raise ValueError('`mixture_distribution` components ({}) does not '
                         'equal `components_distribution.batch_shape[-1]` '
                         '({})'.format(km, kc))
    elif self.validate_args:
      if km is None:
        mixture_dist_param = tf.convert_to_tensor(mixture_dist_param)
        km = tf.shape(mixture_dist_param)[-1]
      if kc is None:
        component_bst = self.components_distribution.batch_shape_tensor()
        kc = component_bst[-1]
      assertions += [
          assert_util.assert_equal(
              km,
              kc,
              message=('`mixture_distribution` components does not equal '
                       '`components_distribution.batch_shape[-1]`')),
      ]

    return assertions


# TODO(b/182603117): Add a `ct_util.args_are_all_composite_tensor` method, or a
# similar class decorator that takes a mapping between arg positions and kwarg
# names and builds the CT or non-CT version of the distribution.
class MixtureSameFamily(
    _MixtureSameFamily, distribution.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_MixtureSameFamily`."""

    if cls is MixtureSameFamily:
      if args:
        mixture_distribution = args[0]
      else:
        mixture_distribution = kwargs.get('mixture_distribution')
      if len(args) > 1:
        components_distribution = args[1]
      else:
        components_distribution = kwargs.get('components_distribution')

      if not (isinstance(mixture_distribution, tf.__internal__.CompositeTensor)
              and isinstance(
                  components_distribution, tf.__internal__.CompositeTensor)):
        return _MixtureSameFamily(*args, **kwargs)
    return super(MixtureSameFamily, cls).__new__(cls)


MixtureSameFamily.__doc__ = _MixtureSameFamily.__doc__ + '\n' + (
    'If `mixture_distribution` and `components_distribution` are both '
    '`CompositeTensor`s, then the resulting `MixtureSameFamily` instance is a '
    '`CompositeTensor` as well. Otherwise, a non-`CompositeTensor` '
    '`_MixtureSameFamily` instance is created instead. Distribution subclasses '
    'that inherit from `MixtureSameFamily` will also inherit from '
    '`CompositeTensor`.')


def _outer_squared_difference(x, y):
  """Convenience function analogous to tf.squared_difference."""
  z = x - y
  return z[..., tf.newaxis, :] * z[..., tf.newaxis]


@tfp_custom_gradient.custom_gradient(
    vjp_fwd=lambda x: (tf.identity(x), ()),
    vjp_bwd=lambda _, dx: tfp_custom_gradient.prevent_gradient(  # pylint: disable=g-long-lambda
        dx, message='Second derivative is not implemented.'),
    jvp_fn=lambda primals, tangents: (  # pylint: disable=g-long-lambda
        tfp_custom_gradient.prevent_gradient(
            primals[0],
            message='Second derivative is not implemented.'),
        tfp_custom_gradient.prevent_gradient(
            tangents[0],
            message='Second derivative is not implemented.')))
def _prevent_2nd_derivative(x):
  """Disables computation of the second derivatives for a tensor.

  NB: you need to apply a non-identity function to the output tensor for the
  exception to be raised.

  Args:
    x: A tensor.

  Returns:
    A tensor with the same value and the same derivative as x, but that raises
    LookupError when trying to compute the second derivatives.
  """
  return tf.identity(x)
