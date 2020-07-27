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
import warnings

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import custom_gradient as tfp_custom_gradient
from tensorflow_probability.python.internal import distribution_util as distribution_utils
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.gradient import value_and_batch_jacobian
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.util.seed_stream import TENSOR_SEED_MSG_PREFIX


# Cause all warnings to always be triggered.
# Not having this means subsequent calls won't trigger the warning.
warnings.filterwarnings('always',
                        module='tensorflow_probability.*mixture_same_family',
                        append=True)  # Don't override user-set filters.


class MixtureSameFamily(distribution.Distribution):
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
               name='MixtureSameFamily'):
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

      super(MixtureSameFamily, self).__init__(
          dtype=self._components_distribution.dtype,
          reparameterization_type=reparameterization_type,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          parameters=parameters,
          name=name)

  def _event_ndims(self):
    return prefer_static.rank_from_shape(
        self.components_distribution.event_shape_tensor,
        self.components_distribution.event_shape)

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
      raise NotImplementedError('Cannot slice MixtureSameFamily with unknown '
                                'mixture_distribution rank')
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
    return self.components_distribution.batch_shape_tensor()[:-1]

  def _batch_shape(self):
    return tensorshape_util.with_rank_at_least(
        self.components_distribution.batch_shape, 1)[:-1]

  def _event_shape_tensor(self):
    return self.components_distribution.event_shape_tensor()

  def _event_shape(self):
    return self.components_distribution.event_shape

  def _sample_n(self, n, seed):
    components_seed, mix_seed = samplers.split_seed(seed,
                                                    salt='MixtureSameFamily')
    try:
      seed_stream = SeedStream(seed, salt='MixtureSameFamily')
    except TypeError as e:  # Can happen for Tensor seeds.
      seed_stream = None
      seed_stream_err = e
    try:
      x = self.components_distribution.sample(  # [n, B, k, E]
          n, seed=components_seed)
      if seed_stream is not None:
        seed_stream()  # Advance even if unused.
    except TypeError as e:
      if ('Expected int for argument' not in str(e) and
          TENSOR_SEED_MSG_PREFIX not in str(e)):
        raise
      if seed_stream is None:
        raise seed_stream_err
      msg = ('Falling back to stateful sampling for `components_distribution` '
             '{} of type `{}`. Please update to use `tf.random.stateless_*` '
             'RNGs. This fallback may be removed after 20-Aug-2020. {}')
      warnings.warn(msg.format(self.components_distribution.name,
                               type(self.components_distribution),
                               str(e)))
      x = self.components_distribution.sample(  # [n, B, k, E]
          n, seed=seed_stream())

    event_shape = None
    event_ndims = tensorshape_util.rank(self.event_shape)
    if event_ndims is None:
      event_shape = self.components_distribution.event_shape_tensor()
      event_ndims = prefer_static.rank_from_shape(event_shape)
    event_ndims_static = tf.get_static_value(event_ndims)

    num_components = None
    if event_ndims_static is not None:
      num_components = tf.compat.dimension_value(
          x.shape[-1 - event_ndims_static])
    # We could also check if num_components can be computed statically from
    # self.mixture_distribution's logits or probs.
    if num_components is None:
      num_components = tf.shape(x)[-1 - event_ndims]

    # TODO(jvdillon): Consider using tf.gather (by way of index unrolling).
    npdt = dtype_util.as_numpy_dtype(x.dtype)
    try:
      mix_sample = self.mixture_distribution.sample(
          n, seed=mix_seed)  # [n, B] or [n]
    except TypeError as e:
      if ('Expected int for argument' not in str(e) and
          TENSOR_SEED_MSG_PREFIX not in str(e)):
        raise
      if seed_stream is None:
        raise seed_stream_err
      msg = ('Falling back to stateful sampling for `mixture_distribution` '
             '{} of type `{}`. Please update to use `tf.random.stateless_*` '
             'RNGs. This fallback may be removed after 20-Aug-2020. ({})')
      warnings.warn(msg.format(self.mixture_distribution.name,
                               type(self.mixture_distribution),
                               str(e)))
      mix_sample = self.mixture_distribution.sample(
          n, seed=seed_stream())  # [n, B] or [n]
    mask = tf.one_hot(
        indices=mix_sample,  # [n, B] or [n]
        depth=num_components,
        on_value=npdt(1),
        off_value=npdt(0))    # [n, B, k] or [n, k]

    # Pad `mask` to [n, B, k, [1]*e] or [n, [1]*b, k, [1]*e] .
    batch_ndims = prefer_static.rank(x) - event_ndims - 1
    mask_batch_ndims = prefer_static.rank(mask) - 1
    pad_ndims = batch_ndims - mask_batch_ndims
    mask_shape = prefer_static.shape(mask)
    mask = tf.reshape(
        mask,
        shape=prefer_static.concat([
            mask_shape[:-1],
            prefer_static.ones([pad_ndims], dtype=tf.int32),
            mask_shape[-1:],
            prefer_static.ones([event_ndims], dtype=tf.int32),
        ], axis=0))

    if x.dtype in [tf.bfloat16, tf.float16, tf.float32, tf.float64,
                   tf.complex64, tf.complex128]:
      masked = tf.math.multiply_no_nan(x, mask)
    else:
      masked = x * mask
    ret = tf.reduce_sum(masked, axis=-1 - event_ndims)  # [n, B, E]

    if self._reparameterize:
      if event_shape is None:
        event_shape = self.components_distribution.event_shape_tensor()
      ret = self._reparameterize_sample(ret, event_shape=event_shape)

    return ret

  def _log_prob(self, x):
    x = self._pad_sample_dims(x)
    log_prob_x = self.components_distribution.log_prob(x)  # [S, B, k]
    log_mix_prob = tf.math.log_softmax(
        self.mixture_distribution.logits_parameter(), axis=-1)  # [B, k]
    return tf.reduce_logsumexp(log_prob_x + log_mix_prob, axis=-1)  # [S, B]

  def _mean(self):
    probs = self.mixture_distribution.probs_parameter()  # [B, k] or [k]
    component_means = self.components_distribution.mean()  # [B, k, E]
    event_ndims = self._event_ndims()

    # reshape probs to [B, k, [1]*e] or [k, [1]*e]
    probs = tf.reshape(probs, prefer_static.concat([
        prefer_static.shape(probs),
        prefer_static.ones([event_ndims], dtype=tf.int32)
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
    probs = self.mixture_distribution.probs_parameter()  # [B, k] or [k]
    component_means = self.components_distribution.mean()  # [B, k, E]
    component_vars = self.components_distribution.variance()  # [B, k, E]
    event_ndims = self._event_ndims()

    # reshape probs to [B, k, [1]*e] or [k, [1]*e]
    probs = tf.reshape(probs, prefer_static.concat([
        prefer_static.shape(probs),
        prefer_static.ones([event_ndims], dtype=tf.int32)
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

    probs = self.mixture_distribution.probs_parameter()  # [B, k] or [k]
    component_means = self.components_distribution.mean()  # [B, k, E]
    component_covars = self.components_distribution.covariance()  # [B, k, E, E]

    # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
    probs = probs[..., tf.newaxis, tf.newaxis]  # [B, k, 1, 1] or [k, 1, 1]
    mean_cond_var = tf.reduce_sum(
        probs * component_covars, axis=-3)  # [B, E, E]
    mean = tf.reduce_sum(
        probs[..., 0] * component_means, axis=-2, keepdims=True)  # [B, 1, E]
    var_cond_mean = tf.reduce_sum(
        probs * _outer_squared_difference(component_means, mean),
        axis=-3)  # [B, E, E]
    return mean_cond_var + var_cond_mean  # [B, E, E]

  def _pad_sample_dims(self, x, event_ndims=None):
    with tf.name_scope('pad_sample_dims'):
      if event_ndims is None:
        event_ndims = self._event_ndims()
      ndims = prefer_static.rank(x)
      shape = tf.convert_to_tensor(prefer_static.shape(x))
      d = ndims - event_ndims
      x = tf.reshape(
          x, shape=prefer_static.concat([shape[:d], [1], shape[d:]], axis=0))
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

    Arguments:
      x: Sample of mixture distribution
      event_shape: The event shape of this distribution

    Returns:
      Tensor with same value as x, but with reparameterization gradients
    """
    # Remove the existing gradients of x wrt parameters of the components.
    x = tf.stop_gradient(x)

    event_size = prefer_static.cast(
        prefer_static.reduce_prod(event_shape), dtype=tf.int32)
    x_2d_shape = [-1, event_size]  # [S*prod(B), prod(E)]

    # Perform distributional transform of x in [S, B, E] shape,
    # but have Jacobian of size [S*prod(B), prod(E), prod(E)].
    def reshaped_distributional_transform(x_2d):
      return tf.reshape(
          self._distributional_transform(
              tf.reshape(x_2d, tf.shape(x)), event_shape),
          x_2d_shape)

    # transform_2d: [S*prod(B), prod(E)]
    # jacobian: [S*prod(B), prod(E), prod(E)]
    x_2d = tf.reshape(x, x_2d_shape)
    transform_2d, jacobian = value_and_batch_jacobian(
        reshaped_distributional_transform, x_2d)

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
    surrogate_x = tf.reshape(surrogate_x_2d, tf.shape(x))

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

    Arguments:
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
      event_ndims = prefer_static.rank_from_shape(event_shape)
      x_padded = self._pad_sample_dims(
          x, event_ndims=event_ndims)  # [S, B, 1, E]
      log_prob_x = univariate_components.log_prob(x_padded)  # [S, B, k, E]
      cdf_x = univariate_components.cdf(x_padded)  # [S, B, k, E]

      # log prob_k (x_1, ..., x_i-1)
      event_size = prefer_static.cast(
          prefer_static.reduce_prod(event_shape), dtype=tf.int32)
      cumsum_log_prob_x = tf.reshape(
          tf.math.cumsum(
              # [S*prod(B)*k, prod(E)]
              tf.reshape(log_prob_x, [-1, event_size]),
              exclusive=True,
              axis=-1),
          tf.shape(log_prob_x))  # [S, B, k, E]

      event_ndims = prefer_static.rank_from_shape(event_shape)
      logits_mix_prob = self.mixture_distribution.logits_parameter()
      logits_mix_prob = tf.reshape(
          logits_mix_prob,  # [k] or [B, k]
          prefer_static.concat([
              prefer_static.shape(logits_mix_prob),
              prefer_static.ones([event_ndims], dtype=tf.int32),
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

    mdbs = self.mixture_distribution.batch_shape
    cdbs = tensorshape_util.with_rank_at_least(
        self.components_distribution.batch_shape, 1)[:-1]
    if (tensorshape_util.is_fully_defined(mdbs)
        and tensorshape_util.is_fully_defined(cdbs)):
      if tensorshape_util.rank(mdbs) != 0 and mdbs != cdbs:
        raise ValueError(
            '`mixture_distribution.batch_shape` (`{}`) is not '
            'compatible with `components_distribution.batch_shape` '
            '(`{}`)'.format(tensorshape_util.as_list(mdbs),
                            tensorshape_util.as_list(cdbs)))
    elif self.validate_args:
      if not tensorshape_util.is_fully_defined(mdbs):
        mixture_dist_param = tf.convert_to_tensor(mixture_dist_param)
        mdbs = tf.shape(mixture_dist_param)[:-1]
      if not tensorshape_util.is_fully_defined(cdbs):
        if component_bst is None:
          component_bst = self.components_distribution.batch_shape_tensor()
        cdbs = component_bst[:-1]
      assertions += [
          assert_util.assert_equal(
              distribution_utils.pick_vector(
                  tf.equal(tf.shape(mdbs)[0], 0), cdbs, mdbs),
              cdbs,
              message=(
                  '`mixture_distribution.batch_shape` is not '
                  'compatible with `components_distribution.batch_shape`'))
      ]

    return assertions


def _outer_squared_difference(x, y):
  """Convenience function analogous to tf.squared_difference."""
  z = x - y
  return z[..., tf.newaxis, :] * z[..., tf.newaxis]


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
    return tfp_custom_gradient.prevent_gradient(
        dy, message='Second derivative is not implemented.')

  return tf.identity(x), grad
