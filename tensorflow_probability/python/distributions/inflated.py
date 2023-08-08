# Copyright 2022 The TensorFlow Probability Authors.
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
"""A mixture of a point-mass and another distribution."""

import inspect
import warnings

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import sigmoid as sigmoid_bijector
from tensorflow_probability.python.distributions import categorical
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import mixture
from tensorflow_probability.python.distributions import negative_binomial
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.util.deferred_tensor import DeferredTensor

__all__ = ['Inflated', 'inflated_factory', 'ZeroInflatedNegativeBinomial']


def _safe_value_for_distribution(dist):
  """Returns an x for which it is safe to differentiate dist.logprob(x)."""
  return dist.sample(seed=samplers.zeros_seed())


class _Inflated(mixture.Mixture):
  """A mixture of a point-mass and another distribution.

  Under the hood, this is implemented as a mixture.Mixture, and so
  supports all of the methods of that class.

  ### Examples:

  ```python
  zinb = Inflated(
           tfd.NegativeBinomial(5.0, probs=0.1), inflated_loc_prob=0.2)
  sample = zinb.sample(seed=jax.random.PRNGKey(0))
  ```
  """

  def __init__(self,
               distribution,
               inflated_loc_logits=None,
               inflated_loc_probs=None,
               inflated_loc=0.0,
               inflated_loc_atol=None,
               inflated_loc_rtol=None,
               validate_args=False,
               allow_nan_stats=True,
               name='Inflated'):
    """Initialize the Inflated distribution.

    Args:
      distribution: The tfp.Distribution to combine with a point mass at x. This
        code is intended to be used only with discrete distributions; when used
        with continuous distributions sampling will work but log_probs will be a
        sum of values with different units.
      inflated_loc_logits: A scalar or tensor containing the excess log-odds for
        the point mass at inflated_loc.  Only one of `inflated_loc_probs` or
        `inflated_loc_logits` should be passed in.
      inflated_loc_probs: A scalar or tensor containing the mixture weights for
        the point mass at inflated_loc.  Only one of `inflated_loc_probs` or
        `inflated_loc_logits` should be passed in.
      inflated_loc: A scalar or tensor containing the locations of the point
        mass component of the mixture.
      inflated_loc_atol:  Non-negative `Tensor` of same `dtype` as
        `inflated_loc` and broadcastable shape.  The absolute tolerance for
        comparing closeness to `inflated_loc`.  Default is `0`.
      inflated_loc_rtol:  Non-negative `Tensor` of same `dtype` as
        `inflated_loc` and broadcastable shape.  The relative tolerance for
        comparing closeness to `inflated_loc`.  Default is `0`.
      validate_args: If true, inconsistent batch or event sizes raise a runtime
        error.
      allow_nan_stats: If false, any undefined statistics for any batch memeber
        raise an exception.
      name: An optional name for the distribution.
    """
    parameters = dict(locals())
    if (inflated_loc_logits is None) == (inflated_loc_probs is None):
      raise ValueError('Must pass inflated_loc_logits or inflated_loc_probs, '
                       'but not both.')
    if not isinstance(distribution, distribution_lib.DiscreteDistributionMixin):
      warnings.warn('You have created an Inflated distribution with '
                    f'{distribution.name}, which is not discrete. ')
    with tf.name_scope(name) as name:
      dtype = dtype_util.common_dtype(
          [inflated_loc_logits, inflated_loc_probs, inflated_loc],
          dtype_hint=distribution.dtype)
      self._distribution = distribution
      self._inflated_loc_logits = tensor_util.convert_nonref_to_tensor(
          inflated_loc_logits, dtype=dtype, name='inflated_loc_logits')
      self._inflated_loc_probs = tensor_util.convert_nonref_to_tensor(
          inflated_loc_probs, dtype=dtype, name='inflated_loc_probs')
      self._inflated_loc = tensor_util.convert_nonref_to_tensor(
          inflated_loc, dtype=dtype, name='inflated_loc')
      self._inflated_loc_atol = tensor_util.convert_nonref_to_tensor(
          0 if inflated_loc_atol is None else inflated_loc_atol,
          dtype=dtype, name='inflated_loc_atol')
      self._inflated_loc_rtol = tensor_util.convert_nonref_to_tensor(
          0 if inflated_loc_rtol is None else inflated_loc_rtol,
          dtype=dtype, name='inflated_loc_rtol')

      if inflated_loc_probs is None:
        cat_logits = DeferredTensor(
            self._inflated_loc_logits,
            lambda logit: tf.stack([logit, -logit], axis=-1),
            dtype=self._inflated_loc_logits.dtype,
            shape=self._inflated_loc_logits.shape + (2,))
        self._categorical_dist = categorical.Categorical(
            logits=cat_logits,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats)
        probs_or_logits = self._inflated_loc_logits
      else:
        cat_probs = DeferredTensor(
            self._inflated_loc_probs,
            lambda probs: tf.stack(  # pylint: disable=g-long-lambda
                [probs, tf.ones_like(
                    probs, dtype=probs.dtype) - probs], axis=-1),
            dtype=self._inflated_loc_probs.dtype,
            shape=self._inflated_loc_probs.shape + (2,)
        )
        self._categorical_dist = categorical.Categorical(
            probs=cat_probs,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats)
        probs_or_logits = self._inflated_loc_probs

      self._deterministic = deterministic.Deterministic(
          DeferredTensor(
              probs_or_logits,
              # pylint: disable=g-long-lambda
              lambda _: tf.broadcast_to(self._inflated_loc,
                                        ps.shape(probs_or_logits)),
              shape=probs_or_logits.shape),
          atol=self._inflated_loc_atol,
          rtol=self._inflated_loc_rtol,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats)

      super(_Inflated, self).__init__(
          cat=self._categorical_dist,
          components=[
              self._deterministic,
              distribution
          ],
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
      self._parameters = parameters

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        distribution=parameter_properties.BatchedComponentProperties(),
        inflated_loc_logits=parameter_properties.ParameterProperties(),
        inflated_loc_probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False
        ),
        inflated_loc=parameter_properties.ParameterProperties())

  def _almost_inflated_loc(self, x):
    # pylint: disable=protected-access
    return tf.abs(x - self._inflated_loc) <= self._deterministic._slack(
        self._inflated_loc)
    # pylint: enable=protected-access

  def _log_prob(self, x):
    # We override the log_prob implementation from Mixture in the case
    # where we are inflating a continuous distribution, because we have
    # found that this "censored" version gives a good maximum likelihood
    # estimate of the continuous distribution's parameters but the
    # default implementation doesn't.  This follows the proposal in
    # https://arxiv.org/pdf/2010.09647.pdf for summing distributions of
    # different Hausdorff dimension.
    if isinstance(self._distribution,
                  distribution_lib.DiscreteDistributionMixin):
      return super(_Inflated, self)._log_prob(x)
    else:
      # Enable non-NaN gradients of the log_prob, even if the gradient of
      # the continuous distribution is NaN at _inflated_loc.  See
      # https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
      # for details.
      safe_x = tf.where(
          self._almost_inflated_loc(x),
          _safe_value_for_distribution(self._distribution),
          x)
      return tf.where(
          self._almost_inflated_loc(x),
          self._categorical_dist.log_prob(0),
          self._categorical_dist.log_prob(1) +
          self._distribution.log_prob(safe_x))

  @property
  def distribution(self):
    """The distribution used for the non-inflated part."""
    return self._distribution

  @property
  def inflated_loc_logits(self):
    """The log-odds for the point mass part of the distribution."""
    return self._inflated_loc_logits

  @property
  def inflated_loc_probs(self):
    """The mixture weight(s) for the point mass part of the distribution."""
    return self._inflated_loc_probs

  @property
  def inflated_loc(self):
    """The location to add probability mass to."""
    return self._inflated_loc


class Inflated(_Inflated, distribution_lib.AutoCompositeTensorDistribution):

  def __new__(cls, *args, **kwargs):
    """Maybe return a non-`CompositeTensor` `_Inflated`."""

    if cls is Inflated:
      if args:
        distribution = args[0]
      else:
        distribution = kwargs.get('distribution')

      if not auto_composite_tensor.is_composite_tensor(distribution):
        return _Inflated(*args, **kwargs)
    return super(Inflated, cls).__new__(cls)


Inflated.__doc__ = _Inflated.__doc__ + '\n' + (
    'If `distribution` is a `CompositeTensor`s, then the resulting `Inflated` '
    'instance is a `CompositeTensor` as well. Otherwise, a '
    'non-`CompositeTensor` `_Inflated` instance is created instead. '
    'Distribution subclasses that inherit from `Inflated` will also inherit '
    'from `CompositeTensor`.')


def inflated_factory(default_name, distribution_class, inflated_loc,
                     **more_kwargs):
  """Create Inflated subclasses for specific distributions and positions.

  Example usages:
    SpikeAndSlab = inflated_factory('SpikeAndSlab', tfd.Normal, 0.0)
    s_and_s = SpikeAndSlab(inflated_loc_probs=0.3, loc=5.0, scale=2.0)

    ZeroInflatedNegativeBinomial = inflated_factory(
        'ZeroInflatedNegativeBinomial', tfd.NegativeBinomial, 0.0)
    zinb = ZeroInflatedNegativeBinomial(inflated_loc_probs=0.2, probs=0.5,
                                        total_count=10.0)

  Args:
    default_name:  The name of the subclass, unless the user passes a
      name argument to init.
    distribution_class:  A tfd.Distribution class.
    inflated_loc:  The scalar position to inflate.
    **more_kwargs: Additional keyword arguments to pass to the
      distribution_class.

  Returns:
    A Inflated subclass that is the inflated version of distribution_class.
  """

  def my_init(self,
              inflated_loc_logits=None, inflated_loc_probs=None,
              name=default_name, **kwargs):
    parameters = dict(locals())
    if 'distribution' in kwargs:
      dist = kwargs['distribution']
    else:
      dist = distribution_class(**{**kwargs, **more_kwargs})
    Inflated.__init__(self, dist, inflated_loc_logits, inflated_loc_probs,
                      inflated_loc, name=name)
    # pylint: disable=protected-access
    self._parameters = {**parameters, **more_kwargs}
    # pylint: enable=protected-access

  def my_parameter_properties(unused_cls, dtype, num_classes=None):
    return dict(
        inflated_loc_logits=parameter_properties.ParameterProperties(),
        inflated_loc_probs=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=sigmoid_bijector.Sigmoid,
            is_preferred=False),
        **distribution_class.parameter_properties(dtype, num_classes))

  # In order to make auto_composite_tensor work, we need to do two things:
  # 1) Define property methods for each of the distribution_class's
  # constructor arguments, and
  # 2) Fool auto_composite_tensor into thinking those arguments are present
  # in the __init__ method for the Inflated subclass we are creating.  This
  # we do by inserting the "corrected" signature into auto_composite_tensor's
  # cache.

  methods_dict = {
      '__init__': my_init,
      '_parameter_properties': classmethod(my_parameter_properties)
  }

  distribution_signature = inspect.signature(distribution_class.__init__)
  for p in distribution_signature.parameters.keys():
    if p == 'name':
      continue
    def property_getter(self, param=p):
      # pylint: disable=protected-access
      return getattr(self._distribution, param)
    # pylint: enable=protected-access
    methods_dict[p] = property(property_getter)
  for k, v in more_kwargs.items():
    def another_property_getter(unused_self, value=v):
      return value
    methods_dict[k] = property(another_property_getter)

  newclass = type(default_name, (Inflated,), methods_dict)

  init_fn_signature = inspect.signature(my_init)
  new_parameters = tuple(list(distribution_signature.parameters.values()) +
                         [init_fn_signature.parameters['inflated_loc_logits'],
                          init_fn_signature.parameters['inflated_loc_probs']])
  new_signature = init_fn_signature.replace(parameters=new_parameters)
  # pylint: disable=protected-access
  auto_composite_tensor._sig_cache[newclass.__init__] = new_signature
  # pylint: enable=protected-access

  return newclass


ZeroInflatedNegativeBinomial = inflated_factory(
    'ZeroInflatedNegativeBinomial',
    negative_binomial.NegativeBinomial,
    0.0,
    require_integer_total_count=False)
