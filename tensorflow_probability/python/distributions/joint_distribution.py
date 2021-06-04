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
"""The `JointDistribution` base class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import itertools
import warnings
import numpy as np
import six

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import composition
from tensorflow_probability.python.bijectors import identity as identity_bijector
from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import log_prob_ratio
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.util.seed_stream import TENSOR_SEED_MSG_PREFIX

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'JointDistribution',
]


JAX_MODE = False


class ValueWithTrace(collections.namedtuple(
    'ValueWithTrace',
    ['value', 'traced'])):
  """Represents an RV's realized value, and related quantity(s) to trace.

     Traced quantities may include the RV's distribution, log prob, etc. The
     realized value is not itself traced unless explicitly included in `traced`.

     During model execution, each call to `sample_and_trace_fn` returns a
     `ValueWithTrace` structure. The model uses the realized `value`, e.g., to
     define the distributions of downstream random variables, while the
     `traced` quantities are accumulated and returned to the caller (for JDs
     that use `vectorized_map`, they will be returned from inside the vmap, and
     so must be `Tensor`s or `BatchableCompositeTensor`s).

  Components:
    value: The realized value of this random variable. May be a Tensor or
      nested structure of Tensors.
    traced: Quantitiy(s) to be accumulated and
      returned from the execution of the probabilistic program. These
      may have any type (for example, they may be structures, contain
      `tfd.Distribution` instances, etc).
  """


def trace_distributions_and_values(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces both the distribution and sampled value."""
  if value is None:
    value = dist.sample(sample_shape, seed=seed)
  elif tf.nest.is_nested(dist.dtype) and any(
      v is None for v in tf.nest.flatten(value)):
    # TODO(siege): This is making an assumption that nested dtype => partial
    # value support, which is not necessarily reasonable.
    value = dist.sample(sample_shape, seed=seed, value=value)
  return ValueWithTrace(value=value, traced=(dist, value))


def trace_distributions_only(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces the sampled value."""
  ret = trace_distributions_and_values(dist, sample_shape, seed, value)
  return ret._replace(traced=ret.traced[0])


def trace_values_only(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces the sampled value."""
  ret = trace_distributions_and_values(dist, sample_shape, seed, value)
  return ret._replace(traced=ret.traced[1])


def trace_values_and_log_probs(dist, sample_shape, seed, value=None):
  """Draws a sample, and traces both the sampled value and its log density."""
  if value is None:
    value, lp = dist.experimental_sample_and_log_prob(sample_shape, seed=seed)
  elif tf.nest.is_nested(dist.dtype) and any(
      v is None for v in tf.nest.flatten(value)):
    # TODO(siege): This is making an assumption that nested dtype => partial
    # value support, which is not necessarily reasonable.
    value, lp = dist.experimental_sample_and_log_prob(
        sample_shape, seed=seed, value=value)
  else:
    lp = dist.log_prob(value)
  return ValueWithTrace(value=value, traced=(value, lp))


CALLING_CONVENTION_DESCRIPTION = """
The measure methods of `JointDistribution` (`log_prob`, `prob`, etc.)
can be called either by passing a single structure of tensors or by using
named args for each part of the joint distribution state. For example,

```python
jd = tfd.JointDistributionSequential([
    tfd.Normal(0., 1.),
    lambda z: tfd.Normal(z, 1.)
], validate_args=True)
jd.dtype
# ==> [tf.float32, tf.float32]
z, x = sample = jd.sample()
# The following calling styles are all permissable and produce the exactly
# the same output.
assert (jd.{method}(sample) ==
        jd.{method}(value=sample) ==
        jd.{method}(z, x) ==
        jd.{method}(z=z, x=x) ==
        jd.{method}(z, x=x))

# These calling possibilities also imply that one can also use `*`
# expansion, if `sample` is a sequence:
jd.{method}(*sample)
# and similarly, if `sample` is a map, one can use `**` expansion:
jd.{method}(**sample)
```

`JointDistribution` component distributions names are resolved via
`jd._flat_resolve_names()`, which is implemented by each `JointDistribution`
subclass (see subclass documentation for details). Generally, for components
where a name was provided---
either explicitly as the `name` argument to a distribution or as a key in a
dict-valued JointDistribution, or implicitly, e.g., by the argument name of
a `JointDistributionSequential` distribution-making function---the provided
name will be used. Otherwise the component will receive a dummy name; these
may change without warning and should not be relied upon.

Note: not all `JointDistribution` subclasses support all calling styles;
for example, `JointDistributionNamed` does not support positional arguments
(aka "unnamed arguments") unless the provided model specifies an ordering of
variables (i.e., is an `collections.OrderedDict` or `collections.namedtuple`
rather than a plain `dict`).

Note: care is taken to resolve any potential ambiguity---this is generally
possible by inspecting the structure of the provided argument and "aligning"
it to the joint distribution output structure (defined by `jd.dtype`). For
example,

```python
trivial_jd = tfd.JointDistributionSequential([tfd.Exponential(1.)])
trivial_jd.dtype  # => [tf.float32]
trivial_jd.{method}([4.])
# ==> Tensor with shape `[]`.
{method_abbr} = trivial_jd.{method}(4.)
# ==> Tensor with shape `[]`.
```

Notice that in the first call, `[4.]` is interpreted as a list of one
scalar while in the second call the input is a scalar. Hence both inputs
result in identical scalar outputs. If we wanted to pass an explicit
vector to the `Exponential` component---creating a vector-shaped batch
of `{method}`s---we could instead write
`trivial_jd.{method}(np.array([4]))`.

Args:
  *args: Positional arguments: a `value` structure or component values
    (see above).
  **kwargs: Keyword arguments: a `value` structure or component values
    (see above). May also include `name`, specifying a Python string name
    for ops generated by this method.
"""


# Avoids name collision with measure function (`log_prob`, `prob`, etc.) args.
FORBIDDEN_COMPONENT_NAMES = ('value', 'name')


@six.add_metaclass(abc.ABCMeta)
class JointDistribution(distribution_lib.Distribution):
  """Joint distribution over one or more component distributions.

  This distribution enables both sampling and joint probability computation from
  a single model specification.

  A joint distribution is a collection of possibly interdependent distributions.

  **Note**: unlike other non-`JointDistribution` distributions in
  `tfp.distributions`, `JointDistribution.sample` (and subclasses) return a
  structure of  `Tensor`s rather than a `Tensor`.  A structure can be a `list`,
  `tuple`, `dict`, `collections.namedtuple`, etc. Accordingly
  `joint.batch_shape` returns a structure of `TensorShape`s for each of the
  distributions' batch shapes and `joint.batch_shape_tensor()` returns a
  structure of `Tensor`s for each of the distributions' event shapes. (Same with
  `event_shape` analogues.)

  #### Subclass Requirements

  Subclasses implement:

  - `_model_coroutine`: A generator that yields a sequence of
    `tfd.Distribution`-like instances.

  - `_model_flatten`: takes a structured input and returns a sequence. The
    sequence order must match the order distributions are yielded from
    `_model_coroutine`.

  - `_model_unflatten`: takes a sequence and returns a structure matching the
    semantics of the `JointDistribution` subclass.

  Subclasses initialize:

  - `_single_sample_distributions`: an empty dictionary, which will hold a
      mapping from graph id to a prototypical list of component distributions
      sampled from the model.
  """

  class Root(collections.namedtuple('Root', ['distribution'])):
    """Wrapper for coroutine distributions which lack distribution parents."""
    __slots__ = ()

  # TODO(b/166658748): Once the bug is resolved, we should be able to eliminate
  #   this workaround that disables sanitize_seed for JD*AB.
  _stateful_to_stateless = True

  @property
  def _require_root(self):
    return True

  def _get_single_sample_distributions(self, candidate_dists=None):
    """Returns a list of dists from a single sample of the model."""

    # If we have cached distributions with Eager tensors, return those.
    ds = self._single_sample_distributions.get(-1, None)
    if ds is not None and all([d is not None for d in ds]):
      return ds

    # Otherwise, retrieve or build distributions for the current graph.
    graph_id = -1 if tf.executing_eagerly() else id(tf.constant(True).graph)
    ds = self._single_sample_distributions.get(graph_id, None)
    if ds is None or any([d is None for d in ds]):
      if candidate_dists is not None:
        ds = candidate_dists
      else:
        ds = self._execute_model(
            seed=samplers.zeros_seed(),  # Constant seed for CSE.
            sample_and_trace_fn=trace_distributions_only)
      self._single_sample_distributions[graph_id] = ds
    return ds

  # Override `tf.Module`'s `_flatten` method to ensure that distributions are
  # instantiated, so that accessing `.variables` or `.trainable_variables` gives
  # consistent results.
  def _flatten(self, *args, **kwargs):
    self._get_single_sample_distributions()
    return super(JointDistribution, self)._flatten(*args, **kwargs)

  @abc.abstractmethod
  def _model_unflatten(self, xs):
    raise NotImplementedError()

  @abc.abstractmethod
  def _model_flatten(self, xs):
    raise NotImplementedError()

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `Distribution`."""
    return self._model_unflatten([
        d.dtype for d in self._get_single_sample_distributions()])

  @property
  def reparameterization_type(self):
    """Describes how samples from the distribution are reparameterized.

    Currently this is one of the static instances
    `tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

    Returns:
      reparameterization_type: `ReparameterizationType` of each distribution in
        `model`.
    """
    return self._model_unflatten([
        d.reparameterization_type
        for d in self._get_single_sample_distributions()])

  @property
  def experimental_shard_axis_names(self):
    """Indicates whether part distributions have active shard axis names."""
    return self._model_unflatten([
        d.experimental_shard_axis_names
        for d in self._get_single_sample_distributions()])

  @property
  def use_vectorized_map(self):
    return False

  @property
  def batch_shape(self):
    """Shape of a single sample from a single event index as a `TensorShape`.

    May be partially defined or unknown.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Returns:
      batch_shape: `tuple` of `TensorShape`s representing the `batch_shape` for
        each distribution in `model`.
    """
    return self._model_unflatten([
        d.batch_shape for d in self._get_single_sample_distributions()])

  def batch_shape_tensor(self, sample_shape=(), name='batch_shape_tensor'):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      sample_shape: The sample shape under which to evaluate the joint
        distribution. Sample shape at root (toplevel) nodes may affect the batch
        or event shapes of child nodes.
      name: name to give to the op

    Returns:
      batch_shape: `Tensor` representing batch shape of each distribution in
        `model`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          self._map_attr_over_dists(
              'batch_shape_tensor',
              dists=(self.sample_distributions(sample_shape)
                     if sample_shape else None)))

  @property
  def event_shape(self):
    """Shape of a single sample from a single batch as a `TensorShape`.

    May be partially defined or unknown.

    Returns:
      event_shape: `tuple` of `TensorShape`s representing the `event_shape` for
        each distribution in `model`.
    """
    # Caching will not leak graph Tensors since this is a static attribute.
    if not hasattr(self, '_cached_event_shape'):
      self._cached_event_shape = [
          d.event_shape
          for d in self._get_single_sample_distributions()]
    # Unflattening *after* retrieving from cache prevents tf.Module from
    # wrapping the returned value.
    return self._model_unflatten(self._cached_event_shape)

  def event_shape_tensor(self, sample_shape=(), name='event_shape_tensor'):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      sample_shape: The sample shape under which to evaluate the joint
        distribution. Sample shape at root (toplevel) nodes may affect the batch
        or event shapes of child nodes.
      name: name to give to the op
    Returns:
      event_shape: `tuple` of `Tensor`s representing the `event_shape` for each
        distribution in `model`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          self._map_attr_over_dists(
              'event_shape_tensor',
              dists=(self.sample_distributions(sample_shape)
                     if sample_shape else None)))

  def sample_distributions(self, sample_shape=(), seed=None, value=None,
                           name='sample_distributions', **kwargs):
    """Generate samples and the (random) distributions.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for generating random numbers.
      value: `list` of `Tensor`s in `distribution_fn` order to use to
        parameterize other ("downstream") distribution makers.
        Default value: `None` (i.e., draw a sample from each distribution).
      name: name prepended to ops created by this function.
        Default value: `"sample_distributions"`.
      **kwargs: This is an alternative to passing a `value`, and achieves the
        same effect. Named arguments will be used to parameterize other
        dependent ("downstream") distribution-making functions. If a `value`
        argument is also provided, raises a ValueError.

    Returns:
      distributions: a `tuple` of `Distribution` instances for each of
        `distribution_fn`.
      samples: a `tuple` of `Tensor`s with prepended dimensions `sample_shape`
        for each of `distribution_fn`.
    """
    with self._name_and_control_scope(name):
      value = self._resolve_value(value=value,
                                  allow_partially_specified=True,
                                  **kwargs)
      might_have_batch_dims = (
          distribution_util.shape_may_be_nontrivial(sample_shape)
          or value is not None)
      if self.use_vectorized_map and might_have_batch_dims:
        raise NotImplementedError('`sample_distributions` with nontrivial '
                                  'sample shape is not yet supported '
                                  'for autovectorized JointDistributions.')

      ds, xs = self._call_flat_sample_distributions(
          sample_shape, seed=seed, value=value)
      if not might_have_batch_dims:
        # This is a single sample with no pinned values; this call will cache
        # the distributions if they are not already cached.
        self._get_single_sample_distributions(candidate_dists=ds)

      return self._model_unflatten(ds), self._model_unflatten(xs)

  def _call_flat_sample_distributions(self,
                                      sample_shape=(),
                                      seed=None,
                                      value=None):
    return zip(*self._call_execute_model(
        sample_shape,
        seed=seed,
        value=value,
        sample_and_trace_fn=trace_distributions_and_values))

  def log_prob_parts(self, value, name='log_prob_parts'):
    """Log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `log_prob_parts` and to parameterize other ("downstream")
        distributions.
      name: name prepended to ops created by this function.
        Default value: `"log_prob_parts"`.

    Returns:
      log_prob_parts: a `tuple` of `Tensor`s representing the `log_prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      xs = self._map_measure_over_dists('log_prob', value)
      return self._model_unflatten(
          maybe_check_wont_broadcast(xs, self.validate_args))

  def prob_parts(self, value, name='prob_parts'):
    """Probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `prob_parts` and to parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"prob_parts"`.

    Returns:
      prob_parts: a `tuple` of `Tensor`s representing the `prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      xs = self._map_measure_over_dists('prob', value)
      return self._model_unflatten(
          maybe_check_wont_broadcast(xs, self.validate_args))

  def unnormalized_log_prob_parts(
      self, value, name='unnormalized_log_prob_parts'):
    """Unnormalized log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which
        we compute the `unnormalized_log_prob_parts` and to
        parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"unnormalized_log_prob_parts"`.

    Returns:
      unnormalized_log_prob_parts: a `tuple` of `Tensor`s representing
        the `unnormalized_log_prob` for each `distribution_fn`
        evaluated at each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      xs = self._map_measure_over_dists('unnormalized_log_prob', value)
      return self._model_unflatten(
          maybe_check_wont_broadcast(xs, self.validate_args))

  def unnormalized_prob_parts(self, value, name='unnormalized_prob_parts'):
    """Unnormalized probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which
        we compute the `unnormalized_prob_parts` and to parameterize
        other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"unnormalized_prob_parts"`.

    Returns:
      unnormalized_prob_parts: a `tuple` of `Tensor`s representing the
        `unnormalized_prob` for each `distribution_fn` evaluated at
        each corresponding `value`.
    """
    with self._name_and_control_scope(name):
      xs = self._map_measure_over_dists('unnormalized_prob', value)
      return self._model_unflatten(
          maybe_check_wont_broadcast(xs, self.validate_args))

  def is_scalar_event(self, name='is_scalar_event'):
    """Indicates that `event_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_event: `bool` scalar `Tensor` for each distribution in `model`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          [self._is_scalar_helper(shape, shape_tensor)  # pylint: disable=g-complex-comprehension
           for (shape, shape_tensor) in zip(
               self._model_flatten(self.event_shape),
               self._model_flatten(self.event_shape_tensor()))])

  def is_scalar_batch(self, name='is_scalar_batch'):
    """Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor` for each distribution in `model`.
    """
    with self._name_and_control_scope(name):
      return self._model_unflatten(
          self._map_attr_over_dists('is_scalar_batch'))

  def _log_prob(self, value):
    xs = self._map_measure_over_dists('log_prob', value)
    return sum(maybe_check_wont_broadcast(xs, self.validate_args))

  def _unnormalized_log_prob(self, value):
    xs = self._map_measure_over_dists('unnormalized_log_prob', value)
    return sum(maybe_check_wont_broadcast(xs, self.validate_args))

  @distribution_util.AppendDocstring(kwargs_dict={
      'value': ('`Tensor`s structured like `type(model)` used to parameterize '
                'other dependent ("downstream") distribution-making functions. '
                'Using `None` for any element will trigger a sample from the '
                'corresponding distribution. Default value: `None` '
                '(i.e., draw a sample from each distribution).')})
  def _sample_n(self, sample_shape, seed, value=None):
    might_have_batch_dims = (
        distribution_util.shape_may_be_nontrivial(sample_shape)
        or value is not None)
    if might_have_batch_dims:
      xs = self._call_execute_model(
          sample_shape,
          seed=seed,
          value=value,
          sample_and_trace_fn=trace_values_only)
    else:
      ds, xs = zip(*self._call_execute_model(
          sample_shape,
          seed=seed,
          value=value,
          sample_and_trace_fn=trace_distributions_and_values))
      # This is a single sample with no pinned values; this call will cache
      # the distributions if they are not already cached.
      self._get_single_sample_distributions(candidate_dists=ds)

    return self._model_unflatten(xs)

  # TODO(b/189122177): Implement _sample_and_log_prob for distributed JDs.
  def _sample_and_log_prob(self, sample_shape, seed, value=None, **kwargs):
    xs, lps = zip(
        *self._call_execute_model(
            sample_shape,
            seed=seed,
            value=self._resolve_value(value=value,
                                      allow_partially_specified=True,
                                      **kwargs),
            sample_and_trace_fn=trace_values_and_log_probs))
    return (self._model_unflatten(xs),
            sum(maybe_check_wont_broadcast(lps, self.validate_args)))

  def _map_measure_over_dists(self, attr, value):
    if any(x is None for x in tf.nest.flatten(value)):
      raise ValueError('No `value` part can be `None`; saw: {}.'.format(value))

    if not callable(attr):
      attr_name = attr
      attr = lambda d, x: getattr(d, attr_name)(x)

    return self._call_execute_model(
        sample_shape=(),
        value=value,
        seed=samplers.zeros_seed(),
        sample_and_trace_fn=(
            lambda dist, value, **_: ValueWithTrace(value=value,  # pylint: disable=g-long-lambda
                                                    traced=attr(dist, value))))

  def _map_attr_over_dists(self, attr, dists=None):
    dists = (self._get_single_sample_distributions()
             if dists is None else dists)
    return (getattr(d, attr)() for d in dists)

  def _resolve_value(self, *args, allow_partially_specified=False, **kwargs):
    """Resolves a `value` structure from user-passed arguments."""
    value = kwargs.pop('value', None)
    if not (args or kwargs):
      # Fast path when `value` is the only kwarg. The case where `value` is
      # passed as a positional arg is handled by `_resolve_value_from_args`
      # below.
      return _sanitize_value(self, value)
    elif value is not None:
      raise ValueError('Supplied both `value` and keyword '
                       'arguments to parameterize sampling. Supplied keyword '
                       'arguments were: {}. '.format(
                           ', '.join(map(str, kwargs))))

    names = self._flat_resolve_names()
    if allow_partially_specified:
      kwargs.update({k: kwargs.get(k, None) for k in names})  # In place update.
    value, unmatched_kwargs = _resolve_value_from_args(
        args,
        kwargs,
        dtype=self.dtype,
        flat_names=names,
        model_flatten_fn=self._model_flatten,
        model_unflatten_fn=self._model_unflatten)
    if unmatched_kwargs:
      join = lambda args: ', '.join(str(j) for j in args)
      kwarg_names = join(k for k, v in kwargs.items() if v is not None)
      dist_name_str = join(names)
      unmatched_str = join(unmatched_kwargs)
      raise ValueError(
          'Found unexpected keyword arguments. Distribution names '
          'are\n{}\nbut received\n{}\nThese names were '
          'invalid:\n{}'.format(dist_name_str, kwarg_names, unmatched_str))
    return _sanitize_value(self, value)

  def _call_execute_model(self,
                          sample_shape=(),
                          seed=None,
                          value=None,
                          sample_and_trace_fn=trace_distributions_and_values):
    return self._execute_model(
        sample_shape,
        seed=seed,
        value=value if value is None else self._model_flatten(value),
        sample_and_trace_fn=sample_and_trace_fn)

  # Override the base method to capture *args and **kwargs, so we can
  # implement more flexible custom calling semantics.
  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='log_prob', method_abbr='lp'))
  def log_prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Log probability density/mass function.

    ${calling_convention_description}

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    name = kwargs.pop('name', 'log_prob')
    return self._call_log_prob(self._resolve_value(*args, **kwargs), name=name)

  # Override the base method to capture *args and **kwargs, so we can
  # implement more flexible custom calling semantics.
  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='prob', method_abbr='prob'))
  def prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Probability density/mass function.

    ${calling_convention_description}

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    name = kwargs.pop('name', 'prob')
    return self._call_prob(self._resolve_value(*args, **kwargs), name=name)

  # Override the base method to capture *args and **kwargs, so we can
  # implement more flexible custom calling semantics.
  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='unnormalized_log_prob', method_abbr='lp'))
  def unnormalized_log_prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Unnormalized log probability density/mass function.

    ${calling_convention_description}

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    name = kwargs.pop('name', 'unnormalized_log_prob')
    return self._call_unnormalized_log_prob(
        self._resolve_value(*args, **kwargs), name=name)

  def _flat_resolve_names(self, dummy_name='var'):
    """Resolves a name for each random variable in the model."""
    names = []
    names_used = set()
    for dummy_idx, d in enumerate(self._get_single_sample_distributions()):
      name = get_explicit_name_for_component(d)
      if name is None:
        name = '{}{}'.format(dummy_name, dummy_idx)
      if name in names_used:
        raise ValueError('Duplicated distribution name: {}'.format(name))
      else:
        names_used.add(name)
      names.append(name)
    return names

  # We need to bypass base Distribution reshaping logic, so we
  # tactically implement the `_call_sample_n` redirector.  We don't want to
  # override the public level because then tfp.layers can't take generic
  # `Distribution.sample` as argument for the `convert_to_tensor_fn` parameter.
  def _call_sample_n(self, sample_shape, seed, value=None, **kwargs):
    return self._sample_n(
        sample_shape,
        seed=seed() if callable(seed) else seed,
        value=self._resolve_value(value=value,
                                  allow_partially_specified=True,
                                  **kwargs))

  def _execute_model(self,
                     sample_shape=(),
                     seed=None,
                     value=None,
                     stop_index=None,
                     sample_and_trace_fn=trace_distributions_and_values):
    """Executes `model`, creating both samples and distributions."""
    values_out = []
    if samplers.is_stateful_seed(seed):
      seed_stream = SeedStream(seed, salt='JointDistribution')
      if not self._stateful_to_stateless:
        seed = None
    else:
      seed_stream = None  # We got a stateless seed for seed=.

    # TODO(b/166658748): Make _stateful_to_stateless always True (eliminate it).
    if self._stateful_to_stateless and (seed is not None or not JAX_MODE):
      seed = samplers.sanitize_seed(seed, salt='JointDistribution')
    gen = self._model_coroutine()
    index = 0
    d = next(gen)
    if self._require_root and not isinstance(d, self.Root):
      raise ValueError('First distribution yielded by coroutine must '
                       'be wrapped in `Root`.')
    try:
      while True:
        actual_distribution = d.distribution if isinstance(d, self.Root) else d
        # Ensure reproducibility even when xs are (partially) set. Always split.
        stateful_sample_seed = None if seed_stream is None else seed_stream()
        if seed is None:
          stateless_sample_seed = None
        else:
          stateless_sample_seed, seed = samplers.split_seed(seed)

        value_at_index = None
        if (value is not None and len(value) > index and
            value[index] is not None):
          value_at_index = _sanitize_value(actual_distribution, value[index])
        try:
          next_value, traced_values = sample_and_trace_fn(
              actual_distribution,
              sample_shape=sample_shape if isinstance(d, self.Root) else (),
              seed=(stateful_sample_seed if stateless_sample_seed is None
                    else stateless_sample_seed),
              value=value_at_index)
        except TypeError as e:
          if ('Expected int for argument' not in str(e) and
              TENSOR_SEED_MSG_PREFIX not in str(e)) or (
                  stateful_sample_seed is None):
            raise
          msg = (
              'Falling back to stateful sampling for distribution #{index} '
              '(0-based) of type `{dist_cls}` with component name '
              '{component_name} and `dist.name` "{dist_name}". Please '
              'update to use `tf.random.stateless_*` RNGs. This fallback may '
              'be removed after 20-Dec-2020. ({exc})')
          component_name = get_explicit_name_for_component(actual_distribution)
          if component_name is None:
            component_name = '[None specified]'
          else:
            component_name = '"{}"'.format(component_name)
          warnings.warn(msg.format(
              index=index,
              component_name=component_name,
              dist_name=actual_distribution.name,
              dist_cls=type(actual_distribution),
              exc=str(e)))
          next_value, traced_values = sample_and_trace_fn(
              actual_distribution,
              sample_shape=sample_shape if isinstance(d, self.Root) else (),
              seed=stateful_sample_seed,
              value=value_at_index)
        if self._validate_args:
          with tf.control_dependencies(
              itertools.chain.from_iterable(
                  self._assert_compatible_shape(index, sample_shape, value_part)
                  for value_part in tf.nest.flatten(next_value))):
            values_out.append(
                tf.nest.map_structure(
                    lambda x: tf.identity(x) if tf.is_tensor(x) else x,
                    traced_values))
        else:
          values_out.append(traced_values)

        index += 1
        if stop_index is not None and index == stop_index:
          break
        d = gen.send(next_value)
    except StopIteration:
      pass
    return values_out

  def _assert_compatible_shape(self, index, sample_shape, samples):
    requested_shape, _ = self._expand_sample_shape_to_vector(
        tf.convert_to_tensor(sample_shape, dtype=tf.int32),
        name='requested_shape')
    actual_shape = prefer_static.shape(samples)
    actual_rank = prefer_static.rank_from_shape(actual_shape)
    requested_rank = prefer_static.rank_from_shape(requested_shape)

    # We test for two properties we expect of yielded distributions:
    # (1) The rank of the tensor of generated samples must be at least
    #     as large as the rank requested.
    # (2) The requested shape must be a prefix of the shape of the
    #     generated tensor of samples.
    # We attempt to perform test (1) statically first.
    # We don't need to do this explicitly for test (2) because
    # `assert_equal` evaluates statically if it can.
    static_actual_rank = tf.get_static_value(actual_rank)
    static_requested_rank = tf.get_static_value(requested_rank)

    assertion_message = ('Samples yielded by distribution #{} are not '
                         'consistent with `sample_shape` passed to '
                         '`JointDistributionCoroutine` '
                         'distribution.'.format(index))

    # TODO Remove this static check (b/138738650)
    if (static_actual_rank is not None and
        static_requested_rank is not None):
      # We're able to statically check the rank
      if static_actual_rank < static_requested_rank:
        raise ValueError(assertion_message)
      else:
        control_dependencies = []
    else:
      # We're not able to statically check the rank
      control_dependencies = [
          assert_util.assert_greater_equal(
              actual_rank, requested_rank,
              message=assertion_message)
          ]

    with tf.control_dependencies(control_dependencies):
      trimmed_actual_shape = actual_shape[:requested_rank]

    control_dependencies = [
        assert_util.assert_equal(
            requested_shape, trimmed_actual_shape,
            message=assertion_message)
    ]

    return control_dependencies

  def _default_event_space_bijector(self, *args, **kwargs):
    if bool(args) or bool(kwargs):
      return _DefaultJointBijector(self.experimental_pin(*args, **kwargs))
    return _DefaultJointBijector(self)

  def experimental_pin(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Pins some parts, returning an unnormalized distribution object.

    The calling convention is much like other `JointDistribution` methods (e.g.
    `log_prob`), but with the difference that not all parts are required. In
    this respect, the behavior is similar to that of the `sample` function's
    `value` argument.

    ### Examples:

    ```
    # Given the following joint distribution:
    jd = tfd.JointDistributionSequential([
        tfd.Normal(0., 1., name='z'),
        tfd.Normal(0., 1., name='y'),
        lambda y, z: tfd.Normal(y + z, 1., name='x')
    ], validate_args=True)

    # The following calls are all permissible and produce
    # `JointDistributionPinned` objects behaving identically.
    PartialXY = collections.namedtuple('PartialXY', 'x,y')
    PartialX = collections.namedtuple('PartialX', 'x')
    assert (jd.experimental_pin(x=2.).pins ==
            jd.experimental_pin(x=2., z=None).pins ==
            jd.experimental_pin(dict(x=2.)).pins ==
            jd.experimental_pin(dict(x=2., y=None)).pins ==
            jd.experimental_pin(PartialXY(x=2., y=None)).pins ==
            jd.experimental_pin(PartialX(x=2.)).pins ==
            jd.experimental_pin(None, None, 2.).pins ==
            jd.experimental_pin([None, None, 2.]).pins)
    ```

    Args:
      *args: Positional arguments: a value structure or component values (see
        above).
      **kwargs: Keyword arguments: a value structure or component values (see
        above). May also include `name`, specifying a Python string name for ops
        generated by this method.

    Returns:
      pinned: a `tfp.experimental.distributions.JointDistributionPinned` with
        the given values pinned.
    """
    # Import deferred to avoid circular dependency
    # JD.experimental_pin <> JDPinned <> experimental.marginalize.JDCoroutine.
    from tensorflow_probability.python.experimental.distributions import joint_distribution_pinned as jd_pinned  # pylint: disable=g-import-not-at-top
    return jd_pinned.JointDistributionPinned(self, *args, **kwargs)


def get_explicit_name_for_component(d):
  """Returns the explicitly-passed `name` of a Distribution, or None."""
  name = d.parameters.get('name', None)
  if name and d.__class__.__name__ in name:
    name = None

  if name and hasattr(d, '__init__'):
    spec = tf_inspect.getfullargspec(d.__init__)
    default_name = dict(
        zip(spec.args[len(spec.args) - len(spec.defaults or ()):],
            spec.defaults or ())
        ).get('name', None)
    if name == default_name:
      name = None

  if name in FORBIDDEN_COMPONENT_NAMES:
    raise ValueError('Distribution name "{}" is not allowed as a '
                     'JointDistribution component; please choose a different '
                     'name.'.format(name))
  return name


def _resolve_value_from_args(args,
                             kwargs,
                             dtype,
                             flat_names,
                             model_flatten_fn,
                             model_unflatten_fn):
  """Resolves a `value` structure matching `dtype` from a function call.

  This offers semantics equivalent to a Python callable `f(x1, x2, ..., xN)`,
  where `'x1', 'x2', ..., 'xN' = self._flat_resolve_names()` are the names of
  the model's component distributions. Arguments may be passed by position
  (`f(1., 2., 3.)`), by name (`f(x1=1., x2=2., x3=3.)`), or by a combination
  of approaches (`f(1., 2., x3=3.)`).

  Passing a `value` structure directly (as in `jd.log_prob(jd.sample())`) is
  supported by an optional `value` kwarg (`f(value=[1., 2., 3.])`), or by
  simply passing the value as the sole positional argument
  (`f([1., 2., 3.])`). For models having only a single component, a single
  positional argument that matches the structural type (e.g., a single Tensor,
  or a nested list or dict of Tensors) of that component is interpreted as
  specifying it; otherwise a single positional argument is interpreted as
  the overall `value`.

  Args:
    args: Positional arguments passed to the function being called.
    kwargs: Keyword arguments passed to the function being called.
    dtype: Nested structure of `dtype`s of model components.
    flat_names: Iterable of Python `str` names of model components.
    model_flatten_fn: Python `callable` that takes a structure and returns a
      list representing the flattened structure.
    model_unflatten_fn: Python `callable` that takes an iterable and returns a
      structure.
  Returns:
    value: A structure in which the observed arguments are arranged to match
      `dtype`.
    unmatched_kwargs: Python `dict` containing any keyword arguments that don't
      correspond to model components.
  Raises:
    ValueError: if the number of args passed doesn't match the number of
      model components, or if positional arguments are passed to a dict-valued
      distribution.
  """

  value = kwargs.pop('value', None)
  if value is not None:  # Respect 'value' as an explicit kwarg.
    return value, kwargs

  matched_kwargs = {k for k in flat_names if k in kwargs}
  unmatched_kwargs = {k: v for (k, v) in kwargs.items()
                      if k not in matched_kwargs}

  # If we have only a single positional arg, we need to disambiguate it by
  # examining the model structure.
  if len(args) == 1 and not matched_kwargs:
    if len(dtype) > 1:  # Model has multiple variables; arg must be a structure.
      return args[0], unmatched_kwargs
    # Otherwise the model has one variable. If its structure matches the arg,
    # interpret the arg as its value.
    first_component_dtype = model_flatten_fn(dtype)[0]
    try:
      # TODO(davmre): this assertion will falsely trigger if args[0] contains
      # nested lists that the user intends to be converted to Tensor. We should
      # try to relax it slightly (without creating false negatives).
      tf.nest.assert_same_structure(
          first_component_dtype, args[0], check_types=False)
      return model_unflatten_fn(args), unmatched_kwargs
    except (ValueError, TypeError):     # If RV doesn't match the arg, interpret
      return args[0], unmatched_kwargs  # the arg as a 'value' structure.

  num_components_specified = len(args) + len(kwargs) - len(unmatched_kwargs)
  if num_components_specified != len(flat_names):
    raise ValueError('Joint distribution expected values for {} components {}; '
                     'saw {} (from args {} and kwargs {}).'.format(
                         len(flat_names),
                         flat_names,
                         num_components_specified,
                         args,
                         kwargs))

  if args and (isinstance(dtype, dict) and not
               isinstance(dtype, collections.OrderedDict)):
    raise ValueError("Joint distribution with unordered variables can't "
                     "take positional args (saw {}).".format(args))

  value = model_unflatten_fn(kwargs[k] if k in kwargs else args[i]
                             for i, k in enumerate(flat_names))
  return value, unmatched_kwargs


def maybe_check_wont_broadcast(flat_xs, validate_args):
  """Verifies that `parts` don't broadcast."""
  flat_xs = tuple(flat_xs)  # So we can receive generators.
  if not validate_args:
    # Note: we don't try static validation because it is theoretically
    # possible that a user wants to take advantage of broadcasting.
    # Only when `validate_args` is `True` do we enforce the validation.
    return flat_xs
  msg = 'Broadcasting probably indicates an error in model specification.'
  s = tuple(prefer_static.shape(x) for x in flat_xs)
  if all(prefer_static.is_numpy(s_) for s_ in s):
    if not all(np.all(a == b) for a, b in zip(s[1:], s[:-1])):
      raise ValueError(msg)
    return flat_xs
  assertions = [assert_util.assert_equal(a, b, message=msg)
                for a, b in zip(s[1:], s[:-1])]
  with tf.control_dependencies(assertions):
    return tuple(tf.identity(x) for x in flat_xs)


# pylint: disable=protected-access
class _DefaultJointBijector(composition.Composition):
  """Minimally-viable event space bijector for `JointDistribution`."""

  def __init__(self, jd, parameters=None, bijector_fn=None):
    parameters = dict(locals()) if parameters is None else parameters

    with tf.name_scope('default_joint_bijector') as name:
      if bijector_fn is None:
        bijector_fn = lambda d: d.experimental_default_event_space_bijector()
      bijectors = tuple(bijector_fn(d)
                        for d in jd._get_single_sample_distributions())
      i_min_event_ndims = tf.nest.map_structure(
          prefer_static.size, jd.event_shape)
      f_min_event_ndims = jd._model_unflatten([
          b.inverse_event_ndims(nd) for b, nd in
          zip(bijectors, jd._model_flatten(i_min_event_ndims))])
      super(_DefaultJointBijector, self).__init__(
          bijectors=bijectors,
          forward_min_event_ndims=f_min_event_ndims,
          inverse_min_event_ndims=i_min_event_ndims,
          validate_args=jd.validate_args,
          parameters=parameters,
          name=name)
      self._jd = jd
      self._bijector_fn = bijector_fn

  def _conditioned_bijectors(self, samples, constrained=False):
    if samples is None:
      return self.bijectors

    bijectors = []
    gen = self._jd._model_coroutine()
    cond = None
    for rv in self._jd._model_flatten(samples):
      d = gen.send(cond)
      dist = d.distribution if type(d).__name__ == 'Root' else d
      bij = self._bijector_fn(dist)

      if bij is None:
        bij = identity_bijector.Identity()
      bijectors.append(bij)

      # If the RV is not yet constrained, transform it.
      cond = rv if constrained else bij.forward(rv)
    return bijectors

  @property
  def _parts_interact(self):
    # The bijector that operates on input part B may in general be a
    # function of input part A. This dependence is not visible to the
    # Composition base class, so we annotate it explicitly.
    return True

  def _walk_forward(self, step_fn, values, _jd_conditioning=None):  # pylint: disable=invalid-name
    bijectors = self._conditioned_bijectors(_jd_conditioning, constrained=False)
    return self._jd._model_unflatten(tuple(
        step_fn(bij, value) for bij, value in
        zip(bijectors, self._jd._model_flatten(values))))

  def _walk_inverse(self, step_fn, values, _jd_conditioning=None):  # pylint: disable=invalid-name
    bijectors = self._conditioned_bijectors(_jd_conditioning, constrained=True)
    return self._jd._model_unflatten(tuple(
        step_fn(bij, value) for bij, value
        in zip(bijectors, self._jd._model_flatten(values))))

  def _forward(self, x, **kwargs):
    return super(_DefaultJointBijector, self)._forward(
        x, _jd_conditioning=x, **kwargs)

  def _forward_log_det_jacobian(self, x, event_ndims, **kwargs):
    return super(_DefaultJointBijector, self)._forward_log_det_jacobian(
        x, event_ndims, _jd_conditioning=x, **kwargs)

  def _inverse(self, y, **kwargs):
    return super(_DefaultJointBijector, self)._inverse(
        y, _jd_conditioning=y, **kwargs)

  def _inverse_log_det_jacobian(self, y, event_ndims, **kwargs):
    return super(_DefaultJointBijector, self)._inverse_log_det_jacobian(
        y, event_ndims, _jd_conditioning=y, **kwargs)


def _sanitize_value(distribution, value):
  """Ensures `value` matches `distribution.dtype`, adding `None`s as needed."""
  if value is None:
    return value

  if not tf.nest.is_nested(distribution.dtype):
    return tf.convert_to_tensor(value, dtype_hint=distribution.dtype)

  if len(value) < len(distribution.dtype):
    # Fill in missing entries with `None`.
    if hasattr(distribution.dtype, 'keys'):
      value = {k: value.get(k, None) for k in distribution.dtype.keys()}
    else:  # dtype is a sequence.
      value = [value[i] if i < len(value) else None
               for i in range(len(distribution.dtype))]

  value = nest_util.cast_structure(value, distribution.dtype)
  jdlike_attrs = [
      '_get_single_sample_distributions',
      '_model_flatten',
      '_model_unflatten',
  ]
  if all(hasattr(distribution, attr) for attr in jdlike_attrs):
    flat_dists = distribution._get_single_sample_distributions()
    flat_value = distribution._model_flatten(value)
    flat_value = map(_sanitize_value, flat_dists, flat_value)
    return distribution._model_unflatten(flat_value)
  else:
    # A joint distribution that isn't tfd.JointDistribution-like; assume it has
    # some reasonable dtype semantics. We can't use this for
    # tfd.JointDistribution because we might have a None standing in for a
    # sub-tree (e.g. consider omitting a nested JD).
    return nest.map_structure_up_to(
        distribution.dtype,
        lambda x, d: x if x is None else tf.convert_to_tensor(x, dtype_hint=d),
        value,
        distribution.dtype,
    )


@log_prob_ratio.RegisterLogProbRatio(JointDistribution)
def _jd_log_prob_ratio(p, x, q, y, name=None):
  """Implements `log_prob_ratio` for tfd.JointDistribution*."""
  with tf.name_scope(name or 'jd_log_prob_ratio'):
    tf.nest.assert_same_structure(x, y)
    ps, _ = p.sample_distributions(value=x, seed=samplers.zeros_seed())
    qs, _ = q.sample_distributions(value=y, seed=samplers.zeros_seed())
    tf.nest.assert_same_structure(ps, qs)
    log_prob_ratio_parts = nest.map_structure_up_to(
        ps, log_prob_ratio.log_prob_ratio, ps, x, qs, y)
    return tf.add_n(tf.nest.flatten(log_prob_ratio_parts))
