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
"""Base classes for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib
import functools
import inspect
import types
import decorator

import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions.internal import slicing
from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.internal import tensorshape_util
# Symbol import needed to avoid BUILD-dependency cycle
from tensorflow_probability.python.math.generic import log1mexp
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'Distribution',
]

_DISTRIBUTION_PUBLIC_METHOD_WRAPPERS = [
    'batch_shape',
    'batch_shape_tensor',
    'cdf',
    'covariance',
    'cross_entropy',
    'entropy',
    'event_shape',
    'event_shape_tensor',
    'kl_divergence',
    'log_cdf',
    'log_prob',
    'log_survival_function',
    'mean',
    'mode',
    'prob',
    'sample',
    'stddev',
    'survival_function',
    'variance',
]


_ALWAYS_COPY_PUBLIC_METHOD_WRAPPERS = ['kl_divergence', 'cross_entropy']


UNSET_VALUE = object()


JAX_MODE = False  # Overwritten by rewrite script.


@six.add_metaclass(abc.ABCMeta)
class _BaseDistribution(tf.Module):
  """Abstract base class needed for resolving subclass hierarchy."""
  pass


def _copy_fn(fn):
  """Create a deep copy of fn.

  Args:
    fn: a callable

  Returns:
    A `FunctionType`: a deep copy of fn.

  Raises:
    TypeError: if `fn` is not a callable.
  """
  if not callable(fn):
    raise TypeError('fn is not callable: {}'.format(fn))
  # The blessed way to copy a function. copy.deepcopy fails to create a
  # non-reference copy. Since:
  #   types.FunctionType == type(lambda: None),
  # and the docstring for the function type states:
  #
  #   function(code, globals[, name[, argdefs[, closure]]])
  #
  #   Create a function object from a code object and a dictionary.
  #   ...
  #
  # Here we can use this to create a new function with the old function's
  # code, globals, closure, etc.
  return types.FunctionType(
      code=fn.__code__, globals=fn.__globals__,
      name=fn.__name__, argdefs=fn.__defaults__,
      closure=fn.__closure__)


def _update_docstring(old_str, append_str):
  """Update old_str by inserting append_str just before the 'Args:' section."""
  old_str = old_str or ''
  old_str_lines = old_str.split('\n')

  # Step 0: Prepend spaces to all lines of append_str. This is
  # necessary for correct markdown generation.
  append_str = '\n'.join('    %s' % line for line in append_str.split('\n'))

  # Step 1: Find mention of 'Args':
  has_args_ix = [
      ix for ix, line in enumerate(old_str_lines)
      if line.strip().lower() == 'args:']
  if has_args_ix:
    final_args_ix = has_args_ix[-1]
    return ('\n'.join(old_str_lines[:final_args_ix])
            + '\n\n' + append_str + '\n\n'
            + '\n'.join(old_str_lines[final_args_ix:]))
  else:
    return old_str + '\n\n' + append_str


def _convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):
  """Converts the given `value` to a (structure of) `Tensor`.

  This function converts Python objects of various types to a (structure of)
  `Tensor` objects. It accepts `Tensor` objects, numpy arrays, Python lists, and
  Python scalars. For example:

  Args:
    value: An object whose structure matches that of `dtype ` and/or
      `dtype_hint` and for which each leaf has a registered `Tensor` conversion
      function.
    dtype: Optional (structure of) element type for the returned tensor. If
      missing, the type is inferred from the type of `value`.
    dtype_hint: Optional (structure of) element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a dtype in
      mind when converting to a tensor, so dtype_hint can be used as a soft
      preference.  If the conversion to `dtype_hint` is not possible, this
      argument has no effect.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    tensor: A (structure of) `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.
  """
  if (tf.nest.is_nested(dtype) or
      tf.nest.is_nested(dtype_hint)):
    if dtype is None:
      fn = lambda v, dh: tf.convert_to_tensor(v, dtype_hint=dh, name=name)
      return nest.map_structure_up_to(dtype_hint, fn, value, dtype_hint,
                                      # Allow list<->tuple conflation.
                                      check_types=False)
    elif dtype_hint is None:
      fn = lambda v, d: tf.convert_to_tensor(v, dtype=d, name=name)
      return nest.map_structure_up_to(dtype, fn, value, dtype,
                                      check_types=False)
    else:
      fn = lambda v, d, dh: tf.convert_to_tensor(  # pylint: disable=g-long-lambda
          v, dtype=d, dtype_hint=dh, name=name)
      return nest.map_structure_up_to(dtype, fn, value, dtype, dtype_hint,
                                      check_types=False, expand_composites=True)
  return tf.convert_to_tensor(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name)


def _remove_dict_keys_with_value(dict_, val):
  """Removes `dict` keys which have have `self` as value."""
  return {k: v for k, v in dict_.items() if v is not val}


def _set_sample_static_shape_for_tensor(x,
                                        event_shape,
                                        batch_shape,
                                        sample_shape):
  """Helper to `_set_sample_static_shape`; sets shape info for a `Tensor`."""
  sample_shape = tf.TensorShape(tf.get_static_value(sample_shape))

  ndims = tensorshape_util.rank(x.shape)
  sample_ndims = tensorshape_util.rank(sample_shape)
  batch_ndims = tensorshape_util.rank(batch_shape)
  event_ndims = tensorshape_util.rank(event_shape)

  # Infer rank(x).
  if (ndims is None and
      sample_ndims is not None and
      batch_ndims is not None and
      event_ndims is not None):
    ndims = sample_ndims + batch_ndims + event_ndims
    tensorshape_util.set_shape(x, [None] * ndims)

  # Infer sample shape.
  if ndims is not None and sample_ndims is not None:
    shape = tensorshape_util.concatenate(sample_shape,
                                         [None] * (ndims - sample_ndims))
    tensorshape_util.set_shape(x, shape)

  # Infer event shape.
  if ndims is not None and event_ndims is not None:
    shape = tf.TensorShape(
        [None]*(ndims - event_ndims)).concatenate(event_shape)
    tensorshape_util.set_shape(x, shape)

  # Infer batch shape.
  if batch_ndims is not None:
    if ndims is not None:
      if sample_ndims is None and event_ndims is not None:
        sample_ndims = ndims - batch_ndims - event_ndims
      elif event_ndims is None and sample_ndims is not None:
        event_ndims = ndims - batch_ndims - sample_ndims
    if sample_ndims is not None and event_ndims is not None:
      shape = tf.TensorShape([None]*sample_ndims).concatenate(
          batch_shape).concatenate([None]*event_ndims)
      tensorshape_util.set_shape(x, shape)
  return x


class _DistributionMeta(abc.ABCMeta):
  """Helper metaclass for tfp.Distribution."""

  def __new__(mcs, classname, baseclasses, attrs):
    """Control the creation of subclasses of the Distribution class.

    The main purpose of this method is to properly propagate docstrings
    from private Distribution methods, like `_log_prob`, into their
    public wrappers as inherited by the Distribution base class
    (e.g. `log_prob`).

    Args:
      classname: The name of the subclass being created.
      baseclasses: A tuple of parent classes.
      attrs: A dict mapping new attributes to their values.

    Returns:
      The class object.

    Raises:
      TypeError: If `Distribution` is not a subclass of `BaseDistribution`, or
        the new class is derived via multiple inheritance and the first
        parent class is not a subclass of `BaseDistribution`.
      AttributeError:  If `Distribution` does not implement e.g. `log_prob`.
      ValueError:  If a `Distribution` public method lacks a docstring.
    """
    if not baseclasses:  # Nothing to be done for Distribution
      raise TypeError('Expected non-empty baseclass. Does Distribution '
                      'not subclass _BaseDistribution?')
    which_base = [
        base for base in baseclasses
        if base == _BaseDistribution or issubclass(base, Distribution)]
    base = which_base[0] if which_base else None
    if base is None or base == _BaseDistribution:
      # Nothing to be done for Distribution or unrelated subclass.
      return super(_DistributionMeta, mcs).__new__(
          mcs, classname, baseclasses, attrs)
    if not issubclass(base, Distribution):
      raise TypeError('First parent class declared for {} must be '
                      'Distribution, but saw "{}"'.format(
                          classname, base.__name__))
    for attr in _DISTRIBUTION_PUBLIC_METHOD_WRAPPERS:
      if attr in attrs:
        # The method is being overridden, do not update its docstring.
        continue
      special_attr = '_{}'.format(attr)
      class_attr_value = attrs.get(attr, None)
      base_attr_value = getattr(base, attr, None)
      if not base_attr_value:
        raise AttributeError(
            'Internal error: expected base class "{}" to '
            'implement method "{}"'.format(base.__name__, attr))
      class_special_attr_value = attrs.get(special_attr, None)
      class_special_attr_docstring = (
          None if class_special_attr_value is None else
          tf_inspect.getdoc(class_special_attr_value))
      if (class_special_attr_docstring or
          attr in _ALWAYS_COPY_PUBLIC_METHOD_WRAPPERS):
        class_attr_value = _copy_fn(base_attr_value)
        attrs[attr] = class_attr_value

      if not class_special_attr_docstring:
        # No docstring to append.
        continue
      class_attr_docstring = tf_inspect.getdoc(base_attr_value)
      if class_attr_docstring is None:
        raise ValueError(
            'Expected base class fn to contain a docstring: {}.{}'.format(
                base.__name__, attr))
      class_attr_value.__doc__ = _update_docstring(
          class_attr_value.__doc__,
          'Additional documentation from `{}`:\n\n{}'.format(
              classname, class_special_attr_docstring))

    # Now we'll intercept the default __init__ if it exists.
    default_init = attrs.get('__init__', None)
    if default_init is None:
      # The class has no __init__ because its abstract. (And we won't add one.)
      return super(_DistributionMeta, mcs).__new__(
          mcs, classname, baseclasses, attrs)

    # pylint: disable=protected-access
    # For a comparison of different methods for wrapping functions, see:
    # https://hynek.me/articles/decorators/
    @decorator.decorator
    def wrapped_init(wrapped, self_, *args, **kwargs):
      """A 'master `__init__`' which is always called."""
      # We can't use `wrapped` because it results in a self reference which
      # confounds `tf.function`.
      del wrapped
      # Note: if we ever want to have things set in `self` before `__init__` is
      # called, here is the place to do it.
      self_._parameters = None
      default_init(self_, *args, **kwargs)
      # Note: if we ever want to override things set in `self` by subclass
      # `__init__`, here is the place to do it.
      if self_._parameters is None:
        # We prefer subclasses will set `parameters = dict(locals())` because
        # this has nearly zero overhead. However, failing to do this, we will
        # resolve the input arguments dynamically and only when needed.
        dummy_self = tuple()
        self_._parameters = self_._no_dependency(lambda: (  # pylint: disable=g-long-lambda
            _remove_dict_keys_with_value(
                inspect.getcallargs(default_init, dummy_self, *args, **kwargs),
                dummy_self)))
      elif hasattr(self_._parameters, 'pop'):
        self_._parameters = self_._no_dependency(
            _remove_dict_keys_with_value(self_._parameters, self_))
    # pylint: enable=protected-access

    attrs['__init__'] = wrapped_init(default_init)  # pylint: disable=no-value-for-parameter,assignment-from-no-return
    return super(_DistributionMeta, mcs).__new__(
        mcs, classname, baseclasses, attrs)


@six.add_metaclass(_DistributionMeta)
class Distribution(_BaseDistribution):
  """A generic probability distribution base class.

  `Distribution` is a base class for constructing and organizing properties
  (e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian).

  #### Subclassing

  Subclasses are expected to implement a leading-underscore version of the
  same-named function. The argument signature should be identical except for
  the omission of `name='...'`. For example, to enable `log_prob(value,
  name='log_prob')` a subclass should implement `_log_prob(value)`.

  Subclasses can append to public-level docstrings by providing
  docstrings for their method specializations. For example:

  ```python
  @distribution_util.AppendDocstring('Some other details.')
  def _log_prob(self, value):
    ...
  ```

  would add the string "Some other details." to the `log_prob` function
  docstring. This is implemented as a simple decorator to avoid python
  linter complaining about missing Args/Returns/Raises sections in the
  partial docstrings.

  #### Broadcasting, batching, and shapes

  All distributions support batches of independent distributions of that type.
  The batch shape is determined by broadcasting together the parameters.

  The shape of arguments to `__init__`, `cdf`, `log_cdf`, `prob`, and
  `log_prob` reflect this broadcasting, as does the return value of `sample`.

  `sample_n_shape = [n] + batch_shape + event_shape`, where `sample_n_shape` is
  the shape of the `Tensor` returned from `sample(n)`, `n` is the number of
  samples, `batch_shape` defines how many independent distributions there are,
  and `event_shape` defines the shape of samples from each of those independent
  distributions. Samples are independent along the `batch_shape` dimensions, but
  not necessarily so along the `event_shape` dimensions (depending on the
  particulars of the underlying distribution).

  Using the `Uniform` distribution as an example:

  ```python
  minval = 3.0
  maxval = [[4.0, 6.0],
            [10.0, 12.0]]

  # Broadcasting:
  # This instance represents 4 Uniform distributions. Each has a lower bound at
  # 3.0 as the `minval` parameter was broadcasted to match `maxval`'s shape.
  u = Uniform(minval, maxval)

  # `event_shape` is `TensorShape([])`.
  event_shape = u.event_shape
  # `event_shape_t` is a `Tensor` which will evaluate to [].
  event_shape_t = u.event_shape_tensor()

  # Sampling returns a sample per distribution. `samples` has shape
  # [5, 2, 2], which is [n] + batch_shape + event_shape, where n=5,
  # batch_shape=[2, 2], and event_shape=[].
  samples = u.sample(5)

  # The broadcasting holds across methods. Here we use `cdf` as an example. The
  # same holds for `log_cdf` and the likelihood functions.

  # `cum_prob` has shape [2, 2] as the `value` argument was broadcasted to the
  # shape of the `Uniform` instance.
  cum_prob_broadcast = u.cdf(4.0)

  # `cum_prob`'s shape is [2, 2], one per distribution. No broadcasting
  # occurred.
  cum_prob_per_dist = u.cdf([[4.0, 5.0],
                             [6.0, 7.0]])

  # INVALID as the `value` argument is not broadcastable to the distribution's
  # shape.
  cum_prob_invalid = u.cdf([4.0, 5.0, 6.0])
  ```

  #### Shapes

  There are three important concepts associated with TensorFlow Distributions
  shapes:

  - Event shape describes the shape of a single draw from the distribution;
    it may be dependent across dimensions. For scalar distributions, the event
    shape is `[]`. For a 5-dimensional MultivariateNormal, the event shape is
    `[5]`.
  - Batch shape describes independent, not identically distributed draws, aka a
    "collection" or "bunch" of distributions.
  - Sample shape describes independent, identically distributed draws of batches
    from the distribution family.

  The event shape and the batch shape are properties of a Distribution object,
  whereas the sample shape is associated with a specific call to `sample` or
  `log_prob`.

  For detailed usage examples of TensorFlow Distributions shapes, see
  [this tutorial](
  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb)

  #### Parameter values leading to undefined statistics or distributions.

  Some distributions do not have well-defined statistics for all initialization
  parameter values. For example, the beta distribution is parameterized by
  positive real numbers `concentration1` and `concentration0`, and does not have
  well-defined mode if `concentration1 < 1` or `concentration0 < 1`.

  The user is given the option of raising an exception or returning `NaN`.

  ```python
  a = tf.exp(tf.matmul(logits, weights_a))
  b = tf.exp(tf.matmul(logits, weights_b))

  # Will raise exception if ANY batch member has a < 1 or b < 1.
  dist = distributions.beta(a, b, allow_nan_stats=False)
  mode = dist.mode().eval()

  # Will return NaN for batch members with either a < 1 or b < 1.
  dist = distributions.beta(a, b, allow_nan_stats=True)  # Default behavior
  mode = dist.mode().eval()
  ```

  In all cases, an exception is raised if *invalid* parameters are passed, e.g.

  ```python
  # Will raise an exception if any Op is run.
  negative_a = -1.0 * a  # beta distribution by definition has a > 0.
  dist = distributions.beta(negative_a, b, allow_nan_stats=True)
  dist.mean().eval()
  ```

  """

  def __init__(self,
               dtype,
               reparameterization_type,
               validate_args,
               allow_nan_stats,
               parameters=None,
               graph_parents=None,
               name=None):
    """Constructs the `Distribution`.

    **This is a private method for subclass use.**

    Args:
      dtype: The type of the event samples. `None` implies no type-enforcement.
      reparameterization_type: Instance of `ReparameterizationType`.
        If `tfd.FULLY_REPARAMETERIZED`, then samples from the distribution are
        fully reparameterized, and straight-through gradients are supported.
        If `tfd.NOT_REPARAMETERIZED`, then samples from the distribution are not
        fully reparameterized, and straight-through gradients are either
        partially unsupported or are not supported at all.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      parameters: Python `dict` of parameters used to instantiate this
        `Distribution`.
      graph_parents: Python `list` of graph prerequisites of this
        `Distribution`.
      name: Python `str` name prefixed to Ops created by this class. Default:
        subclass name.

    Raises:
      ValueError: if any member of graph_parents is `None` or not a `Tensor`.
    """
    if not name:
      name = type(self).__name__
      name = name_util.camel_to_lower_snake(name)
    name = name_util.get_name_scope_name(name)
    name = name_util.strip_invalid_chars(name)
    super(Distribution, self).__init__(name=name)
    self._name = name

    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
      if t is None or not tf.is_tensor(t):
        raise ValueError('Graph parent item %d is not a Tensor; %s.' % (i, t))
    self._dtype = dtype
    self._reparameterization_type = reparameterization_type
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    self._parameters = self._no_dependency(parameters)
    self._parameters_sanitized = False
    self._graph_parents = graph_parents
    self._initial_parameter_control_dependencies = tuple(
        d for d in self._parameter_control_dependencies(is_init=True)
        if d is not None)
    if self._initial_parameter_control_dependencies:
      self._initial_parameter_control_dependencies = (
          tf.group(*self._initial_parameter_control_dependencies),)

  @classmethod
  def param_shapes(cls, sample_shape, name='DistributionParamShapes'):
    """Shapes of parameters given the desired shape of a call to `sample()`.

    This is a class method that describes what key/value arguments are required
    to instantiate the given `Distribution` so that a particular shape is
    returned for that instance's call to `sample()`.

    Subclasses should override class method `_param_shapes`.

    Args:
      sample_shape: `Tensor` or python list/tuple. Desired shape of a call to
        `sample()`.
      name: name to prepend ops with.

    Returns:
      `dict` of parameter name to `Tensor` shapes.
    """
    with tf.name_scope(name):
      return cls._param_shapes(sample_shape)

  @classmethod
  def param_static_shapes(cls, sample_shape):
    """param_shapes with static (i.e. `TensorShape`) shapes.

    This is a class method that describes what key/value arguments are required
    to instantiate the given `Distribution` so that a particular shape is
    returned for that instance's call to `sample()`. Assumes that the sample's
    shape is known statically.

    Subclasses should override class method `_param_shapes` to return
    constant-valued tensors when constant values are fed.

    Args:
      sample_shape: `TensorShape` or python list/tuple. Desired shape of a call
        to `sample()`.

    Returns:
      `dict` of parameter name to `TensorShape`.

    Raises:
      ValueError: if `sample_shape` is a `TensorShape` and is not fully defined.
    """
    if isinstance(sample_shape, tf.TensorShape):
      if not tensorshape_util.is_fully_defined(sample_shape):
        raise ValueError('TensorShape sample_shape must be fully defined')
      sample_shape = tensorshape_util.as_list(sample_shape)

    params = cls.param_shapes(sample_shape)

    static_params = {}
    for name, shape in params.items():
      static_shape = tf.get_static_value(shape)
      if static_shape is None:
        raise ValueError(
            'sample_shape must be a fully-defined TensorShape or list/tuple')
      static_params[name] = tf.TensorShape(static_shape)

    return static_params

  @staticmethod
  def _param_shapes(sample_shape):
    raise NotImplementedError('_param_shapes not implemented')

  @property
  def name(self):
    """Name prepended to all ops created by this `Distribution`."""
    return self._name if hasattr(self, '_name') else None

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `Distribution`."""
    return self._dtype if hasattr(self, '_dtype') else None

  @property
  def parameters(self):
    """Dictionary of parameters used to instantiate this `Distribution`."""
    # Remove 'self', '__class__', or other special variables. These can appear
    # if the subclass used: `parameters = dict(locals())`.
    if (not hasattr(self, '_parameters_sanitized') or
        not self._parameters_sanitized):
      p = self._parameters() if callable(self._parameters) else self._parameters
      self._parameters = self._no_dependency({
          k: v for k, v in p.items()
          if not k.startswith('__') and v is not self})
      self._parameters_sanitized = True
    return dict(self._parameters)

  @classmethod
  def _params_event_ndims(cls):
    """Returns a dict mapping constructor argument names to per-event rank.

    Distributions may implement this method to provide support for slicing
    (`__getitem__`) on the batch axes.

    Examples: Normal has scalar parameters, so would return
    `{'loc': 0, 'scale': 0}`. On the other hand, MultivariateNormalTriL has
    vector loc and matrix scale, so returns `{'loc': 1, 'scale_tril': 2}`. When
    a distribution accepts multiple parameterizations, either all possible
    parameters may be specified by the dict, e.g. Bernoulli returns
    `{'logits': 0, 'probs': 0}`, or if convenient only the parameters relevant
    to this instance may be specified.

    Parameter dtypes are inferred from Tensor attributes on the distribution
    where available, e.g. `bernoulli.probs`, 'mvn.scale_tril', falling back with
    a warning to the dtype of the distribution.

    Returns:
      params_event_ndims: Per-event parameter ranks, a `str->int dict`.
    """
    raise NotImplementedError(
        '{} does not support batch slicing; must implement '
        '_params_event_ndims.'.format(cls))

  def __getitem__(self, slices):
    """Slices the batch axes of this distribution, returning a new instance.

    ```python
    b = tfd.Bernoulli(logits=tf.zeros([3, 5, 7, 9]))
    b.batch_shape  # => [3, 5, 7, 9]
    b2 = b[:, tf.newaxis, ..., -2:, 1::2]
    b2.batch_shape  # => [3, 1, 5, 2, 4]

    x = tf.random.normal([5, 3, 2, 2])
    cov = tf.matmul(x, x, transpose_b=True)
    chol = tf.cholesky(cov)
    loc = tf.random.normal([4, 1, 3, 1])
    mvn = tfd.MultivariateNormalTriL(loc, chol)
    mvn.batch_shape  # => [4, 5, 3]
    mvn.event_shape  # => [2]
    mvn2 = mvn[:, 3:, ..., ::-1, tf.newaxis]
    mvn2.batch_shape  # => [4, 2, 3, 1]
    mvn2.event_shape  # => [2]
    ```

    Args:
      slices: slices from the [] operator

    Returns:
      dist: A new `tfd.Distribution` instance with sliced parameters.
    """
    return slicing.batch_slice(self, self._params_event_ndims(), {}, slices)

  def __iter__(self):
    raise TypeError('{!r} object is not iterable'.format(type(self).__name__))

  @property
  def reparameterization_type(self):
    """Describes how samples from the distribution are reparameterized.

    Currently this is one of the static instances
    `tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

    Returns:
      An instance of `ReparameterizationType`.
    """
    return self._reparameterization_type

  @property
  def allow_nan_stats(self):
    """Python `bool` describing behavior when a stat is undefined.

    Stats return +/- infinity when it makes sense. E.g., the variance of a
    Cauchy distribution is infinity. However, sometimes the statistic is
    undefined, e.g., if a distribution's pdf does not achieve a maximum within
    the support of the distribution, the mode is undefined. If the mean is
    undefined, then by definition the variance is undefined. E.g. the mean for
    Student's T for df = 1 is undefined (no clear way to say it is either + or -
    infinity), so the variance = E[(X - mean)**2] is also undefined.

    Returns:
      allow_nan_stats: Python `bool`.
    """
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Python `bool` indicating possibly expensive checks are enabled."""
    return self._validate_args

  def copy(self, **override_parameters_kwargs):
    """Creates a deep copy of the distribution.

    Note: the copy distribution may continue to depend on the original
    initialization arguments.

    Args:
      **override_parameters_kwargs: String/value dictionary of initialization
        arguments to override with new values.

    Returns:
      distribution: A new instance of `type(self)` initialized from the union
        of self.parameters and override_parameters_kwargs, i.e.,
        `dict(self.parameters, **override_parameters_kwargs)`.
    """
    try:
      # We want track provenance from origin variables, so we use batch_slice
      # if this distribution supports slicing. See the comment on
      # PROVENANCE_ATTR in slicing.py
      return slicing.batch_slice(self, self._params_event_ndims(),
                                 override_parameters_kwargs, Ellipsis)
    except NotImplementedError:
      parameters = dict(self.parameters, **override_parameters_kwargs)
      d = type(self)(**parameters)
      # pylint: disable=protected-access
      d._parameters = parameters
      d._parameters_sanitized = True
      # pylint: enable=protected-access
      return d

  def _batch_shape_tensor(self):
    raise NotImplementedError(
        'batch_shape_tensor is not implemented: {}'.format(type(self).__name__))

  def batch_shape_tensor(self, name='batch_shape_tensor'):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
    with self._name_and_control_scope(name):
      # Joint distributions may have a structured `batch shape_tensor` or a
      # single `batch_shape_tensor` that applies to all components. (Simple
      # distributions always have a single `batch_shape_tensor`.) If the
      # distribution's `batch_shape` is an instance of `tf.TensorShape`, we
      # infer that `batch_shape_tensor` is not structured.
      shallow_structure = (None if isinstance(self.batch_shape, tf.TensorShape)
                           else self.dtype)
      if all([tensorshape_util.is_fully_defined(s)
              for s in nest.flatten_up_to(
                  shallow_structure, self.batch_shape, check_types=False)]):
        batch_shape = nest.map_structure_up_to(
            shallow_structure,
            tensorshape_util.as_list,
            self.batch_shape, check_types=False)
      else:
        batch_shape = self._batch_shape_tensor()
      return nest.map_structure_up_to(
          shallow_structure,
          lambda s: tf.identity(  # pylint: disable=g-long-lambda
              tf.convert_to_tensor(s, dtype=tf.int32), name='batch_shape'),
          batch_shape, check_types=False)

  def _batch_shape(self):
    return None

  @property
  def batch_shape(self):
    """Shape of a single sample from a single event index as a `TensorShape`.

    May be partially defined or unknown.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Returns:
      batch_shape: `TensorShape`, possibly unknown.
    """
    batch_shape = self._batch_shape()
    # See comment in `batch_shape_tensor()` on structured batch shapes. If
    # `_batch_shape()` is a `tf.TensorShape` instance or a flat list/tuple that
    # does not contain `tf.TensorShape`s, we infer that it is not structured.
    if (isinstance(batch_shape, tf.TensorShape)
        or all(len(path) == 1 and not isinstance(s, tf.TensorShape)
               for path, s in nest.flatten_with_tuple_paths(batch_shape))):
      return tf.TensorShape(batch_shape)
    return nest.map_structure_up_to(
        self.dtype, tf.TensorShape, batch_shape, check_types=False)

  def _event_shape_tensor(self):
    raise NotImplementedError(
        'event_shape_tensor is not implemented: {}'.format(type(self).__name__))

  def event_shape_tensor(self, name='event_shape_tensor'):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    """
    with self._name_and_control_scope(name):
      if all([tensorshape_util.is_fully_defined(s)
              for s in nest.flatten(self.event_shape)]):
        event_shape = nest.map_structure_up_to(
            self.dtype,
            tensorshape_util.as_list,
            self.event_shape, check_types=False)
      else:
        event_shape = self._event_shape_tensor()
      return nest.map_structure_up_to(
          self.dtype,
          lambda s: tf.identity(  # pylint: disable=g-long-lambda
              tf.convert_to_tensor(s, dtype=tf.int32), name='event_shape'),
          event_shape, check_types=False)

  def _event_shape(self):
    return None

  @property
  def event_shape(self):
    """Shape of a single sample from a single batch as a `TensorShape`.

    May be partially defined or unknown.

    Returns:
      event_shape: `TensorShape`, possibly unknown.
    """
    return nest.map_structure_up_to(
        self.dtype, tf.TensorShape, self._event_shape(), check_types=False)

  def is_scalar_event(self, name='is_scalar_event'):
    """Indicates that `event_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_event: `bool` scalar `Tensor`.
    """
    with self._name_and_control_scope(name):
      return tf.convert_to_tensor(
          self._is_scalar_helper(self.event_shape, self.event_shape_tensor),
          name='is_scalar_event')

  def is_scalar_batch(self, name='is_scalar_batch'):
    """Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor`.
    """
    with self._name_and_control_scope(name):
      return tf.convert_to_tensor(
          self._is_scalar_helper(self.batch_shape, self.batch_shape_tensor),
          name='is_scalar_batch')

  def _sample_n(self, n, seed=None, **kwargs):
    raise NotImplementedError('sample_n is not implemented: {}'.format(
        type(self).__name__))

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    """Wrapper around _sample_n."""
    with self._name_and_control_scope(name):
      if JAX_MODE and seed is None:
        raise ValueError('Must provide JAX PRNGKey as `dist.sample(seed=.)`')
      sample_shape = tf.cast(sample_shape, tf.int32, name='sample_shape')
      sample_shape, n = self._expand_sample_shape_to_vector(
          sample_shape, 'sample_shape')
      samples = self._sample_n(
          n, seed=seed() if callable(seed) else seed, **kwargs)
      batch_event_shape = tf.shape(samples)[1:]
      final_shape = tf.concat([sample_shape, batch_event_shape], 0)
      samples = tf.reshape(samples, final_shape)
      samples = self._set_sample_static_shape(samples, sample_shape)
      return samples

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer or `tfp.util.SeedStream` instance, for seeding PRNG.
      name: name to give to the op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
    return self._call_sample_n(sample_shape, seed, name, **kwargs)

  def _call_log_prob(self, value, name, **kwargs):
    """Wrapper around _log_prob."""
    value = _convert_to_tensor(value, name='value', dtype_hint=self.dtype)
    with self._name_and_control_scope(name, value, kwargs):
      if hasattr(self, '_log_prob'):
        return self._log_prob(value, **kwargs)
      if hasattr(self, '_prob'):
        return tf.math.log(self._prob(value, **kwargs))
      raise NotImplementedError('log_prob is not implemented: {}'.format(
          type(self).__name__))

  def log_prob(self, value, name='log_prob', **kwargs):
    """Log probability density/mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    return self._call_log_prob(value, name, **kwargs)

  def _call_prob(self, value, name, **kwargs):
    """Wrapper around _prob."""
    value = _convert_to_tensor(value, name='value', dtype_hint=self.dtype)
    with self._name_and_control_scope(name, value, kwargs):
      if hasattr(self, '_prob'):
        return self._prob(value, **kwargs)
      if hasattr(self, '_log_prob'):
        return tf.exp(self._log_prob(value, **kwargs))
      raise NotImplementedError('prob is not implemented: {}'.format(
          type(self).__name__))

  def prob(self, value, name='prob', **kwargs):
    """Probability density/mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    return self._call_prob(value, name, **kwargs)

  def _call_log_cdf(self, value, name, **kwargs):
    """Wrapper around _log_cdf."""
    value = _convert_to_tensor(value, name='value', dtype_hint=self.dtype)
    with self._name_and_control_scope(name, value, kwargs):
      if hasattr(self, '_log_cdf'):
        return self._log_cdf(value, **kwargs)
      if hasattr(self, '_cdf'):
        return tf.math.log(self._cdf(value, **kwargs))
      raise NotImplementedError('log_cdf is not implemented: {}'.format(
          type(self).__name__))

  def log_cdf(self, value, name='log_cdf', **kwargs):
    """Log cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```none
    log_cdf(x) := Log[ P[X <= x] ]
    ```

    Often, a numerical approximation can be used for `log_cdf(x)` that yields
    a more accurate answer than simply taking the logarithm of the `cdf` when
    `x << -1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      logcdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    return self._call_log_cdf(value, name, **kwargs)

  def _call_cdf(self, value, name, **kwargs):
    """Wrapper around _cdf."""
    value = _convert_to_tensor(value, name='value', dtype_hint=self.dtype)
    with self._name_and_control_scope(name, value, kwargs):
      if hasattr(self, '_cdf'):
        return self._cdf(value, **kwargs)
      if hasattr(self, '_log_cdf'):
        return tf.exp(self._log_cdf(value, **kwargs))
      raise NotImplementedError('cdf is not implemented: {}'.format(
          type(self).__name__))

  def cdf(self, value, name='cdf', **kwargs):
    """Cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```none
    cdf(x) := P[X <= x]
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      cdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    return self._call_cdf(value, name, **kwargs)

  def _log_survival_function(self, value, **kwargs):
    raise NotImplementedError(
        'log_survival_function is not implemented: {}'.format(
            type(self).__name__))

  def _call_log_survival_function(self, value, name, **kwargs):
    """Wrapper around _log_survival_function."""
    value = _convert_to_tensor(value, name='value', dtype_hint=self.dtype)
    with self._name_and_control_scope(name, value, kwargs):
      try:
        return self._log_survival_function(value, **kwargs)
      except NotImplementedError as original_exception:
        if hasattr(self, '_log_cdf'):
          return log1mexp(self._log_cdf(value, **kwargs))
        if hasattr(self, '_cdf'):
          return tf.math.log1p(-self._cdf(value, **kwargs))
        raise original_exception

  def log_survival_function(self, value, name='log_survival_function',
                            **kwargs):
    """Log survival function.

    Given random variable `X`, the survival function is defined:

    ```none
    log_survival_function(x) = Log[ P[X > x] ]
                             = Log[ 1 - P[X <= x] ]
                             = Log[ 1 - cdf(x) ]
    ```

    Typically, different numerical approximations can be used for the log
    survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
    return self._call_log_survival_function(value, name, **kwargs)

  def _survival_function(self, value, **kwargs):
    raise NotImplementedError('survival_function is not implemented: {}'.format(
        type(self).__name__))

  def _call_survival_function(self, value, name, **kwargs):
    """Wrapper around _survival_function."""
    value = _convert_to_tensor(value, name='value', dtype_hint=self.dtype)
    with self._name_and_control_scope(name, value, kwargs):
      try:
        return self._survival_function(value, **kwargs)
      except NotImplementedError as original_exception:
        if hasattr(self, '_log_cdf'):
          return -tf.math.expm1(self._log_cdf(value, **kwargs))
        if hasattr(self, '_cdf'):
          return 1. - self.cdf(value, **kwargs)
        raise original_exception

  def survival_function(self, value, name='survival_function', **kwargs):
    """Survival function.

    Given random variable `X`, the survival function is defined:

    ```none
    survival_function(x) = P[X > x]
                         = 1 - P[X <= x]
                         = 1 - cdf(x).
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
    return self._call_survival_function(value, name, **kwargs)

  def _entropy(self, **kwargs):
    raise NotImplementedError('entropy is not implemented: {}'.format(
        type(self).__name__))

  def entropy(self, name='entropy', **kwargs):
    """Shannon entropy in nats."""
    with self._name_and_control_scope(name):
      return self._entropy(**kwargs)

  def _mean(self, **kwargs):
    raise NotImplementedError('mean is not implemented: {}'.format(
        type(self).__name__))

  def mean(self, name='mean', **kwargs):
    """Mean."""
    with self._name_and_control_scope(name):
      return self._mean(**kwargs)

  def _quantile(self, value, **kwargs):
    raise NotImplementedError('quantile is not implemented: {}'.format(
        type(self).__name__))

  def _call_quantile(self, value, name, **kwargs):
    with self._name_and_control_scope(name):
      dtype = tf.float32 if tf.nest.is_nested(self.dtype) else self.dtype
      value = tf.convert_to_tensor(value, name='value', dtype_hint=dtype)
      if self.validate_args:
        value = distribution_util.with_dependencies([
            assert_util.assert_less_equal(value, tf.cast(1, value.dtype),
                                          message='`value` must be <= 1'),
            assert_util.assert_greater_equal(value, tf.cast(0, value.dtype),
                                             message='`value` must be >= 0')
        ], value)
      return self._quantile(value, **kwargs)

  def quantile(self, value, name='quantile', **kwargs):
    """Quantile function. Aka 'inverse cdf' or 'percent point function'.

    Given random variable `X` and `p in [0, 1]`, the `quantile` is:

    ```none
    quantile(p) := x such that P[X <= x] == p
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      quantile: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
    return self._call_quantile(value, name, **kwargs)

  def _variance(self, **kwargs):
    raise NotImplementedError('variance is not implemented: {}'.format(
        type(self).__name__))

  def variance(self, name='variance', **kwargs):
    """Variance.

    Variance is defined as,

    ```none
    Var = E[(X - E[X])**2]
    ```

    where `X` is the random variable associated with this distribution, `E`
    denotes expectation, and `Var.shape = batch_shape + event_shape`.

    Args:
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      variance: Floating-point `Tensor` with shape identical to
        `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
    """
    with self._name_and_control_scope(name):
      try:
        return self._variance(**kwargs)
      except NotImplementedError as original_exception:
        try:
          return tf.square(self._stddev(**kwargs))
        except NotImplementedError:
          raise original_exception

  def _stddev(self, **kwargs):
    raise NotImplementedError('stddev is not implemented: {}'.format(
        type(self).__name__))

  def stddev(self, name='stddev', **kwargs):
    """Standard deviation.

    Standard deviation is defined as,

    ```none
    stddev = E[(X - E[X])**2]**0.5
    ```

    where `X` is the random variable associated with this distribution, `E`
    denotes expectation, and `stddev.shape = batch_shape + event_shape`.

    Args:
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      stddev: Floating-point `Tensor` with shape identical to
        `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
    """

    with self._name_and_control_scope(name):
      try:
        return self._stddev(**kwargs)
      except NotImplementedError as original_exception:
        try:
          return tf.sqrt(self._variance(**kwargs))
        except NotImplementedError:
          raise original_exception

  def _covariance(self, **kwargs):
    raise NotImplementedError('covariance is not implemented: {}'.format(
        type(self).__name__))

  def covariance(self, name='covariance', **kwargs):
    """Covariance.

    Covariance is (possibly) defined only for non-scalar-event distributions.

    For example, for a length-`k`, vector-valued distribution, it is calculated
    as,

    ```none
    Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
    ```

    where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
    denotes expectation.

    Alternatively, for non-vector, multivariate distributions (e.g.,
    matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
    under some vectorization of the events, i.e.,

    ```none
    Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
    ```

    where `Cov` is a (batch of) `k' x k'` matrices,
    `0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
    mapping indices of this distribution's event dimensions to indices of a
    length-`k'` vector.

    Args:
      name: Python `str` prepended to names of ops created by this function.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      covariance: Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
        where the first `n` dimensions are batch coordinates and
        `k' = reduce_prod(self.event_shape)`.
    """
    with self._name_and_control_scope(name):
      return self._covariance(**kwargs)

  def _mode(self, **kwargs):
    raise NotImplementedError('mode is not implemented: {}'.format(
        type(self).__name__))

  def mode(self, name='mode', **kwargs):
    """Mode."""
    with self._name_and_control_scope(name):
      return self._mode(**kwargs)

  def _cross_entropy(self, other):
    return kullback_leibler.cross_entropy(
        self, other, allow_nan_stats=self.allow_nan_stats)

  def cross_entropy(self, other, name='cross_entropy'):
    """Computes the (Shannon) cross entropy.

    Denote this distribution (`self`) by `P` and the `other` distribution by
    `Q`. Assuming `P, Q` are absolutely continuous with respect to
    one another and permit densities `p(x) dr(x)` and `q(x) dr(x)`, (Shannon)
    cross entropy is defined as:

    ```none
    H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
    ```

    where `F` denotes the support of the random variable `X ~ P`.

    Args:
      other: `tfp.distributions.Distribution` instance.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      cross_entropy: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
        representing `n` different calculations of (Shannon) cross entropy.
    """
    with self._name_and_control_scope(name):
      return self._cross_entropy(other)

  def _kl_divergence(self, other):
    return kullback_leibler.kl_divergence(
        self, other, allow_nan_stats=self.allow_nan_stats)

  def kl_divergence(self, other, name='kl_divergence'):
    """Computes the Kullback--Leibler divergence.

    Denote this distribution (`self`) by `p` and the `other` distribution by
    `q`. Assuming `p, q` are absolutely continuous with respect to reference
    measure `r`, the KL divergence is defined as:

    ```none
    KL[p, q] = E_p[log(p(X)/q(X))]
             = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x)
             = H[p, q] - H[p]
    ```

    where `F` denotes the support of the random variable `X ~ p`, `H[., .]`
    denotes (Shannon) cross entropy, and `H[.]` denotes (Shannon) entropy.

    Args:
      other: `tfp.distributions.Distribution` instance.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      kl_divergence: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
        representing `n` different calculations of the Kullback-Leibler
        divergence.
    """
    # NOTE: We do not enter a `self._name_and_control_scope` here.  We rely on
    # `tfd.kl_divergence(self, other)` to use `_name_and_control_scope` to apply
    # assertions on both Distributions.
    #
    # Subclasses that override `Distribution.kl_divergence` or `_kl_divergence`
    # must ensure that assertions are applied for both `self` and `other`.
    return self._kl_divergence(other)

  def _default_event_space_bijector(self):
    raise NotImplementedError(
        '_default_event_space_bijector` is not implemented: {}'.format(
            type(self).__name__))

  def _experimental_default_event_space_bijector(self):
    """Bijector mapping the reals (R**n) to the event space of the distribution.

    Returns:
      event_space_bijector: `Bijector` instance or `None`.

    Distributions with continuous support have a
    `_default_event_space_bijector`, a subclass of `tfp.bijectors.Bijector`
    that maps R**n to the distribution's event space. For example, the
    `_default_event_space_bijector` of the `Beta` distribution is
    `tfb.Sigmoid()`, which maps the real line to `[0, 1]`, the support of the
    `Beta` distribution. The purpose of `_default_event_space_bijector` is
    to enable gradient descent in an unconstrained space for Variational
    Inference and Hamiltonian Monte Carlo methods. An effort has been made to
    choose bijectors such that the tails of the distribution in the
    unconstrained space are between Gaussian and Exponential. For distributions
    with discrete event space, `_default_event_space_bijector` returns `None`.
    """
    return self._default_event_space_bijector()

  def __str__(self):
    if self.batch_shape:
      maybe_batch_shape = ', batch_shape=' + _str_tensorshape(self.batch_shape)
    else:
      maybe_batch_shape = ''
    if self.event_shape:
      maybe_event_shape = ', event_shape=' + _str_tensorshape(self.event_shape)
    else:
      maybe_event_shape = ''
    if self.dtype is not None:
      maybe_dtype = ', dtype=' + _str_dtype(self.dtype)
    else:
      maybe_dtype = ''
    return ('tfp.distributions.{type_name}('
            '"{self_name}"'
            '{maybe_batch_shape}'
            '{maybe_event_shape}'
            '{maybe_dtype})'.format(
                type_name=type(self).__name__,
                self_name=self.name or '<unknown>',
                maybe_batch_shape=maybe_batch_shape,
                maybe_event_shape=maybe_event_shape,
                maybe_dtype=maybe_dtype))

  def __repr__(self):
    return ('<tfp.distributions.{type_name} '
            '\'{self_name}\''
            ' batch_shape={batch_shape}'
            ' event_shape={event_shape}'
            ' dtype={dtype}>'.format(
                type_name=type(self).__name__,
                self_name=self.name or '<unknown>',
                batch_shape=_str_tensorshape(self.batch_shape),
                event_shape=_str_tensorshape(self.event_shape),
                dtype=_str_dtype(self.dtype)))

  @contextlib.contextmanager
  def _name_and_control_scope(self, name=None, value=UNSET_VALUE, kwargs=None):
    """Helper function to standardize op scope."""
    # Note: we recieve `kwargs` and not `**kwargs` to ensure no collisions on
    # other args we choose to take in this function.
    with tf.name_scope(self.name):
      with tf.name_scope(name) as name_scope:
        deps = []
        deps.extend(self._initial_parameter_control_dependencies)
        deps.extend(self._parameter_control_dependencies(is_init=False))
        if value is not UNSET_VALUE:
          deps.extend(self._sample_control_dependencies(
              value, **({} if kwargs is None else kwargs)))
        if not deps:
          yield name_scope
          return
        with tf.control_dependencies(deps) as deps_scope:
          yield deps_scope

  def _expand_sample_shape_to_vector(self, x, name):
    """Helper to `sample` which ensures input is 1D."""
    x_static_val = tf.get_static_value(x)
    if x_static_val is None:
      prod = tf.reduce_prod(x)
    else:
      prod = np.prod(x_static_val, dtype=dtype_util.as_numpy_dtype(x.dtype))

    x = distribution_util.expand_to_vector(x, tensor_name=name)
    return x, prod

  def _set_sample_static_shape(self, x, sample_shape):
    """Helper to `sample`; sets static shape info."""
    batch_shape = self.batch_shape
    if (tf.nest.is_nested(self.dtype)
        and not tf.nest.is_nested(batch_shape)):
      batch_shape = tf.nest.map_structure(
          lambda _: batch_shape, self.dtype)

    return tf.nest.map_structure(
        functools.partial(
            _set_sample_static_shape_for_tensor, sample_shape=sample_shape),
        x, self.event_shape, batch_shape)

  def _is_scalar_helper(self, static_shape, dynamic_shape_fn):
    """Implementation for `is_scalar_batch` and `is_scalar_event`."""
    if tensorshape_util.rank(static_shape) is not None:
      return tensorshape_util.rank(static_shape) == 0
    shape = dynamic_shape_fn()
    if tf.compat.dimension_value(shape.shape[0]) is not None:
      # If the static_shape is correctly written then we should never execute
      # this branch. We keep it just in case there's some unimagined corner
      # case.
      return tensorshape_util.as_list(shape.shape) == [0]
    return tf.equal(tf.shape(shape)[0], 0)

  def _parameter_control_dependencies(self, is_init):
    """Returns a list of ops to be executed in members with graph deps.

    Typically subclasses override this function to return parameter specific
    assertions (eg, positivity of `scale`, etc.).

    Args:
      is_init: Python `bool` indicating that the call site is `__init__`.

    Returns:
      dependencies: `list`-like of ops to be executed in member functions with
        graph dependencies.
    """
    return ()

  def _sample_control_dependencies(self, value, **kwargs):
    """Returns a list of ops to be executed to validate distribution samples.

    The ops are executed in methods that take distribution samples as an
    argument (e.g. `log_prob` and `cdf`). They validate that `value` is
    within the support of the distribution. Typically subclasses override this
    function to return assertions specific to the distribution (e.g. samples
    from `Beta` must be between `0` and `1`). By convention, finite bounds of
    the support are considered valid samples, since `sample` may output values
    that are numerically equivalent to the bounds.

    Args:
      value: `float` or `double` `Tensor`.
      **kwargs: Additional keyword args.

    Returns:
      assertions: `list`-like of ops to be executed in member functions that
        take distribution samples as input.
    """
    return ()


class _PrettyDict(dict):
  """`dict` with stable `repr`, `str`."""

  def __str__(self):
    pairs = (': '.join([str(k), str(v)]) for k, v in sorted(self.items()))
    return '{' + ', '.join(pairs) + '}'

  def __repr__(self):
    pairs = (': '.join([repr(k), repr(v)]) for k, v in sorted(self.items()))
    return '{' + ', '.join(pairs) + '}'


def _recursively_replace_dict_for_pretty_dict(x):
  """Recursively replace `dict`s with `_PrettyDict`."""
  # We use "PrettyDict" because collections.OrderedDict repr/str has the word
  # "OrderedDict" in it. We only want to print "OrderedDict" if in fact the
  # input really is an OrderedDict.
  if isinstance(x, dict):
    return _PrettyDict({
        k: _recursively_replace_dict_for_pretty_dict(v)
        for k, v in x.items()})
  if (isinstance(x, collections.Sequence) and
      not isinstance(x, six.string_types)):
    args = (_recursively_replace_dict_for_pretty_dict(x_) for x_ in x)
    is_named_tuple = (isinstance(x, tuple) and
                      hasattr(x, '_asdict') and
                      hasattr(x, '_fields'))
    return type(x)(*args) if is_named_tuple else type(x)(args)
  if isinstance(x, collections.Mapping):
    return type(x)(**{k: _recursively_replace_dict_for_pretty_dict(v)
                      for k, v in x.items()})
  return x


def _str_tensorshape(x):
  def _str(s):
    if tensorshape_util.rank(s) is None:
      return '?'
    return str(tensorshape_util.as_list(s)).replace('None', '?')
  # Because Python2 `dict`s are unordered, we must replace them with
  # `PrettyDict`s so __str__, __repr__ are deterministic.
  x = _recursively_replace_dict_for_pretty_dict(x)
  return str(tf.nest.map_structure(_str, x)).replace('\'', '')


def _str_dtype(x):
  def _str(s):
    if s is None:
      return '?'
    return dtype_util.name(s)
  # Because Python2 `dict`s are unordered, we must replace them with
  # `PrettyDict`s so __str__, __repr__ are deterministic.
  x = _recursively_replace_dict_for_pretty_dict(x)
  return str(tf.nest.map_structure(_str, x)).replace('\'', '')
