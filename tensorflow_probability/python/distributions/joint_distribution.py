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
"""The `JointDistributionSequential` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util

from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


def _make_summary_statistic(attr):
  """Factory for making summary statistics, eg, mean, mode, stddev."""
  def _fn(self):
    if not all(self._is_distribution_instance):  # pylint: disable=protected-access
      raise ValueError(
          'Can only compute ' + attr + 'when all distributions are '
          'independent.')
    return tuple(getattr(d, attr)() for d in self.distribution_fn)
  return _fn


class JointDistributionSequential(distribution_lib.Distribution):
  """Joint distribution parameterized by distribution-making functions.

  This distribution enables both sampling and joint probability computation from
  a single model specification.

  A joint distribution is a collection of possibly interdependent distributions.
  Like `tf.keras.Sequential`, the `JointDistributionSequential` can be specified
  via a `list` of functions (each responsible for making a
  `tfp.distributions.Distribution`-like instance).  Unlike
  `tf.keras.Sequential`, each function can depend on the output of all previous
  elements rather than only the immediately previous.

  #### Mathematical Details

  The `JointDistributionSequential` implements the chain rule of probability.
  That is, the probability function of a length-`d` vector `x` is,

  ```none
  p(x) = prod{ p(x[i] | x[:i]) : i = 0, ..., (d - 1) }
  ```

  The `JointDistributionSequential` is parameterized by a `list` comprised of
  either:

  1. `tfp.distributions.Distribution`-like instances or,
  2. `callable`s which return a `tfp.distributions.Distribution`-like instance.

  Each `list` element implements the `i`-th *full conditional distribution*,
  `p(x[i] | x[:i])`. The "conditioned on" elements are represented by the
  `callable`'s required arguments. Directly providing a `Distribution`-like
  instance is a convenience and is semantically identical a zero argument
  `callable`.

  Denote the `i`-th `callable`s non-default arguments as `args[i]`. Since the
  `callable` is the conditional manifest, `0 <= len(args[i]) <= i - 1`. When
  `len(args[i]) < i - 1`, the `callable` only depends on a subset of the
  previous distributions, specifically those at indexes:
  `range(i - 1, i - 1 - num_args[i], -1)`.
  (See "Examples" and "Discussion" for why the order is reversed.)

  #### Examples

  ```python
  tfd = tfp.distributions

  # Consider the following generative model:
  #     e ~ Exponential(rate=[100,120])
  #     g ~ Gamma(concentration=e[0], rate=e[1])
  #     n ~ Normal(loc=0, scale=2.)
  #     m ~ Normal(loc=n, scale=g)
  #     for i = 1, ..., 12:
  #       x[i] ~ Bernoulli(logits=m)

  # In TFP, we can write this as:
  joint = tfd.JointDistributionSequential([
                   tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),  # e
      lambda    e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),    # g
                   tfd.Normal(loc=0, scale=2.),                           # n
      lambda n, g: tfd.Normal(loc=n, scale=g)                             # m
      lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12)                # x
  ])
  # (Notice the 1:1 correspondence between "math" and "code".)

  x = joint.sample()
  # ==> A length-5 list of tfd.Distribution instances
  joint.log_prob(x)
  # ==> A scalar `Tensor` representing the total log prob under all four
  #     distributions.

  joint._resolve_graph()
  # ==> (('e', ()),
  #      ('g', ('e',)),
  #      ('n', ()),
  #      ('m', ('n', 'g')),
  #      ('x', ('m',)))
  ```

  #### Discussion

  `JointDistributionSequential` builds each distribution in `list` order; list
  items must be either a:

  1. `tfd.Distribution`-like instance (e.g., `e` and `n`), or a
  2. Python `callable` (e.g., `g`, `m`, `x`).

  Regarding #1, an object is deemed "`tfd.Distribution`-like" if it has a
  `sample`, `log_prob`, and distribution properties, e.g., `batch_shape`,
  `event_shape`, `dtype`.

  Regarding #2, in addition to using a function (or `lambda`), supplying a TFD
  "`class`" is also permissible, this also being a "Python `callable`." For
  example, instead of writing `lambda n, g: tfd.Normal(loc=n, scale=g) ` one
  could have simply written `tfd.Normal`.

  Notice that directly providing a `tfd.Distribution`-like instance means there
  cannot exist a (dynamic) dependency on other distributions; it is
  "independent" both "computationally" and "statistically." The same is
  self-evidently true of zero-argument `callable`s.

  A distribution instance depends on other distribution instances through the
  distribution making function's *required arguments*. If the distribution maker
  has `k` required arguments then the `JointDistributionSequential` calls the
  maker with samples produced by the previous `k` distributions.

  **Note**: **maker arguments are provided in reverse order** of the previous
  elements in the list. In the example, notice that `m` depends on `n` and `g`
  in this order. The order is reversed for convenience. We reverse arguments
  under the heuristic observation that many graphical models have chain-like
  dependencies which are self-evidently topologically sorted from the human
  cognitive process of postulating the generative process. By feeding the
  previous num required args in reverse order we (often) enable a simpler maker
  function signature. If the maker needs to depend on distribution previous to
  one which is not a dependency, one must use a dummy arg, to "gobble up" the
  unused dependency, e.g.,
  `lambda _ignore, actual_dependency: SomeDistribution(actual_dependency)`.

  **Note**: unlike every other distribution in `tfp.distributions`,
  `JointDistributionSequential.sample` returns a list of `Tensor`s rather than a
  `Tensor`.  Accordingly `joint.batch_shape` returns a list of `TensorShape`s
  for each of the distributions' batch shapes and `joint.batch_shape_tensor()`
  returns a list of `Tensor`s for each of the distributions' event shapes. (Same
  with `event_shape` analogues.)
  """

  def __init__(self, distribution_fn, validate_args=False, name=None):
    """Construct the `JointDistributionSequential` distribution.

    Args:
      distribution_fn: Python list of either tfd.Distribution instances and/or
        lambda functions which take the `k` previous distributions and returns a
        new tfd.Distribution instance.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      name: The name for ops managed by the distribution.
        Default value: `"JointDistributionSequential"`.
    """
    parameters = dict(locals())
    name = name or 'JointDistributionSequential'
    with tf.compat.v1.name_scope(name) as name:
      self._distribution_fn = distribution_fn
      [
          self._wrapped_distribution_fn,
          self._args,
          self._is_distribution_instance,
      ] = zip(*[_unify_call_signature(i, dist_fn)
                for i, dist_fn in enumerate(distribution_fn)])
      self._most_recently_built_distributions = [
          d if is_dist else None for d, is_dist
          in zip(distribution_fn, self._is_distribution_instance)]
      self._always_use_specified_sample_shape = False
      super(JointDistributionSequential, self).__init__(
          dtype=tuple(
              None if d is None else d.dtype
              for d in self._most_recently_built_distributions),
          reparameterization_type=tuple(
              None if d is None else d.reparameterization_type
              for d in self._most_recently_built_distributions),
          validate_args=validate_args,
          allow_nan_stats=False,
          parameters=parameters,
          graph_parents=[],
          name=name)

  @property
  def distribution_fn(self):
    return self._distribution_fn

  @property
  def dtype(self):
    """The `DType` of `Tensor`s handled by this `Distribution`."""
    return tuple(None if d is None else d.dtype
                 for d in self._most_recently_built_distributions)

  @property
  def reparameterization_type(self):
    """Describes how samples from the distribution are reparameterized.

    Currently this is one of the static instances
    `tfd.FULLY_REPARAMETERIZED` or `tfd.NOT_REPARAMETERIZED`.

    Returns:
      reparameterization_type: `tuple` of `ReparameterizationType` for each
        `distribution_fn`.
    """
    return tuple(None if d is None else d.reparameterization_type
                 for d in self._most_recently_built_distributions)

  def batch_shape_tensor(self, name='batch_shape_tensor'):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `tuple` of `Tensor`s representing the `batch_shape` for each
        of `distribution_fn`.
    """
    with self._name_scope(name):
      if all(self._is_distribution_instance):
        ds = self.distribution_fn
      else:
        ds, _ = self.sample_distributions(seed=42)  # Const seed for maybe CSE.
      return tuple(d.batch_shape_tensor() for d in ds)

  @property
  def batch_shape(self):
    """Shape of a single sample from a single event index as a `TensorShape`.

    May be partially defined or unknown.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Returns:
      batch_shape: `tuple` of `TensorShape`s representing the `batch_shape` for
        each of `distribution_fn`.
    """
    # The following cannot leak graph Tensors in eager because `batch_shape` is
    # a `TensorShape`.
    return tuple(tf.TensorShape(None) if d is None else d.batch_shape
                 for d in self._most_recently_built_distributions)

  def event_shape_tensor(self, name='event_shape_tensor'):
    """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `tuple` of `Tensor`s representing the `event_shape` for each
        of `distribution_fn`.
    """
    with self._name_scope(name):
      if all(self._is_distribution_instance):
        ds = self.distribution_fn
      else:
        ds, _ = self.sample_distributions(seed=42)  # Const seed for maybe CSE.
      return tuple(d.event_shape_tensor() for d in ds)

  @property
  def event_shape(self):
    """Shape of a single sample from a single batch as a `TensorShape`.

    May be partially defined or unknown.

    Returns:
      event_shape: `tuple` of `TensorShape`s representing the `event_shape` for
        each of `distribution_fn`.
    """
    # The following cannot leak graph Tensors in eager because `batch_shape` is
    # a `TensorShape`.
    return tuple(tf.TensorShape(None) if d is None else d.event_shape
                 for d in self._most_recently_built_distributions)

  def sample_distributions(self, sample_shape=(), seed=None, value=None,
                           name='sample_distributions'):
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

    Returns:
      distributions: a `tuple` of `Distribution` instances for each of
        `distribution_fn`.
      samples: a `tuple` of `Tensor`s with prepended dimensions `sample_shape`
        for each of `distribution_fn`.
    """
    seed = seed_stream.SeedStream('JointDistributionSequential', seed)
    with self._name_scope(name, values=[sample_shape, seed, value]):
      ds = []
      if value is None:
        xs = [None] * len(self._wrapped_distribution_fn)
      else:
        xs = list(value)
        xs.extend([None]*(len(self._wrapped_distribution_fn) - len(xs)))
      if len(xs) != len(self.distribution_fn):
        raise ValueError('Number of `xs`s must match number of '
                         'distributions.')
      for i, (dist_fn, args) in enumerate(zip(self._wrapped_distribution_fn,
                                              self._args)):
        ds.append(dist_fn(*xs[:i]))  # Chain rule of probability.
        if xs[i] is None:
          # TODO(b/129364796): We should ignore args prefixed with `_`; this
          # would mean we more often identify when to use `sample_shape=()`
          # rather than `sample_shape=sample_shape`.
          xs[i] = ds[-1].sample(
              () if args and not self._always_use_specified_sample_shape
              else sample_shape, seed=seed())
        else:
          xs[i] = tf.convert_to_tensor(value=xs[i], dtype_hint=ds[-1].dtype)
          seed()  # Ensure reproducibility even when xs are (partially) set.
      self._most_recently_built_distributions = ds
      return tuple(ds), tuple(xs)

  def sample(self, sample_shape=(), seed=None, value=None, name='sample'):
    """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for generating random numbers.
      value: `list` of `Tensor`s in `distribution_fn` order to use to
        parameterize other ("downstream") distribution makers.
        Default value: `None` (i.e., draw a sample from each distribution).
      name: name prepended to ops created by this function.
        Default value: `"sample"`.

    Returns:
      samples: a `tuple` of `Tensor`s with prepended dimensions `sample_shape`
        for each of `distribution_fn`.
    """
    with self._name_scope(name, values=[sample_shape, seed, value]):
      _, xs = self.sample_distributions(sample_shape, seed, value)
      return xs

  def _call_sample_n(self, sample_shape, seed, name, **kwargs):
    # Implemented here so generically calling `Distribution.sample` still works.
    # (This is needed for convenient Tensor coercion in tfp.layers.)
    return self.sample(sample_shape, seed, value=None, name=name)

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
    if any(v is None for v in value):
      raise ValueError('No `value` part can be `None`.')
    with self._name_scope(name, values=[value]):
      return _maybe_check_wont_broadcast(
          (d.log_prob(x) for d, x
           in zip(*self.sample_distributions(value=value))),
          self.validate_args)

  def prob_parts(self, value, name='prob_parts'):
    """Log probability density/mass function.

    Args:
      value: `list` of `Tensor`s in `distribution_fn` order for which we compute
        the `prob_parts` and to parameterize other ("downstream") distributions.
      name: name prepended to ops created by this function.
        Default value: `"prob_parts"`.

    Returns:
      prob_parts: a `tuple` of `Tensor`s representing the `prob` for
        each `distribution_fn` evaluated at each corresponding `value`.
    """
    if any(v is None for v in value):
      raise ValueError('No `value` part can be `None`.')
    with self._name_scope(name, values=[value]):
      return _maybe_check_wont_broadcast(
          (d.prob(x) for d, x
           in zip(*self.sample_distributions(value=value))),
          self.validate_args)

  def _resolve_graph(self, distribution_names=None, leaf_name='x'):
    """Creates a `tuple` of `tuple`s of dependencies.

    This function is **experimental**. That said, we encourage its use
    and ask that you report problems to `tfprobability@tensorflow.org`.

    Args:
      distribution_names: `list` of `str` or `None` names corresponding to each
        of `distribution_fn` elements. (`None`s are expanding into the
        appropriate `str`.)
      leaf_name: `str` used when no maker depends on a particular
        `distribution_fn` element.

    Returns:
      graph: `tuple` of `(str tuple)` pairs representing the name of each
        distribution (maker) and the names of its dependencies.

    #### Example

    ```python
    d = tfd.JointDistributionSequential([
                     tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
        lambda    e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
                     tfd.Normal(loc=0, scale=2.),
        lambda n, g: tfd.Normal(loc=n, scale=g),
    ])
    d._resolve_graph()
    # ==> (
    #       ('e', ()),
    #       ('g', ('e',)),
    #       ('n', ()),
    #       ('x', ('n', 'g')),
    #     )
    ```

    """
    # TODO(b/129008220): Robustify this procedure. Eg, handle collisions better,
    # ignore args prefixed with `_`.
    if distribution_names is None or any(n is None for n in self._args):
      distribution_names = _resolve_distribution_names(
          self._args, distribution_names, leaf_name)
    if len(set(distribution_names)) != len(distribution_names):
      raise ValueError('Distribution names must be unique: {}'.format(
          distribution_names))
    if len(distribution_names) != len(self.distribution_fn):
      raise ValueError('Distribution names must be 1:1 with `rvs`.')
    return tuple(zip(distribution_names, self._args))

  _mean = _make_summary_statistic('mean')
  _mode = _make_summary_statistic('mode')
  _stddev = _make_summary_statistic('stddev')
  _variance = _make_summary_statistic('variance')

  def _log_prob(self, xs):
    return sum(self.log_prob_parts(xs))

  def _entropy(self):
    """Shannon entropy in nats."""
    if not all(self._is_distribution_instance):
      raise ValueError(
          'Can only compute entropy when all distributions are independent.')
    return sum(_maybe_check_wont_broadcast(
        (d.entropy() for d in self.distribution_fn),
        self.validate_args))

  def _cross_entropy(self, other):
    if (not isinstance(other, JointDistributionSequential) or
        len(self.distribution_fn) != len(other.distribution_fn)):
      raise ValueError(
          'Can only compute cross entropy between `JointDistributionSequential`s '
          'with the same number of component distributions.')
    if (not all(self._is_distribution_instance) or
        not all(other._is_distribution_instance)):  # pylint: disable=protected-access
      raise ValueError(
          'Can only compute cross entropy when all component distributions '
          'are independent.')
    return sum(_maybe_check_wont_broadcast(
        (d0.cross_entropy(d1) for d0, d1
         in zip(self.distribution_fn, other.distribution_fn)),
        self.validate_args))

  def __getitem__(self, slices):
    # There will be some users that who won't like how we've implemented
    # slicing. The currently implemented policy is to slice "upstream" and let
    # post sliced distributions propagate forward. While this is efficient it
    # also means that downstream makers *might* no longer correctly broadcast
    # with the new inputs.
    #
    # For example:
    #
    # ```python
    # d = JointDistributionSequential([
    #   tfd.Normal(0, scale=[0.5, 1, 1.5]),
    #   lambda m: tfd.Normal(loc=m, scale=[1, 2, 3]),
    # ])
    # d[1:]
    # # ==> An exception: the following is not possible:
    # #     Normal(loc=Normal(0, [1, 1.5]).sample(), scale=[1, 2, 3])
    # #     since shape `[2]` cannot broadcast with shape `[3]`.
    # ```
    #
    # While not supporting this is technically not a bug (its an API decision)
    # it certainly won't be universally loved. That this is ok follows from the
    # fact that it will at least fail loudly--unless you slice down so a
    # dimension now has size `1`--this will broadcast. Here again we argue this
    # is "ok" because at this level of TFP users are ultimately responsible for
    # ensuring the correct shapes in their models. Moreover, this problem can
    # be worked around by using a `Deterministic` node in lieu of any constants.
    #
    # Once JointDistributionSequential can be parameterized by a single maker
    # function, we can choose to support more slicing scenarios by building
    # every distribution as we would normally then slicing distributions once
    # built (and passing that as the sole maker to `self.copy`.) Note however
    # that this would not be as performant as we've currently implemented
    # things.
    dfn = []
    def _sliced_maker(d):
      def _fn():
        return d()[slices]
      return _fn
    for d, is_dist, args in zip(self.distribution_fn,
                                self._is_distribution_instance,
                                self._args):
      if is_dist:
        dfn.append(d[slices])
      elif args:
        # Don't slice makers which have inputs; they'll be sliced via upstream.
        dfn.append(d)
      else:
        # Makers which have no deps need slicing, just like distribution
        # instances.
        dfn.append(_sliced_maker(d))
    return self.copy(distribution_fn=dfn)

  def is_scalar_event(self, name='is_scalar_event'):
    """Indicates that `event_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_event: `bool` scalar `Tensor`.
    """
    with self._name_scope(name):
      if all(self._is_distribution_instance):
        ds = self.distribution_fn
      else:
        ds, _ = self.sample_distributions(seed=42)  # Const seed for maybe CSE.
      # As an added optimization we could also check
      # self._most_recently_built_distributions.
      return tuple(None if d is None else d.is_scalar_event() for d in ds)

  def is_scalar_batch(self, name='is_scalar_batch'):
    """Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor`.
    """
    with self._name_scope(name):
      if all(self._is_distribution_instance):
        ds = self.distribution_fn
      else:
        ds, _ = self.sample_distributions(seed=42)  # Const seed for maybe CSE.
      # As an added optimization we could also check
      # self._most_recently_built_distributions.
      return tuple(None if d is None else d.is_scalar_batch() for d in ds)


def _unify_call_signature(i, dist_fn):
  """Relieves arg unpacking burden from call site."""
  if distribution_util.is_distribution_instance(dist_fn):
    return (lambda *_: dist_fn), (), True
  if not callable(dist_fn):
    raise TypeError('{} must be either `tfd.Distribution`-like or '
                    '`callable`.'.format(dist_fn))
  args = _get_required_args(dist_fn)
  num_args = len(args)
  @functools.wraps(dist_fn)
  def wrapped_dist_fn(*xs):
    args = [] if i < 1 else xs[(i - 1)::-1]
    return dist_fn(*args[:num_args])
  return wrapped_dist_fn, args, False


def _get_required_args(fn):
  """Returns the distribution's required args."""
  argspec = distribution_util.getfullargspec(fn)
  args = argspec.args
  if tf_inspect.isclass(fn):
    # Remove the `self` arg.
    args = args[1:]
  if argspec.defaults:
    # Remove the args which have defaults. By convention we only feed
    # *required args*. This means some distributions must always be wrapped
    # with a `lambda`, e.g., `lambda logits: tfd.Bernoulli(logits=logits)`
    # or `lambda probs: tfd.Bernoulli(probs=probs)`.
    args = args[:-len(argspec.defaults)]
  return tuple(args)


def _resolve_distribution_names(dist_fn_args, dist_names, leaf_name):
  """Uses arg names to resolve distribution names."""
  if dist_names is None:
    dist_names = []
  else:
    dist_names = dist_names.copy()
  n = len(dist_fn_args)
  dist_names.extend([None]*(n - len(dist_names)))
  for i_, args in enumerate(reversed(dist_fn_args)):
    if not args:
      continue  # There's no args to analyze.
    i = n - i_ - 1
    for j, arg_name in enumerate(args):
      dist_names[i - j - 1] = arg_name
  j = 0
  for i_ in range(len(dist_names)):
    i = n - i_ - 1
    if dist_names[i] is None:
      dist_names[i] = leaf_name if j == 0 else '{}{}'.format(leaf_name, j)
      j += 1
  return tuple(dist_names)


def _maybe_check_wont_broadcast(parts, validate_args):
  """Verifies that `parts` dont broadcast."""
  parts = tuple(parts)  # So we can receive generators.
  if not validate_args:
    # Note: we don't try static validation because it is theoretically
    # possible that a user wants to take advantage of broadcasting.
    # Only when `validate_args` is `True` do we enforce the validation.
    return parts
  msg = 'Broadcasting probably indicates an error in model specification.'
  s = tuple(part.shape for part in parts)
  if all(s_.is_fully_defined() for s_ in s):
    if not all(a == b for a, b in zip(s[1:], s[:-1])):
      raise ValueError(msg)
    return parts
  assertions = [tf.compat.v1.assert_equal(a, b, message=msg)
                for a, b in zip(s[1:], s[:-1])]
  with tf.control_dependencies(assertions):
    return tuple(tf.identity(part) for part in parts)


@kullback_leibler.RegisterKL(JointDistributionSequential,
                             JointDistributionSequential)
def _kl_joint_joint(d0, d1, name=None):
  """Calculate the KL divergence between two `JointDistributionSequential`s.

  Args:
    d0: instance of a `JointDistributionSequential` object.
    d1: instance of a `JointDistributionSequential` object.
    name: (optional) Name to use for created operations.
      Default value: `"kl_joint_joint"`.

  Returns:
    kl_joint_joint: `Tensor` The sum of KL divergences between elemental
      distributions of two joint distributions.

  Raises:
    ValueError: when joint distributions have a different number of elemental
      distributions.
    ValueError: when either joint distribution has a distribution with dynamic
      dependency, i.e., when either joint distribution is not a collection of
      independent distributions.
  """
  if len(d0.distribution_fn) != len(d1.distribution_fn):
    raise ValueError(
        'Can only compute KL divergence between JointDistributionSequential '
        'distributions with the same number of component distributions.')
  if (not all(d0._is_distribution_instance) or  # pylint: disable=protected-access
      not all(d1._is_distribution_instance)):  # pylint: disable=protected-access
    raise ValueError(
        'Can only compute KL divergence when all distributions are '
        'independent.')
  with tf.compat.v1.name_scope(name, 'kl_joint_joint'):
    return sum(kullback_leibler.kl_divergence(d0_, d1_)
               for d0_, d1_ in zip(d0.distribution_fn, d1.distribution_fn))
