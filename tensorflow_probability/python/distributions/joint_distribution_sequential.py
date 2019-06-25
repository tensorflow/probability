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

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import joint_distribution as joint_distribution_lib
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import seed_stream
from tensorflow_probability.python.internal import distribution_util
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import


__all__ = [
    'JointDistributionSequential',
]


def _make_summary_statistic(attr):
  """Factory for making summary statistics, eg, mean, mode, stddev."""
  def _fn(self):
    if any(self._dist_fn_args):  # pylint: disable=protected-access
      raise ValueError(
          'Can only compute ' + attr + ' when all distributions are '
          'independent; {}'.format(self.model))
    return self._model_unflatten(  # pylint: disable=protected-access
        getattr(d(), attr)() for d in self._dist_fn_wrapped)  # pylint: disable=protected-access
  return _fn


class JointDistributionSequential(joint_distribution_lib.JointDistribution):
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
  # ==> A length-5 list of Tensors
  joint.log_prob(x)
  # ==> A scalar `Tensor` representing the total log prob under all five
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
  example, instead of writing:
  `lambda loc, scale: tfd.Normal(loc=loc, scale=scale)`
  one could have simply written `tfd.Normal`.

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

  **Note**: unlike other non-`JointDistribution` distributions in
  `tfp.distributions`, `JointDistribution.sample` (and subclasses) return a
  structure of  `Tensor`s rather than a `Tensor`.  A structure can be anything
  which is `list`-like, e.g., a `list` or `tuple` of `distribution` makers.
  Accordingly `joint.batch_shape` returns a `list`-like structure of
  `TensorShape`s for each of the distributions' batch shapes and
  `joint.batch_shape_tensor()` returns a `list`-like structure of `Tensor`s for
  each of the distributions' event shapes. (Same with `event_shape` analogues.)
  """

  def __init__(self, model, validate_args=False, name=None):
    """Construct the `JointDistributionSequential` distribution.

    Args:
      model: Python list of either tfd.Distribution instances and/or
        lambda functions which take the `k` previous distributions and returns a
        new tfd.Distribution instance.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `"JointDistributionSequential"`).
    """
    parameters = dict(locals())
    with tf.name_scope(name or 'JointDistributionSequential') as name:
      # Since we rely on `type(model)` and because `tf.Module` redefines
      # lists, tuples, dicts, we create two aliases of `model`.
      self._model_trackable = model
      self._model = self._no_dependency(model)
      self._build(model)
      self._most_recently_built_distributions = [
          None if a else d() for d, a
          in zip(self._dist_fn_wrapped, self._dist_fn_args)]
      self._always_use_specified_sample_shape = False
      super(JointDistributionSequential, self).__init__(
          dtype=None,  # Ignored; we'll override.
          reparameterization_type=None,  # Ignored; we'll override.
          validate_args=validate_args,
          allow_nan_stats=False,
          parameters=parameters,
          graph_parents=[],
          name=name)
      # Check valid structure.
      self._model_unflatten(self._model_flatten(model))

  def _build(self, model):
    """Creates `dist_fn`, `dist_fn_wrapped`, `dist_fn_args`."""
    if not isinstance(model, collections.Sequence):
      raise TypeError('`model` must be `list`-like (saw: {}).'.format(
          type(model).__name__))
    self._dist_fn = model
    self._dist_fn_wrapped, self._dist_fn_args = zip(*[
        _unify_call_signature(i, dist_fn)
        for i, dist_fn in enumerate(model)])

  def _flat_sample_distributions(self, sample_shape=(), seed=None, value=None):
    # This function additionally depends on:
    #   self._dist_fn_wrapped
    #   self._dist_fn_args
    #   self._always_use_specified_sample_shape
    seed = seed_stream.SeedStream('JointDistributionSequential', seed)
    ds = []
    xs = [None]*len(self._dist_fn_wrapped) if value is None else list(value)
    if len(xs) != len(self._dist_fn_wrapped):
      raise ValueError('Number of `xs`s must match number of '
                       'distributions.')
    for i, (dist_fn, args) in enumerate(zip(self._dist_fn_wrapped,
                                            self._dist_fn_args)):
      ds.append(dist_fn(*xs[:i]))  # Chain rule of probability.
      if xs[i] is None:
        # TODO(b/129364796): We should ignore args prefixed with `_`; this
        # would mean we more often identify when to use `sample_shape=()`
        # rather than `sample_shape=sample_shape`.
        xs[i] = ds[-1].sample(
            () if args and not self._always_use_specified_sample_shape
            else sample_shape, seed=seed())
      else:
        xs[i] = tf.convert_to_tensor(xs[i], dtype_hint=ds[-1].dtype)
        seed()  # Ensure reproducibility even when xs are (partially) set.
    # Note: we could also resolve distributions up to the first non-`None` in
    # `self._model_flatten(value)`, however we omit this feature for simplicity,
    # speed, and because it has not yet been requested.
    return ds, xs

  def _model_unflatten(self, xs):
    try:
      return type(self.model)(xs)
    except TypeError as e:
      raise TypeError(
          'Unable to unflatten like `model` with type "{}". '
          'Exception: {}'.format(type(self.model).__name__, e))

  def _model_flatten(self, xs):
    if xs is None:
      return (None,) * len(self._dist_fn_wrapped)
    try:
      xs = tuple(xs)
    except TypeError as e:
      raise TypeError(
          'Unable to flatten like `model` with type "{}". '
          'Exception: {}'.format(type(self.model).__name__, e))
    return xs + (None,)*(len(self._dist_fn_args) - len(xs))

  def _call_attr(self, attr):
    if any(self._dist_fn_args):
      # Const seed for maybe CSE.
      ds, _ = self._flat_sample_distributions(seed=42)
    else:
      ds = tuple(d() for d in self._dist_fn_wrapped)
    return (getattr(d, attr)() for d in ds)

  def _resolve_graph(self, distribution_names=None, leaf_name='x'):
    """Creates a `tuple` of `tuple`s of dependencies.

    This function is **experimental**. That said, we encourage its use
    and ask that you report problems to `tfprobability@tensorflow.org`.

    Args:
      distribution_names: `list` of `str` or `None` names corresponding to each
        of `model` elements. (`None`s are expanding into the
        appropriate `str`.)
      leaf_name: `str` used when no maker depends on a particular
        `model` element.

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
    # This function additionally depends on:
    #   self._dist_fn_args
    #   self._dist_fn_wrapped
    # TODO(b/129008220): Robustify this procedure. Eg, handle collisions better,
    # ignore args prefixed with `_`.
    if distribution_names is None or any(self._dist_fn_args):
      distribution_names = _resolve_distribution_names(
          self._dist_fn_args, distribution_names, leaf_name)
    if len(set(distribution_names)) != len(distribution_names):
      raise ValueError('Distribution names must be unique: {}'.format(
          distribution_names))
    if len(distribution_names) != len(self._dist_fn_wrapped):
      raise ValueError('Distribution names must be 1:1 with `rvs`.')
    return tuple(zip(distribution_names,
                     tuple(() if a is None else a for a in self._dist_fn_args)))

  _mean = _make_summary_statistic('mean')
  _mode = _make_summary_statistic('mode')
  _stddev = _make_summary_statistic('stddev')
  _variance = _make_summary_statistic('variance')

  def _entropy(self):
    """Shannon entropy in nats."""
    if any(self._dist_fn_args):
      raise ValueError(
          'Can only compute entropy when all distributions are independent.')
    return sum(joint_distribution_lib.maybe_check_wont_broadcast(
        (d().entropy() for d in self._dist_fn_wrapped),
        self.validate_args))

  def _cross_entropy(self, other):
    if (not isinstance(other, JointDistributionSequential) or
        len(self.model) != len(other.model)):
      raise ValueError(
          'Can only compute cross entropy between `JointDistributionSequential`s '
          'with the same number of component distributions.')
    if any(self._dist_fn_args) or any(other._dist_fn_args):  # pylint: disable=protected-access
      raise ValueError(
          'Can only compute cross entropy when all component distributions '
          'are independent.')
    return sum(joint_distribution_lib.maybe_check_wont_broadcast(
        (d0().cross_entropy(d1()) for d0, d1
         in zip(self._dist_fn_wrapped, other._dist_fn_wrapped)),  # pylint: disable=protected-access
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
    for d, args in zip(self._dist_fn, self._dist_fn_args):
      if args is None:
        dfn.append(d[slices])
      elif args:
        # Don't slice makers which have inputs; they'll be sliced via upstream.
        dfn.append(d)
      else:
        # Makers which have no deps need slicing, just like distribution
        # instances.
        dfn.append(_sliced_maker(d))
    return self.copy(model=self._model_unflatten(dfn))


def _unify_call_signature(i, dist_fn):
  """Creates `dist_fn_wrapped` which calls `dist_fn` with all prev nodes.

  Args:
    i: Python `int` corresponding to position in topologically sorted DAG.
    dist_fn: Python `callable` which takes a subset of previously constructed
      distributions (in reverse order) and produces a new distribution instance.

  Returns:
    dist_fn_wrapped: Python `callable` which takes all previous distributions
      (in non reverse order) and produces a  new distribution instance.
    args: `tuple` of `str` representing the arg names of `dist_fn` (and in non
      wrapped, "natural" order). `None` is returned only if the input is not a
      `callable`.
  """
  if distribution_util.is_distribution_instance(dist_fn):
    return (lambda *_: dist_fn), None

  if not callable(dist_fn):
    raise TypeError('{} must be either `tfd.Distribution`-like or '
                    '`callable`.'.format(dist_fn))

  args = _get_required_args(dist_fn)
  if not args:
    return (lambda *_: dist_fn()), ()

  @functools.wraps(dist_fn)
  def dist_fn_wrapped(*xs):
    """Calls `dist_fn` with reversed and truncated args."""
    if i != len(xs):
      raise ValueError(
          'Internal Error: Unexpected number of inputs provided to {}-th '
          'distribution maker (dist_fn: {}, expected: {}, saw: {}).'.format(
              i, dist_fn, i, len(xs)))
    if len(xs) < len(args):
      raise ValueError(
          'Internal Error: Too few inputs provided to {}-th distribution maker '
          '(dist_fn: {}, expected: {}, saw: {}).'.format(
              i, dist_fn, len(args), len(xs)))
    return dist_fn(*reversed(xs[-len(args):]))
  return dist_fn_wrapped, args


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
      dist_names[i] = leaf_name if j == 0 else leaf_name + str(j)
      j += 1
  return tuple(dist_names)


def _get_required_args(fn):
  """Returns the distribution's required args."""
  argspec = tf_inspect.getfullargspec(fn)
  args = argspec.args
  if tf_inspect.isclass(fn):
    args = args[1:]  # Remove the `self` arg.
  if argspec.defaults:
    # Remove the args which have defaults. By convention we only feed
    # *required args*. This means some distributions must always be wrapped
    # with a `lambda`, e.g., `lambda logits: tfd.Bernoulli(logits=logits)`
    # or `lambda probs: tfd.Bernoulli(probs=probs)`.
    args = args[:-len(argspec.defaults)]
  return tuple(args)


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
  if len(d0._dist_fn_wrapped) != len(d1._dist_fn_wrapped):  # pylint: disable=protected-access
    raise ValueError(
        'Can only compute KL divergence between when each has the'
        'same number of component distributions.')
  if (not all(a is None for a in d0._dist_fn_args) or  # pylint: disable=protected-access
      not all(a is None for a in d1._dist_fn_args)):  # pylint: disable=protected-access
    raise ValueError(
        'Can only compute KL divergence when all distributions are '
        'independent.')
  with tf.name_scope(name or 'kl_jointseq_jointseq'):
    return sum(kullback_leibler.kl_divergence(d0_(), d1_())
               for d0_, d1_ in zip(d0._dist_fn_wrapped, d1._dist_fn_wrapped))  # pylint: disable=protected-access
