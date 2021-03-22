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
"""The `JointDistributionNamed` class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.internal import distribution_util


__all__ = [
    'JointDistributionNamed',
]


class JointDistributionNamed(
    joint_distribution_sequential.JointDistributionSequential):
  """Joint distribution parameterized by named distribution-making functions.

  This distribution enables both sampling and joint probability computation from
  a single model specification.

  A joint distribution is a collection of possibly interdependent distributions.
  Like `JointDistributionSequential`, `JointDistributionNamed` is parameterized
  by several distribution-making functions. Unlike `JointDistributionNamed`,
  each distribution-making function must have its own key. Additionally every
  distribution-making function's arguments must refer to only specified keys.

  #### Mathematical Details

  Internally `JointDistributionNamed` implements the chain rule of probability.
  That is, the probability function of a length-`d` vector `x` is,

  ```none
  p(x) = prod{ p(x[i] | x[:i]) : i = 0, ..., (d - 1) }
  ```

  The `JointDistributionNamed` is parameterized by a `dict` (or `namedtuple` or
  `collections.OrderedDict`) composed of either:

  1. `tfp.distributions.Distribution`-like instances or,
  2. `callable`s which return a `tfp.distributions.Distribution`-like instance.

  The "conditioned on" elements are represented by the `callable`'s required
  arguments; every argument must correspond to a key in the named
  distribution-making functions. Distribution-makers which are directly a
  `Distribution`-like instance are allowed for convenience and semantically
  identical a zero argument `callable`. When the maker takes no arguments it is
  preferable to directly provide the distribution instance.

  **Name resolution**: `The names of `JointDistributionNamed` components are
  simply the keys specified explicitly in the model definition.

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
  joint = tfd.JointDistributionNamed(dict(
      e=             tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),
      g=lambda    e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),
      n=             tfd.Normal(loc=0, scale=2.),
      m=lambda n, g: tfd.Normal(loc=n, scale=g),
      x=lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12),
  ))
  # Notice the 1:1 correspondence between "math" and "code". Further, notice
  # that unlike `JointDistributionSequential`, there is no need to put the
  # distribution-making functions in topologically sorted order nor is it ever
  # necessary to use dummy arguments to skip dependencies.

  x = joint.sample()
  # ==> A 5-element `dict` of Tensors representing a draw/realization from each
  #     distribution.
  joint.log_prob(x)
  # ==> A scalar `Tensor` representing the total log prob under all five
  #     distributions.

  joint.resolve_graph()
  # ==> (('e', ()),
  #      ('g', ('e',)),
  #      ('n', ()),
  #      ('m', ('n', 'g')),
  #      ('x', ('m',)))
  ```

  #### Discussion

  `JointDistributionNamed` topologically sorts the distribution-making functions
  and calls each by feeding in all previously created dependencies. A
  distribution-maker must either be a:

  1. `tfd.Distribution`-like instance (e.g., `e` and `n` in the above example),
  2. Python `callable` (e.g., `g`, `m`, `x` in the above example).

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
  distribution making function's *required arguments*. The distribution makers'
  arguments are parameterized by samples from the corresponding previously
  constructed distributions. ("Previous" in the sense of a topological sorting
  of dependencies.)

  **Note**: unlike other non-`JointDistribution` distributions in
  `tfp.distributions`, `JointDistribution.sample` (and subclasses) return a
  structure of  `Tensor`s rather than a `Tensor`.  A structure can be a `list`,
  `tuple`, `dict`, `collections.namedtuple`, etc. Accordingly
  `joint.batch_shape` returns a structure of `TensorShape`s for each of the
  distributions' batch shapes and `joint.batch_shape_tensor()` returns a
  structure of `Tensor`s for each of the distributions' event shapes. (Same with
  `event_shape` analogues.)

  **Note**: unlike other non-`JointDistribution` distributions in
  `tfp.distributions`, `JointDistributionNamed.sample` (and subclasses) return a
  structure of  `Tensor`s rather than a `Tensor`.  A structure can be anything
  which is convertible to `dict`. This can be a `dict`,
  `collections.namedtuple`, etc. Accordingly `joint.batch_shape` returns a
  structure of `TensorShape`s for each of the distributions' batch shapes and
  `joint.batch_shape_tensor()` returns a structure of `Tensor`s for each of the
  distributions' event shapes. (Same with `event_shape` analogues.)
  """

  def __init__(self, model, validate_args=False, name=None):
    """Construct the `JointDistributionNamed` distribution.

    Args:
      model: Python `dict`, `collections.OrderedDict`, or `namedtuple` of
        distribution-making functions each with required args corresponding
        only to other keys.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `"JointDistributionNamed"`).
    """
    super(JointDistributionNamed, self).__init__(
        model, validate_args, name or 'JointDistributionNamed')

  def _build(self, model):
    """Creates `dist_fn`, `dist_fn_wrapped`, `dist_fn_args`, `dist_fn_name`."""
    if not _is_dict_like(model):
      raise TypeError('`model` must be convertible to `dict` (saw: {}).'.format(
          type(model).__name__))
    [
        self._dist_fn,
        self._dist_fn_wrapped,
        self._dist_fn_args,
        self._dist_fn_name,  # JointDistributionSequential doesn't have this.
    ] = _prob_chain_rule_model_flatten(model)

  def _model_unflatten(self, xs):
    kwargs_list = zip(self._dist_fn_name, tuple(xs))
    if isinstance(self.model, collections.OrderedDict):
      return collections.OrderedDict(kwargs_list)
    return type(self.model)(**dict(kwargs_list))

  def _model_flatten(self, xs):
    if xs is None:
      return (None,) * len(self._dist_fn_name)
    if hasattr(xs, 'get'):
      return tuple(xs.get(n, None) for n in self._dist_fn_name)
    return tuple(getattr(xs, n) for n in self._dist_fn_name)

  def _flat_resolve_names(self, distribution_names=None, leaf_name='x'):
    return self._dist_fn_name

  _composite_tensor_nonshape_params = ('model',)


class _Node(object):

  def __init__(self, name, parents):
    self.name = name
    self.parents = parents
    self.depth = -1


def _depth(g):
  """Computes the number of edges on longest path from node to root."""
  def _explore(v):
    if v.depth < 0:
      v.depth = ((1 + max([-1] + [_explore(annotated_graph[u])
                                  for u in v.parents]))
                 if v.parents else 0)
    return v.depth
  annotated_graph = {k: _Node(k, v) for k, v in g.items()}
  for v in annotated_graph.values():
    _explore(v)
  return annotated_graph


def _best_order(g):
  """Creates tuple of str tuple-str pairs representing resolved & sorted DAG."""
  if isinstance(g, collections.OrderedDict):
    return g.items()

  def _explore(u):
    """Recursive function to ascend up through unvisited dependencies."""
    if u.depth < 0:
      return  # Already visited.
    if not u.parents:
      result.append((u.name, u.parents))
      u.depth = -1  # Mark visited.
      return
    u.depth = -1  # Mark visited.
    for v in sorted((g.get(p) for p in u.parents),
                    key=lambda v: (v.depth, v.name), reverse=True):
      _explore(v)
    result.append((u.name, u.parents))
  g = _depth(g)
  result = []
  for u in sorted(g.values(), key=lambda v: (v.depth, v.name), reverse=True):
    _explore(u)
  return tuple(result)


def _prob_chain_rule_model_flatten(named_makers):
  """Creates lists of callables suitable for JDSeq."""
  def _make(dist_fn, args):
    if args is None:
      return lambda *_: dist_fn
    if not args:
      return lambda *_: dist_fn()
    def _fn(*xs):
      kwargs = dict([(k, v) for k, v in zip(dist_fn_name, xs) if k in args])
      return dist_fn(**kwargs)
    return _fn
  named_makers = _convert_to_dict(named_makers)

  previous_keys = []
  parents = type(named_makers)()
  for key, dist_fn in named_makers.items():
    if distribution_util.is_distribution_instance(dist_fn):
      parents[key] = None   # pylint: disable=g-long-lambda
    else:
      parents[key] = joint_distribution_sequential._get_required_args(  # pylint: disable=protected-access
          dist_fn,
          # To ensure an acyclic dependence graph, a dist_fn that takes
          # `**kwargs` is treated as depending on all distributions that were
          # defined above it, but not any defined below it.
          previous_args=previous_keys)
    previous_keys.append(key)

  g = _best_order(parents)
  dist_fn_name, dist_fn_args = zip(*g)
  dist_fn_args = tuple(None if a is None else tuple(a) for a in dist_fn_args)
  dist_fn_wrapped = tuple(_make(named_makers[name], parents)
                          for (name, parents) in g)
  dist_fn = tuple(named_makers.get(n) for n in dist_fn_name)
  return dist_fn, dist_fn_wrapped, dist_fn_args, dist_fn_name


def _is_dict_like(x):
  """Returns `True` if input is convertible to `dict`, `False` otherwise."""
  return hasattr(x, '_asdict') or isinstance(x, collections.Mapping)


def _convert_to_dict(x):
  """Converts input to `dict`."""
  if isinstance(x, collections.OrderedDict):
    return x
  if hasattr(x, '_asdict'):
    # Wrap with `OrderedDict` to indicate that namedtuples have a well-defined
    # order (by default, they convert to just `dict` in Python 3.8+).
    return collections.OrderedDict(x._asdict())
  return dict(x)
