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

import collections
import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import joint_distribution_sequential
from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import distribution_util


__all__ = [
    'JointDistributionNamed',
]

JAX_MODE = False


class _JointDistributionNamed(
    joint_distribution_sequential._JointDistributionSequential):  # pylint: disable=protected-access
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

  Consider the following generative model:

  ```
  e ~ Exponential(rate=[100,120])
  g ~ Gamma(concentration=e[0], rate=e[1])
  n ~ Normal(loc=0, scale=2.)
  m ~ Normal(loc=n, scale=g)
  for i = 1, ..., 12:
    x[i] ~ Bernoulli(logits=m)
  ```

  We can code this as:

  ```python
  tfd = tfp.distributions
  joint = tfd.JointDistributionNamed(dict(
      e=             tfd.Exponential(rate=[100, 120]),
      g=lambda    e: tfd.Gamma(concentration=e[0], rate=e[1]),
      n=             tfd.Normal(loc=0, scale=2.),
      m=lambda n, g: tfd.Normal(loc=n, scale=g),
      x=lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12)
    ),
    batch_ndims=0,
    use_vectorized_map=True)
  ```

  Notice the 1:1 correspondence between "math" and "code". Further, notice
  that unlike `JointDistributionSequential`, there is no need to put the
  distribution-making functions in topologically sorted order nor is it ever
  necessary to use dummy arguments to skip dependencies.

  ```python
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
  `tfp.distributions`, `JointDistributionNamed.sample` (and subclasses) return a
  structure of  `Tensor`s rather than a `Tensor`.  A structure can be anything
  which is convertible to `dict`. This can be a `dict`,
  `collections.namedtuple`, etc. Accordingly `joint.event_shape` returns a
  structure of `TensorShape`s for each of the distributions' batch shapes and
  `joint.event_shape_tensor()` returns a structure of `Tensor`s for each of the
  distributions' event shapes.

  **Note**: If a `JointDistributionNamed` instance contains a `callable` that
  closes over a `Tensor`, the `JointDistributionNamed` cannot cross the boundary
  of a `tf.function`. (If this behavior is necessary, an instance of
  `_JointDistributionNamed` may be used instead, at the expense of extra
  `tf.function` retracing.)

  #### Vectorized sampling and model evaluation

  When a joint distribution's `sample` method  is called with
  a `sample_shape` (or the `log_prob` method is called on an input with
  multiple sample dimensions) the model must be equipped to handle
  additional batch dimensions. This may be done manually, or automatically
  by passing `use_vectorized_map=True`. Manual vectorization has historically
  been the default, but we now recommend that most users enable automatic
  vectorization unless they are affected by a specific issue; some
  known issues are listed below.

  When using manually-vectorized joint distributions, each operation in the
  model must account for the possibility of batch dimensions in Distributions
  and their samples. By contrast, auto-vectorized models need only describe
  a *single* sample from the joint distribution; any batch evaluation is
  automated as required using `tf.vectorized_map` (`vmap` in JAX). In many
  cases this allows for significant simplications. For example, the following
  manually-vectorized `tfd.JointDistributionSequential` model:

  ```python
  model = tfd.JointDistributionSequential([
      tfd.Normal(0., tf.ones([3])),
      tfd.Normal(0., 1.),
      lambda y, x: tfd.Normal(x[..., :2] + y[..., tf.newaxis], 1.)
    ])
  ```

  can be written in auto-vectorized form as

  ```python
  model = tfd.JointDistributionSequential([
      tfd.Normal(0., tf.ones([3])),
      tfd.Normal(0., 1.),
      lambda y, x: tfd.Normal(x[:2] + y, 1.)
    ],
    use_vectorized_map=True)
  ```

  in which we were able to avoid explicitly accounting for batch dimensions
  when indexing and slicing computed quantities in the third line.

  **Known limitations of automatic vectorization:**
  - A small fraction of TensorFlow ops are unsupported; models that use an
    unsupported op will raise an error and must be manually vectorized.
  - Sampling large batches may be slow under automatic vectorization because
    TensorFlow's stateless samplers are currently converted using a
    non-vectorized `while_loop`. This limitation applies only in TensorFlow;
    vectorized samplers in JAX should be approximately as fast as manually
    vectorized code.
  - Calling `sample_distributions` with nontrivial `sample_shape` will raise
    an error if the model contains any distributions that are not registered as
    CompositeTensors (TFP's basic distributions are usually fine, but support
    for wrapper distributions like `tfd.Sample` is a work in progress).

  #### Batch semantics and (log-)densities

  **tl;dr:** pass `batch_ndims=0` unless you have a good reason not to.

  Joint distributions now support 'auto-batching' semantics, in which
  the distribution's batch shape is derived by broadcasting the leftmost
  `batch_ndims` dimensions of its components' batch shapes. All remaining
  dimensions are considered to form a single 'event' of the joint distribution.
  If `batch_ndims==0`, then the joint distribution has batch shape `[]`, and all
  component dimensions are treated as event shape. For example, the model

  ```python
  jd = tfd.JointDistributionSequential([
      tfd.Normal(0., tf.ones([3])),
      lambda x: tfd.Normal(x[..., tf.newaxis], tf.ones([3, 2]))
    ],
    batch_ndims=0)
  ```

  creates a joint distribution with batch shape `[]` and event shape
  `([3], [3, 2])`. The log-density of a sample always has shape
  `batch_shape`, so this guarantees that
  `jd.log_prob(jd.sample())` will evaluate to a scalar value. We could
  alternately construct a joint distribution with batch shape `[3]` and event
  shape `([], [2])` by setting `batch_ndims=1`, in which case
  `jd.log_prob(jd.sample())` would evaluate to a value of shape `[3]`.

  Setting `batch_ndims=None` recovers the 'classic' batch semantics (currently
  still the default for backwards-compatibility reasons), in which the joint
  distribution's `log_prob` is computed by naively summing log densities from
  the component distributions. Since these component densities have shapes equal
  to the batch shapes of the individual components, to avoid broadcasting
  errors it is usually necessary to construct the components with identical
  batch shapes. For example, the component distributions in the model above
  have batch shapes of `[3]` and `[3, 2]` respectively, which would raise an
  error if summed directly, but can be aligned by wrapping with
  `tfd.Independent`, as in this model:

  ```python
  jd = tfd.JointDistributionSequential([
      tfd.Normal(0., tf.ones([3])),
      lambda x: tfd.Independent(tfd.Normal(x[..., tf.newaxis], tf.ones([3, 2])),
                                reinterpreted_batch_ndims=1)
    ],
    batch_ndims=None)
  ```

  Here the components both have batch shape `[3]`, so
  `jd.log_prob(jd.sample())` returns a value of shape `[3]`, just as in the
  `batch_ndims=1` case above. In fact, auto-batching semantics are equivalent to
  implicitly wrapping each component `dist` as `tfd.Independent(dist,
  reinterpreted_batch_ndim=(dist.batch_shape.ndims - jd.batch_ndims))`; the only
  vestigial difference is that under auto-batching semantics, the joint
  distribution has a single batch shape `[3]`, while under the classic semantics
  the value of `jd.batch_shape` is a *structure* of the component batch shapes
  `([3], [3])`. Such structured batch shapes will be deprecated in the future,
  since they are inconsistent with the definition of batch shapes used
  elsewhere in TFP.

  #### References

  [1] Dan Piponi, Dave Moore, and Joshua V. Dillon. Joint distributions for
      TensorFlow Probability. _arXiv preprint arXiv:2001.11819__,
      2020. https://arxiv.org/abs/2001.11819

  """

  def __init__(self,
               model,
               batch_ndims=None,
               use_vectorized_map=False,
               validate_args=False,
               experimental_use_kahan_sum=False,
               name=None):
    """Construct the `JointDistributionNamed` distribution.

    Args:
      model: Python `dict`, `collections.OrderedDict`, or `namedtuple` of
        distribution-making functions each with required args corresponding
        only to other keys.
      batch_ndims: `int` `Tensor` number of batch dimensions. The `batch_shape`s
        of all component distributions must be such that the prefixes of
        length `batch_ndims` broadcast to a consistent joint batch shape.
        Default value: `None`.
      use_vectorized_map: Python `bool`. Whether to use `tf.vectorized_map`
        to automatically vectorize evaluation of the model. This allows the
        model specification to focus on drawing a single sample, which is often
        simpler, but some ops may not be supported.
        Default value: `False`.
      validate_args: Python `bool`.  Whether to validate input with asserts.
        If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
        Default value: `False`.
      experimental_use_kahan_sum: Python `bool`. When `True`, we use Kahan
        summation to aggregate independent underlying log_prob values, which
        improves against the precision of a naive float32 sum. This can be
        noticeable in particular for large dimensions in float32. See CPU caveat
        on `tfp.math.reduce_kahan_sum`. This argument has no effect if
        `batch_ndims is None`.
        Default value: `False`.
      name: The name for ops managed by the distribution.
        Default value: `None` (i.e., `"JointDistributionNamed"`).
    """
    super(_JointDistributionNamed, self).__init__(
        model,
        batch_ndims=batch_ndims,
        use_vectorized_map=use_vectorized_map,
        validate_args=validate_args,
        experimental_use_kahan_sum=experimental_use_kahan_sum,
        name=name or 'JointDistributionNamed')

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
  _composite_tensor_shape_params = ()


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
  return hasattr(x, '_asdict') or isinstance(x, collections.abc.Mapping)


def _convert_to_dict(x):
  """Converts input to `dict`."""
  if isinstance(x, collections.OrderedDict):
    return x
  if hasattr(x, '_asdict'):
    # Wrap with `OrderedDict` to indicate that namedtuples have a well-defined
    # order (by default, they convert to just `dict` in Python 3.8+).
    return collections.OrderedDict(x._asdict())
  return dict(x)


class JointDistributionNamed(_JointDistributionNamed,
                             tf.__internal__.CompositeTensor):

  def __new__(cls, *args, **kwargs):
    """Returns a `_JointDistributionNamed` if `model` contains non-CT dists."""
    if cls is JointDistributionNamed:
      if args:
        model = args[0]
      else:
        model = kwargs.get('model')

      if not all(auto_composite_tensor.is_composite_tensor(d)
                 or callable(d) for d in tf.nest.flatten(model)):
        return _JointDistributionNamed(*args, **kwargs)
    return super(JointDistributionNamed, cls).__new__(cls)

  @property
  def _type_spec(self):
    return _JointDistributionNamedSpec.from_instance(self)

  def _convert_variables_to_tensors(self):
    return auto_composite_tensor.convert_variables_to_tensors(self)


def _unflatten_model(components, structure_with_callables):
  model_components = []
  i = 0
  for c in tf.nest.flatten(structure_with_callables):
    if c is None:
      model_components.append(components['model'][i])
      i += 1
    else:
      model_components.append(c)
  return tf.nest.pack_sequence_as(structure_with_callables, model_components)


@auto_composite_tensor.type_spec_register(
    'tfp.distributions.JointDistributionNamedSpec')
class _JointDistributionNamedSpec(
    auto_composite_tensor._AutoCompositeTensorTypeSpec):  # pylint: disable=protected-access
  """Type spec for `JointDistributionNamed`."""

  @property
  def value_type(self):
    return JointDistributionNamed

  def _to_components(self, obj):
    if self._callable_params:
      components = []
      for d in tf.nest.flatten(obj.model):
        if auto_composite_tensor.is_composite_tensor(d):
          components.append(d)
    else:
      components = obj.model
    return dict(model=components)

  def _from_components(self, components):
    if self._callable_params:
      model = _unflatten_model(components, self._structure_with_callables)
    else:
      model = components['model']
    return self.value_type(model, **self._non_tensor_params)

  @classmethod
  def from_instance(cls, obj):
    model_param_specs, callable_model_params = [], []
    for d in tf.nest.flatten(obj.model):
      if auto_composite_tensor.is_composite_tensor(d):
        model_param_specs.append(d._type_spec)  # pylint: disable=protected-access
      else:
        callable_model_params.append(d)
    non_tensor_params = {k: v for k, v in obj.parameters.items()
                         if k != 'model'}

    if callable_model_params:
      callable_params = dict(model=callable_model_params)
      param_specs = dict(model=model_param_specs)
    else:
      callable_params = None
      param_specs = dict(
          model=tf.nest.pack_sequence_as(obj.model, model_param_specs))

    spec = cls(
        param_specs=param_specs,
        non_tensor_params=non_tensor_params,
        non_identifying_kwargs=('name',),
        omit_kwargs=('parameters',),
        prefer_static_value=('batch_ndims',),
        callable_params=callable_params)

    if callable_params:
      # Store the nested structure of `model` so that it can be reconstituted in
      # `_from_components`. If the typespec is built by `_deserialize`, this
      # attribute will not exist -- however, the type spec serializable only if
      # there are no callable elements of `model`, in which case the nested
      # structure of `model` is recorded in `param_specs`.
      structure_with_callables = tf.nest.map_structure(
          lambda x: None if auto_composite_tensor.is_composite_tensor(x) else x,
          obj.model)
      spec._structure_with_callables = structure_with_callables
    return spec


def _pytree_flatten(obj):
  """Flatten method for JAX pytrees."""
  # pylint: disable=protected-access
  components = obj._type_spec._to_components(obj)
  if components:
    keys, values = zip(*components.items())
  else:
    keys, values = (), ()
  metadata = dict(
      non_tensor_params=obj._type_spec._non_tensor_params,
      structure_with_callables=obj._type_spec._structure_with_callables)
  return values, (keys, metadata)


def _pytree_unflatten(cls, aux_data, children):
  keys, metadata = aux_data
  model_dists = dict(list(zip(keys, children)))
  model = _unflatten_model(model_dists, metadata['structure_with_callables'])
  return cls(model, **metadata['non_tensor_params'])


if JAX_MODE:
  from jax import tree_util  # pylint: disable=g-import-not-at-top
  tree_util.register_pytree_node(
      JointDistributionNamed,
      _pytree_flatten,
      functools.partial(_pytree_unflatten, JointDistributionNamed))


JointDistributionNamed.__doc__ = _JointDistributionNamed.__doc__ + (
    '\nIf every element of `model` is a `CompositeTensor` or a callable, the '
    'resulting `JointDistributionNamed` is a `CompositeTensor`. Otherwise, '
    'a non-`CompositeTensor` `_JointDistributionNamed` instance is created.')
