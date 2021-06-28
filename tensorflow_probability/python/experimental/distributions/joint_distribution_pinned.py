# Copyright 2020 The TensorFlow Probability Authors.
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
"""A partially pinned, unnormalized JointDistribution-like object."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.distributions import distribution as distribution_lib
from tensorflow_probability.python.distributions import joint_distribution
from tensorflow_probability.python.distributions import joint_distribution_vmap_mixin
from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import structural_tuple
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import vectorization_util
from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

CALLING_CONVENTION_DESCRIPTION = """
The methods of `JointDistributionPinned` (`unnormalized_log_prob`,
`sample_and_log_weight`, etc.) can be called by passing a single structure
of tensors, a sequence of tensor arguments, or using named args for each
part. For example:

```python
tfde = tfp.experimental.distributions

# Given the following joint distribution:
jd = tfd.JointDistributionSequential([
    tfd.Normal(0., 1., name='z'),
    tfd.Normal(0., 1., name='y'),
    lambda y, z: tfd.Normal(y + z, 1., name='x')
], validate_args=True)

# The following `__init__` styles are all permissible and produce
# `JointDistributionPinned` objects behaving identically.
PartialXY = collections.namedtuple('PartialXY', 'x,y')
PartialX = collections.namedtuple('PartialX', 'x')
OrderedDict = collections.OrderedDict
assert (tfde.JointDistributionPinned(jd, x=2.).pins ==
        tfde.JointDistributionPinned(jd, x=2., z=None).pins ==
        tfde.JointDistributionPinned(jd, dict(x=2.)).pins ==
        tfde.JointDistributionPinned(jd, dict(x=2., y=None)).pins ==
        tfde.JointDistributionPinned(jd, OrderedDict(x=2.)).pins ==
        tfde.JointDistributionPinned(jd, OrderedDict(x=2., y=None)).pins ==
        tfde.JointDistributionPinned(jd, PartialXY(x=2., y=None)).pins ==
        tfde.JointDistributionPinned(jd, PartialX(x=2.)).pins ==
        tfde.JointDistributionPinned(jd, None, None, 2.).pins ==
        tfde.JointDistributionPinned(jd, [None, None, 2.]).pins)
# (Notice that the `pins` attribute is always resolved to a `dict`.)

pinned = tfde.JointDistributionPinned(jd, x=2.)
pinned.dtype
# ==> [tf.float32, tf.float32]
z, y = sample = pinned.sample_unpinned()

# The following calling styles are all permissable and produce the exactly
# the same output.
PartialZY = collections.namedtuple('PartialZY', 'z,y')
assert (pinned.{method}(sample) ==
        pinned.{method}(z, y) ==
        pinned.{method}(z=z, y=y) ==
        pinned.{method}(PartialZY(z=z, y=y)))

# These calling possibilities also imply that one can also use `*`
# expansion, if `sample` is a sequence:
pinned.{method}(*sample)
# and similarly, if `sample` is a map, one can use `**` expansion:
pinned.{method}(**sample)
```

Component distributions' names are resolved via `jd._flat_resolve_names()`,
which is implemented by each `JointDistribution` subclass (see subclass
documentation for details). Generally, for components where a name was
provided---either explicitly as the `name` argument to a distribution or as
a key in a dict-valued JointDistribution, or implicitly, e.g., by the
argument name of a `JointDistributionSequential` distribution-making
function---the provided name will be used. Otherwise the component will
receive a dummy name; these may change without warning and should not be
relied upon.

In general, return types of part-wise methods/properties are determined by
those of the underlying `JointDistribution`'s model type:

- `StructTuple` for `JointDistributionCoroutine`, and for
  `JointDistributionNamed` with `namedtuple` model type.
- `collections.OrderedDict` for `JointDistributionNamed` with `OrderedDict`
  model type.
- `dict` for `JointDistributionNamed` with `dict` model type.
- `tuple` or `list` for `JointDistributionSequential`.

Note: not all `JointDistribution` subclasses support all calling styles;
for example, `JointDistributionNamed` does not support positional arguments
(aka "unnamed arguments") unless the provided model specifies an ordering of
variables (i.e., is an `collections.OrderedDict` or `collections.namedtuple`
rather than a plain `dict`). In the same way, JointDistributionPinned does
not accept unnamed pins for unordered `JointDistributionNamed` models.

Note: care is taken to resolve any potential ambiguity---this is generally
possible by inspecting the structure of the provided argument and "aligning"
it to the joint distribution output structure (defined by `jd.dtype`). For
example,

```python
pinned = tfde.JointDistributionPinned(
    tfd.JointDistributionSequential(
        [tfd.Exponential(1.), lambda s: tfd.Normal(0., s)]),
        None, 1.2)
pinned.dtype  # => [tf.float32]
pinned.{method}([4.])
# ==> Tensor with shape `[]`.
{method_abbr} = pinned.{method}(4.)
# ==> Tensor with shape `[]`.
```

Notice that in the first call, `[4.]` is interpreted as a list of one
scalar while in the second call the input is a scalar. Hence both inputs
result in identical scalar outputs. If we wanted to pass an explicit
vector to the `Exponential` component---creating a vector-shaped batch
of `{method}`s---we could instead write
`pinned.{method}(np.array([4]))`.

Args:
  *args: Positional arguments: a value structure or component values
    (see above).
  **kwargs: Keyword arguments: a value structure or component values
    (see above). May also include `name`, specifying a Python string name
    for ops generated by this method.
"""

UnnormalizedLogProbParts = collections.namedtuple(
    'UnnormalizedLogProbParts', ('pinned', 'unpinned'))

# pylint: disable=protected-access


class JointDistributionPinned(object):
  """A wrapper class for `JointDistribution` which pins, e.g., the evidence.

  This object is experimental; the API may change without warning.

  Think of this object as `functools.partial` for joint distributions. Sampling
  trims off pinned values (after specifying them as `jd.sample(value=pins)` to
  the underlying distribution). Log-density evaluates the joint probability of
  the given event and the pinned values.

  This object represents an unnormalized probability density, and as such is
  not a `tfp.distributions.Distribution`, and lacks `sample` and `log_prob`
  methods. In their place, it provides:

  * `unnormalized_log_prob`, `unnormalized_log_prob_parts`
  * `sample_unpinned`, `sample_and_log_weight`

  Mathematically speaking, the object represents a joint probability density,
  `p(x, y)` where the `x` are pinned and the `y` are unpinned. Accordingly, it
  is also proportional to `p(y | x)`, up to a (generally) intractable
  normalizing constant `p(x)`, i.e. `p(x, y) = p(y | x) p(x)`.

  A common use-case with probabilistic inference is writing out a generative
  model to explain some observed data:

  ```python
  jd = tfd.JointDistributionNamed(dict(
    loc = tfd.Normal(0., 1.),
    scale = tfd.Gamma(1., 1.),
    obs = lambda loc, scale: tfd.Normal(loc, scale),
  ))
  ```

  Later, when we want to infer 'typical' values of `loc` and `scale` conditioned
  on some given `data`, we will often write:

  ```python
  def target_log_prob_fn(loc, scale):
    return jd.log_prob(loc=loc, scale=scale, obs=data)
  ```

  This class enables one to write instead:

  ```python
  partial = tfde.JointDistributionPinned(jd, obs=data)
  target_log_prob_fn = partial.unnormalized_log_prob
  ```

  Or, even more concisely `partial = jd.experimental_pin(obs=data)`.

  This is nice, but it wasn't too hard to write out the `target_log_prob_fn`
  function explicitly.

  Now, let's consider that for many inference and optimization methods, we may
  want to use a smooth change of variables to perform inference in the
  unconstrained space of real numbers. In some cases this transformation can be
  parameter-dependent. For example, if we want to unconstrain the support of
  `tfp.distributions.Uniform(-3., 2.)` to the real line, we might use
  `tfp.bijectors.Sigmoid(low=-3., high=2.)`. In support of such use cases,
  most distributions (including the `JointDistribution*` classes) provide a
  `experimental_default_event_space_bijector()` method.

  When these transformations may be dependent on ancestral parts of a joint
  distribution, and some of those parameters may be pinned, it is helpful to
  have a utility class to bridge the gap and provide the multi-part bijective
  transform. This is the "raison d'etre" of this class.

  The model below is somewhat contrived, but demonstrates the use-case.

  ```python
  tfd = tfp.distributions
  tfde = tfp.experimental.distributions

  n = 75
  dim = 3
  joint = tfd.JointDistributionNamed(dict(
    upper = tfd.Uniform(.4, 1.5),
    concentration = tfd.Gamma(1., .5),
    corr = lambda concentration: tfd.CholeskyLKJ(
        dim, concentration=concentration),
    stddev = lambda upper: tfd.Sample(tfd.Uniform(.2, upper), dim),
    obs = lambda corr, stddev: tfd.Sample(
        tfd.MultivariateNormalTriL(
            loc=tf.zeros([dim]), scale_tril=corr * stddev[..., tf.newaxis]),
        n)
  ))
  fixed_upper = 1.3
  data = joint.sample(upper=fixed_upper)['obs']

  pinned = tfde.JointDistributionPinned(joint, upper=fixed_upper, obs=data)
  bij = pinned.experimental_default_event_space_bijector()
  pulled_back_shape = bij.inverse_event_shape(pinned.event_shape)

  # Fit an ensemble using SGD.
  batch = 16
  uniform_init = tf.nest.map_structure(
      lambda s: tf.random.uniform(tf.concat([[batch], s], axis=0), -2., 2.),
      pulled_back_shape)
  vars = tf.nest.map_structure(tf.Variable, uniform_init)

  opt = tf.optimizers.Adam(.01)

  @tf.function(autograph=False)
  def one_step():
    with tf.GradientTape() as tape:
      lp = pinned.unnormalized_log_prob(bij.forward(vars))
    gradients = tape.gradient(lp, vars)
    opt.apply_gradients(zip(gradients.values(), vars.values()))

  for _ in range(100):
    one_step()

  # Alternatively, sample using MCMC (currently aspirational):
  initial_state = bij.forward(uniform_init)

  kernel = tfp.mcmc.HamiltonianMonteCarlo(
      target_log_prob_fn=pinned.unnormalized_log_prob,
      step_size=.5, num_leapfrog_steps=4)
  # **This line is currently aspirational**, to demonstrate the use-case.
  kernel = tfp.mcmc.TransformedTransitionKernel(kernel, bij)
  tfp.mcmc.sample_chain(10, kernel=kernel, current_state=initial_state)
  ```
  """

  def __init__(self, distribution, *pins, name=None, **named_pins):
    """Constructs a `JointDistributionPinned`.

    ### Examples:

    ```
    JointDistributionPinned(
       tfd.JointDistributionSequential([
           tfd.Gamma(1., 1.), lambda scale: tfd.Normal(0., scale, name='x')]),
       1.)  # Pins the `scale` part.

    JointDistributionPinned(
       tfd.JointDistributionSequential([
           tfd.Gamma(1., 1.), lambda scale: tfd.Normal(0., scale, name='x')]),
       [1.])  # Pins the `scale` part to scalar `1.`.

    JointDistributionPinned(
       tfd.JointDistributionSequential([
           tfd.Gamma(1., 1.), lambda scale: tfd.Normal(0., scale, name='x')]),
       None, 1.)  # Pins the `x` part.

    JointDistributionPinned(
       tfd.JointDistributionSequential([
           tfd.Gamma(1., 1.), lambda scale: tfd.Normal(0., scale, name='x')]),
       [None, 1.])  # Pins the `x` part.

    JointDistributionPinned(
       tfd.JointDistributionSequential([
           tfd.Gamma(1., 1.), lambda scale: tfd.Normal(0., scale, name='x')]),
       scale=1.)  # Pins the `scale` part.

    JointDistributionPinned(
       tfd.JointDistributionSequential([
           tfd.Gamma(1., 1.), lambda scale: tfd.Normal(0., scale, name='x')]),
       x=1.)  # Pins the `x` part.
    ```

    Args:
      distribution: A `tfp.distributions.JointDistribution`.
      *pins: A single object like the `value` argument that may be passed into
        `JointDistribution.sample` (some parts may be `None`), or a sequence of
        objects similar to such sequence as might be passed to
        `JointDistribution.log_prob`, but with the difference that some parts
        may be `None` (`log_prob` would require all parts be specified).
        More precisely, the user may pass (A) a single argument specifiying pins
        of one or more of the parts of the underlying `distribution` either by
        name (i.e. a `dict`, `namedtuple`) or by sequence ordering (`tuple`,
        `list`), or (B) a sequence of arguments which align with the model of
        the underlying distribution (which must be ordered). It is an error to
        use an unordered sequence of pins with an unordered model, e.g. a
        `tfp.distributions.JointDistributionNamed` constructed with a `dict`
        model (`collections.OrderedDict` is allowed).
      name: Python `str` name for this distribution. If `None`, defaults to
        'Pinned{distribution.name}'.
        Default value: `None`.
      **named_pins: Named elements to pin. The names given must align with the
        part names defined by `distribution._flat_resolve_names()`, i.e. either
        the explicitly named parts of `tfp.distributions.JointDistributionNamed`
        or the `name` parameters passed to distributions constructed by the
        model given to `JointDistribution*`.
    """
    if bool(pins) == bool(named_pins):
      raise ValueError('Exactly one of *pins or **named_pins should be set.')

    if name is None:
      name = 'Pinned{}'.format(distribution.name)

    self._distribution = distribution
    self._name = name
    self._pins = _to_pins(distribution, *pins, **named_pins)
    self._use_vectorized_map = getattr(distribution,
                                       'use_vectorized_map',
                                       False)

  @property
  def distribution(self):
    """The underlying distribution being partially pinned."""
    return self._distribution

  @property
  def name(self):
    """Name of this pinned distribution."""
    return self._name

  @property
  def pins(self):
    """Dictionary of pins resolved to names."""
    return self._pins

  @property
  def use_vectorized_map(self):
    """Whether the underlying distribution relies on automatic vectorization."""
    return self._use_vectorized_map

  @property
  def validate_args(self):
    # Used by DefaultJointBijector.
    return self._distribution.validate_args

  def _model_flatten(self, x):
    """Flattens `x` to a tuple."""
    if isinstance(x, dict):
      return tuple(x[n] for n in self._flat_resolve_names())
    return tuple(x)

  def _model_unflatten(self, xs):
    """Unflattens `xs` to a structure like-typed to `self.distribution`."""
    # Use the underlying JD dtype to infer model structure.
    dtype = self.distribution.dtype
    if isinstance(dtype, dict):
      ks = self._flat_resolve_names()
      if len(ks) != len(xs):
        raise ValueError('Invalid xs length {}, ks={}'.format(len(xs), ks))
      return type(dtype)(zip(ks, xs))
    if hasattr(dtype, '_fields') and hasattr(dtype, '_asdict'):
      ks = [k for k in dtype._fields if k not in self.pins]
      return structural_tuple.structtuple(ks)(*xs)
    return type(dtype)(xs)

  def _prune(self, xs, retain=None):
    """Drops fields from `xs`, retaining those specified by `retain`.

    Args:
      xs: A structure like that of `self.distribution.dtype`
      retain: One of `'pinned'` or `'unpinned'`.

    Returns:
      xs: Input `xs`, pruned to retain only the parts specified by `retain`.
    """
    if retain not in ('pinned', 'unpinned'):
      raise ValueError('Invalid value for `retain`: {}'.format(retain))
    def should_retain(k):
      return (k not in self.pins) ^ (retain == 'pinned')
    if isinstance(xs, dict):
      return type(xs)((k, v) for k, v in xs.items() if should_retain(k))
    if hasattr(xs, '_fields') and hasattr(xs, '_asdict'):
      tuple_type = structural_tuple.structtuple(
          [k for k in xs._fields if should_retain(k)])
      return tuple_type(
          **{k: v for k, v in xs._asdict().items() if should_retain(k)})
    names = self.distribution._flat_resolve_names()
    return type(xs)([x for i, x in enumerate(xs) if should_retain(names[i])])

  def _add_pins(self, **kwargs):
    """Adds pinned values to those specified by `**kwargs`."""
    if any(k in self.pins for k in kwargs):
      raise ValueError('Value[s] of {} are already pinned.'.format(
          [k for k in kwargs if k in self.pins]))
    # We rely on _model_unflatten to report any missing fields.
    x = dict(kwargs, **self.pins)
    return self.distribution._model_unflatten(
        [x[n] for n in self.distribution._flat_resolve_names()])

  @property
  def dtype(self):
    """DType of unpinned parts."""
    return tf.nest.map_structure(
        tf.as_dtype,
        self._prune(self.distribution.dtype, retain='unpinned'))

  @property
  def event_shape(self):
    """Statically resolvable event shapes of unpinned parts."""
    return self._prune(self.distribution.event_shape, retain='unpinned')

  def event_shape_tensor(self):
    """Dynamic/graph Tensor event shapes of unpinned parts."""
    return self._prune(self.distribution.event_shape_tensor(),
                       retain='unpinned')

  @property
  def batch_shape(self):
    batch_shape = self.distribution.batch_shape
    if tf.nest.is_nested(batch_shape):
      return self._prune(batch_shape, retain='unpinned')
    return batch_shape

  def batch_shape_tensor(self):
    batch_shape = self.distribution.batch_shape_tensor()
    if tf.nest.is_nested(batch_shape):
      return self._prune(batch_shape, retain='unpinned')
    return batch_shape

  __str__ = distribution_lib.Distribution.__str__
  __repr__ = distribution_lib.Distribution.__repr__

  def experimental_default_event_space_bijector(self, *args, **kwargs):
    """A bijector to pull back unpinned values to unconstrained reals."""
    if args or kwargs:
      return (
          self.experimental_pin(*args, **kwargs)
          .experimental_default_event_space_bijector())
    if self.use_vectorized_map:
      return _DefaultJointBijectorAutoBatchedWithPins(self)
    return joint_distribution._DefaultJointBijector(self)

  def experimental_pin(self, *args, **kwargs):
    """Logical equivalent of `JointDistribution.experimental_pin`.

    For example
    ```
    @tfd.JointDistributionCoroutine
    def model():
        x = yield tfd.Normal(0, 1, name='x'),
        y = yield tfd.Normal(0, 1, name='y'),
        yield tfd.Normal(0, 1, name='z')
    model.experimental_pin(z=1.).experimental_pin(y=.5).event_shape
    # => StructTuple(x=[])
    ```

    Args:
      *args: Positional arguments: a value structure or component values.
      **kwargs: Keyword arguments: a value structure or component values.
        May also include `name`, specifying a Python string name for ops
        generated by this method.

    Returns:
      pinned: a `tfp.experimental.distributions.JointDistributionPinned` with
        the given values pinned in addition to those pins already specified on
        `self`.

    """
    pins = dict(self.pins, **_to_pins(self, *args, **kwargs))
    return JointDistributionPinned(self.distribution, **pins)

  def _flat_resolve_names(self):
    return [n for n in self.distribution._flat_resolve_names()
            if n not in self.pins]

  def sample_unpinned(self, sample_shape=(), seed=None):
    """Draws unnormalized samples using ancestral sampling.

    Conceptually, this is comparable to calling `underlying.sample(value=pins)`,
    then stripping away the pinned parts.

    Args:
      sample_shape: Shape prefix to use when sampling.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      samples: unpinned parts sampled from the underlying distribution.
    """
    return self._prune(
        self.distribution.sample(sample_shape, seed=seed, **self.pins),
        retain='unpinned')

  def sample_and_log_weight(self, sample_shape=(), seed=None):
    """Draws unnormalized samples and their log-weights with ancestral sampling.

    Since this object represents an unnormalized density, we are unable to
    directly sample the distribution. However, we can evaluate the relative
    density of different samples. This function returns the relative log-weight
    alongside the sample. This log-weight is the log-probability of the pinned
    parts at the sampled location (it differs from `unnormalized_log_prob` by
    the log-probability of the unpinned parts).

    Args:
      sample_shape: Shape prefix to use when sampling.
      seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

    Returns:
      samples: unpinned parts drawn from the pinned distribution.
      log_weights: log-weight of the sample. (Log-probability of the pinned
        parts at the sampled location.)
    """
    sample = self.sample_unpinned(sample_shape, seed=seed)
    return sample, self.log_weight(sample)

  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='log_weight', method_abbr='log_wt'))
  def log_weight(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Computes the log relative weight of the given sample.

    This function computes the log-probability of the pinned parts at the given
    location, ignoring the probability of the unpinned parts.

    ${calling_convention_description}

    Returns:
      log_weights: log-weight of the given point, i.e. the log pinned evidence.
    """
    pin_probs = self.unnormalized_log_prob_parts(*args, **kwargs).pinned
    return sum(  # Sum uses +, which broadcasts
        pin_probs.values() if isinstance(pin_probs, dict) else pin_probs)

  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='unnormalized_log_prob', method_abbr='lp'))
  def unnormalized_log_prob(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Computes the unnormalized log-probability.

    ${calling_convention_description}

    Returns:
      unnormalized_log_prob: The joint log-probability of `*xs` or `**kwargs`
        with the pinned parts. It is unnormalized with respect to `*xs` or
        `**kwargs`.
    """
    xs = _to_pins(self, *args, **kwargs)
    return self.distribution.unnormalized_log_prob(self._add_pins(**xs))

  @docstring_util.expand_docstring(
      calling_convention_description=CALLING_CONVENTION_DESCRIPTION.format(
          method='unnormalized_log_prob_parts', method_abbr='lp_parts'))
  def unnormalized_log_prob_parts(self, *args, **kwargs):  # pylint: disable=g-doc-args
    """Computes the unnormalized log-probability of each part.

    ${calling_convention_description}

    Returns:
      pinned: partial log-prob of each pinned part
      unpinned: partial log-prob of each unpinned part
    """
    xs = _to_pins(self, *args, **kwargs)
    parts = self.distribution.unnormalized_log_prob_parts(self._add_pins(**xs))
    return UnnormalizedLogProbParts(
        pinned=self._prune(parts, retain='pinned'),
        unpinned=self._prune(parts, retain='unpinned'))

  def _get_single_sample_distributions(self):
    """Helper for _DefaultJointBijector.{forward/inverse}_event_shape."""
    names = self.distribution._flat_resolve_names()
    ds = self.distribution._get_single_sample_distributions()
    return tuple([d for i, d in enumerate(ds) if names[i] not in self.pins])

  def _model_coroutine(self):
    """Coroutines with _DefaultJointBijector.{forward/inverse}."""
    names = self.distribution._flat_resolve_names()
    gen = self.distribution._model_coroutine()
    d = next(gen)
    index = 0
    try:
      while True:
        while names[index] in self.pins:
          d = gen.send(self.pins[names[index]])
          index += 1
        y = yield d
        d = gen.send(y)
        index += 1
    except StopIteration:
      pass


def _to_pins(dist, *args, **kwargs):
  """Converts *args and **kwargs to a dict of pins.

  Args:
    dist: JointDistribution*-like object with _flat_resolve_names(), dtype, and
      _model_flatten(x).
    *args: Either a sequence of pins that aligns with `_model_flatten` and
      `_flat_resolve_names`, or a single item sequence where `pins[0]`
      structure is compatible with `dist.dtype`.
    **kwargs: Named pins with keys corresponding to part names resolved by
      `_flat_resolve_names()`.

  Returns:
    pins: Python dict mapping resolved names to pinned values.
  """
  forbid_sequences = (
      isinstance(dist.dtype, dict) and
      not isinstance(dist.dtype, collections.OrderedDict))
  if bool(args) == bool(kwargs):
    raise ValueError('Exactly one of *args or **kwargs should be set.')
  dtypes = dist._model_flatten(dist.dtype)
  struct0 = dtypes[0]
  if len(args) == 1:
    # We can interpret a single-element *pins as either a value (matching the
    # structure of dist.dtype [perhaps partially]), or a first-item pin.
    try:
      tf.nest.assert_same_structure(args[0], struct0)
    except (ValueError, TypeError):
      args = args[0]
      if isinstance(args, dict):
        kwargs = args
      elif hasattr(args, '_asdict'):
        kwargs = args._asdict()
  if forbid_sequences and not kwargs:
    raise ValueError(
        'Must provide named pins for unordered dict model type.')
  names = dist._flat_resolve_names()
  dtypes_by_name = dict(zip(names, dtypes))

  def convert_to_tensors(xs, dtype):
    def convert_with_hint(x, dtype):
      return tensor_util.convert_nonref_to_tensor(x, dtype_hint=dtype)
    return nest.map_structure_up_to(
        dtype,  # shallow_tree
        convert_with_hint,  # func
        xs,  # x
        dtype)  # dtype

  if kwargs:
    unexpected = set(kwargs.keys()) - set(names)
    if unexpected:
      raise ValueError('Unexpected parameters: {} (allowed: {})'.format(
          sorted(unexpected), names))
    return {k: convert_to_tensors(v, dtypes_by_name[k])
            for k, v in kwargs.items() if v is not None}
  else:
    if len(args) > len(names):
      raise ValueError('Too many pinned values: {} (allowed: {})'.format(
          len(args), len(names)))
    return {k: convert_to_tensors(v, dtypes_by_name[k])
            for k, v in zip(names, args) if v is not None}


class _DefaultJointBijectorAutoBatchedWithPins(
    joint_distribution_vmap_mixin._DefaultJointBijectorAutoBatched):
  """Bijector that auto-batches over both its inputs *and* any pinned values."""

  def _vectorize_member_fn(self, member_fn, core_ndims):
    # Pinned values must be treated as *inputs* to vectorized bijector
    # members, since the pins can have batch dimensions that coincide
    # with the other values being transformed. For example, given
    #
    # jd = JointDistributionNamedAutoBatched({'
    #   'a': tfd.LogNormal(0., 1.),
    #   'b': lambda a: tfd.Uniform(high=a + tf.ones([3]))})
    # bij = jd.experimental_default_event_space_bijector()
    # sampled = jd.sample([2])  # ==> shape {'a': [2], 'b': [2, 3]}
    #
    # then if we pin a sampled value,
    #
    # pinned_jd = jd.experimental_pin(a=sampled['a'])
    # pinned_bij = pinned_jd.experimental_default_event_space_bijector()
    #
    # then we'd expect `pinned_bij.forward({'b': sampled['b']})` to return the
    # same value for `b` as `bij.forward(sampled)['b']`, in which
    # each of the two batch elements of `b` is transformed wrt the corresponding
    # batch element of `a`. If we used a naive _DefaultJointBijectorAutoBatched
    # instance for `pinned_bij`, we would instead get a shape error when
    # the pinned value for `a` appears in the model with batch shape `[2]`. The
    # solution is to ensure that the pinned value(s) are passed as input(s) to
    # every bijector method that we autovectorize.
    #
    # The approach implemented here uses a heavy hammer: calling any bijector
    # method rebuilds the pinned JD, creates a support bijector for its unpinned
    # values, and then invokes the requested method on that bijector.
    # (Re)creating all these Python objects incurs overhead in eager mode and
    # during `tf.function` tracing, but has no graph side effects, so repeated
    # execution of the traced function should be efficient.
    def build_and_invoke_pinned_bijector(pins, *args):
      bij = joint_distribution._DefaultJointBijector(  # pylint: disable=protected-access
          self._jd.distribution.experimental_pin(**pins),
          **self._bijector_kwargs)
      return member_fn(bij, *args)
    vectorized_fn_of_pins = vectorization_util.make_rank_polymorphic(
        build_and_invoke_pinned_bijector,
        core_ndims=[self._pins_core_ndims] + core_ndims)
    return lambda *args: vectorized_fn_of_pins(self._jd.pins, *args)

  @property
  def _pins_core_ndims(self):
    """Returns a map from names of pinned values to their batch+event ndims."""
    if not hasattr(self, '__pins_core_ndims'):  # Cache on first run.
      pinned_event_shape = {}
      original_jd = self._jd.distribution
      for name, event_shape in zip(
          original_jd._flat_resolve_names(),
          # TODO(davmre): support dynamic rank through `event_shape_tensor`.
          original_jd._model_flatten(original_jd.event_shape)):
        if name in self._jd.pins:
          pinned_event_shape[name] = event_shape

      self.__pins_core_ndims = tf.nest.map_structure(
          lambda s: (tensorshape_util.rank(self._jd.batch_shape) +  # pylint: disable=g-long-lambda
                     tensorshape_util.rank(s)),
          pinned_event_shape)
      if any(nd is None for nd in self.__pins_core_ndims.values()):
        # No inherent reason this can't support Tensor-valued event ranks, but
        # it's annoying, so let's punt for now.
        raise ValueError(
            'Attempting to construct an autovectorized support bijector, but '
            'the rank of the underlying model\'s events is not statically '
            'available. Contact tfprobability@tensorflow.org if you need this '
            'functionality.')
    return self.__pins_core_ndims
