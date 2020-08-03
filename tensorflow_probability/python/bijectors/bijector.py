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
"""Bijector base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib

# Dependency imports
import numpy as np
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import assert_util
from tensorflow_probability.python.internal import cache_util
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import name_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import tensorshape_util


__all__ = [
    'Bijector',
]


SKIP_DTYPE_CHECKS = False


@six.add_metaclass(abc.ABCMeta)
class Bijector(tf.Module):
  r"""Interface for transformations of a `Distribution` sample.

  Bijectors can be used to represent any differentiable and injective
  (one to one) function defined on an open subset of `R^n`.  Some non-injective
  transformations are also supported (see 'Non Injective Transforms' below).

  #### Mathematical Details

  A `Bijector` implements a [smooth covering map](
  https://en.wikipedia.org/wiki/Local_diffeomorphism), i.e., a local
  diffeomorphism such that every point in the target has a neighborhood evenly
  covered by a map ([see also](
  https://en.wikipedia.org/wiki/Covering_space#Covering_of_a_manifold)).
  A `Bijector` is used by `TransformedDistribution` but can be generally used
  for transforming a `Distribution` generated `Tensor`. A `Bijector` is
  characterized by three operations:

  1. Forward

     Useful for turning one random outcome into another random outcome from a
     different distribution.

  2. Inverse

     Useful for 'reversing' a transformation to compute one probability in
     terms of another.

  3. `log_det_jacobian(x)`

     'The log of the absolute value of the determinant of the matrix of all
     first-order partial derivatives of the inverse function.'

     Useful for inverting a transformation to compute one probability in terms
     of another. Geometrically, the Jacobian determinant is the volume of the
     transformation and is used to scale the probability.

     We take the absolute value of the determinant before log to avoid NaN
     values.  Geometrically, a negative determinant corresponds to an
     orientation-reversing transformation.  It is ok for us to discard the sign
     of the determinant because we only integrate everywhere-nonnegative
     functions (probability densities) and the correct orientation is always the
     one that produces a nonnegative integrand.

  By convention, transformations of random variables are named in terms of the
  forward transformation. The forward transformation creates samples, the
  inverse is useful for computing probabilities.

  #### Example Uses

  - Basic properties:

  ```python
  x = ...  # A tensor.
  # Evaluate forward transformation.
  fwd_x = my_bijector.forward(x)
  x == my_bijector.inverse(fwd_x)
  x != my_bijector.forward(fwd_x)  # Not equal because x != g(g(x)).
  ```

  - Computing a log-likelihood:

  ```python
  def transformed_log_prob(bijector, log_prob, x):
    return (bijector.inverse_log_det_jacobian(x, event_ndims=0) +
            log_prob(bijector.inverse(x)))
  ```

  - Transforming a random outcome:

  ```python
  def transformed_sample(bijector, x):
    return bijector.forward(x)
  ```

  #### Example Bijectors

  - 'Exponential'

    ```none
    Y = g(X) = exp(X)
    X ~ Normal(0, 1)  # Univariate.
    ```

    Implies:

    ```none
      g^{-1}(Y) = log(Y)
      |Jacobian(g^{-1})(y)| = 1 / y
      Y ~ LogNormal(0, 1), i.e.,
      prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                = (1 / y) Normal(log(y); 0, 1)
    ```

    Here is an example of how one might implement the `Exp` bijector:

    ```python
      class Exp(Bijector):

        def __init__(self, validate_args=False, name='exp'):
          super(Exp, self).__init__(
              validate_args=validate_args,
              forward_min_event_ndims=0,
              name=name)

        def _forward(self, x):
          return tf.exp(x)

        def _inverse(self, y):
          return tf.log(y)

        def _inverse_log_det_jacobian(self, y):
          return -self._forward_log_det_jacobian(self._inverse(y))

        def _forward_log_det_jacobian(self, x):
          # Notice that we needn't do any reducing, even when`event_ndims > 0`.
          # The base Bijector class will handle reducing for us; it knows how
          # to do so because we called `super` `__init__` with
          # `forward_min_event_ndims = 0`.
          return x
      ```

  - 'Affine'

    ```none
    Y = g(X) = sqrtSigma * X + mu
    X ~ MultivariateNormal(0, I_d)
    ```

    Implies:

    ```none
      g^{-1}(Y) = inv(sqrtSigma) * (Y - mu)
      |Jacobian(g^{-1})(y)| = det(inv(sqrtSigma))
      Y ~ MultivariateNormal(mu, sqrtSigma) , i.e.,
      prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                = det(sqrtSigma)^(-d) *
                  MultivariateNormal(inv(sqrtSigma) * (y - mu); 0, I_d)
      ```

  #### Min_event_ndims and Naming

  Bijectors are named for the dimensionality of data they act on (i.e. without
  broadcasting). We can think of bijectors having an intrinsic `min_event_ndims`
  , which is the minimum number of dimensions for the bijector act on. For
  instance, a Cholesky decomposition requires a matrix, and hence
  `min_event_ndims=2`.

  Some examples:

  `AffineScalar:  min_event_ndims=0`
  `Affine:  min_event_ndims=1`
  `Cholesky:  min_event_ndims=2`
  `Exp:  min_event_ndims=0`
  `Sigmoid:  min_event_ndims=0`
  `SoftmaxCentered:  min_event_ndims=1`

  Note the difference between `Affine` and `AffineScalar`. `AffineScalar`
  operates on scalar events, whereas `Affine` operates on vector-valued events.

  More generally, there is a `forward_min_event_ndims` and an
  `inverse_min_event_ndims`. In most cases, these will be the same.
  However, for some shape changing bijectors, these will be different
  (e.g. a bijector which pads an extra dimension at the end, might have
  `forward_min_event_ndims=0` and `inverse_min_event_ndims=1`.


  #### Jacobian Determinant

  The Jacobian determinant is a reduction over `event_ndims - min_event_ndims`
  (`forward_min_event_ndims` for `forward_log_det_jacobian` and
  `inverse_min_event_ndims` for `inverse_log_det_jacobian`).
  To see this, consider the `Exp` `Bijector` applied to a `Tensor` which has
  sample, batch, and event (S, B, E) shape semantics. Suppose the `Tensor`'s
  partitioned-shape is `(S=[4], B=[2], E=[3, 3])`. The shape of the `Tensor`
  returned by `forward` and `inverse` is unchanged, i.e., `[4, 2, 3, 3]`.
  However the shape returned by `inverse_log_det_jacobian` is `[4, 2]` because
  the Jacobian determinant is a reduction over the event dimensions.

  Another example is the `Affine` `Bijector`. Because `min_event_ndims = 1`, the
  Jacobian determinant reduction is over `event_ndims - 1`.

  It is sometimes useful to implement the inverse Jacobian determinant as the
  negative forward Jacobian determinant. For example,

  ```python
  def _inverse_log_det_jacobian(self, y):
     return -self._forward_log_det_jac(self._inverse(y))  # Note negation.
  ```

  The correctness of this approach can be seen from the following claim.

  - Claim:

      Assume `Y = g(X)` is a bijection whose derivative exists and is nonzero
      for its domain, i.e., `dY/dX = d/dX g(X) != 0`. Then:

      ```none
      (log o det o jacobian o g^{-1})(Y) = -(log o det o jacobian o g)(X)
      ```

  - Proof:

      From the bijective, nonzero differentiability of `g`, the
      [inverse function theorem](
          https://en.wikipedia.org/wiki/Inverse_function_theorem)
      implies `g^{-1}` is differentiable in the image of `g`.
      Applying the chain rule to `y = g(x) = g(g^{-1}(y))` yields
      `I = g'(g^{-1}(y))*g^{-1}'(y)`.
      The same theorem also implies `g^{-1}'` is non-singular therefore:
      `inv[ g'(g^{-1}(y)) ] = g^{-1}'(y)`.
      The claim follows from [properties of determinant](
  https://en.wikipedia.org/wiki/Determinant#Multiplicativity_and_matrix_groups).

  Generally its preferable to directly implement the inverse Jacobian
  determinant.  This should have superior numerical stability and will often
  share subgraphs with the `_inverse` implementation.

  #### Is_constant_jacobian

  Certain bijectors will have constant jacobian matrices. For instance, the
  `Affine` bijector encodes multiplication by a matrix plus a shift, with
  jacobian matrix, the same aforementioned matrix.

  `is_constant_jacobian` encodes the fact that the jacobian matrix is constant.
  The semantics of this argument are the following:

    * Repeated calls to 'log_det_jacobian' functions with the same
      `event_ndims` (but not necessarily same input), will return the first
      computed jacobian (because the matrix is constant, and hence is input
      independent).
    * `log_det_jacobian` implementations are merely broadcastable to the true
      `log_det_jacobian` (because, again, the jacobian matrix is input
      independent). Specifically, `log_det_jacobian` is implemented as the
      log jacobian determinant for a single input.

      ```python
      class Identity(Bijector):

        def __init__(self, validate_args=False, name='identity'):
          super(Identity, self).__init__(
              is_constant_jacobian=True,
              validate_args=validate_args,
              forward_min_event_ndims=0,
              name=name)

        def _forward(self, x):
          return x

        def _inverse(self, y):
          return y

        def _inverse_log_det_jacobian(self, y):
          return -self._forward_log_det_jacobian(self._inverse(y))

        def _forward_log_det_jacobian(self, x):
          # The full log jacobian determinant would be tf.zero_like(x).
          # However, we circumvent materializing that, since the jacobian
          # calculation is input independent, and we specify it for one input.
          return tf.constant(0., x.dtype)

      ```

  #### Subclass Requirements

  - Subclasses typically implement:

      - `_forward`,
      - `_inverse`,
      - `_inverse_log_det_jacobian`,
      - `_forward_log_det_jacobian` (optional),
      - `_is_increasing` (scalar bijectors only)

    The `_forward_log_det_jacobian` is called when the bijector is inverted via
    the `Invert` bijector. If undefined, a slightly less efficiently
    calculation, `-1 * _inverse_log_det_jacobian`, is used.

    If the bijector changes the shape of the input, you must also implement:

      - _forward_event_shape_tensor,
      - _forward_event_shape (optional),
      - _inverse_event_shape_tensor,
      - _inverse_event_shape (optional).

    By default the event-shape is assumed unchanged from input.

  - If the `Bijector`'s use is limited to `TransformedDistribution` (or friends
    like `QuantizedDistribution`) then depending on your use, you may not need
    to implement all of `_forward` and `_inverse` functions.

    Examples:

      1. Sampling (e.g., `sample`) only requires `_forward`.
      2. Probability functions (e.g., `prob`, `cdf`, `survival`) only require
         `_inverse` (and related).
      3. Only calling probability functions on the output of `sample` means
        `_inverse` can be implemented as a cache lookup.

    See 'Example Uses' [above] which shows how these functions are used to
    transform a distribution. (Note: `_forward` could theoretically be
    implemented as a cache lookup but this would require controlling the
    underlying sample generation mechanism.)

  #### Non Injective Transforms

  **WARNING** Handling of non-injective transforms is subject to change.

  Non injective maps `g` are supported, provided their domain `D` can be
  partitioned into `k` disjoint subsets, `Union{D1, ..., Dk}`, such that,
  ignoring sets of measure zero, the restriction of `g` to each subset is a
  differentiable bijection onto `g(D)`.  In particular, this implies that for
  `y in g(D)`, the set inverse, i.e. `g^{-1}(y) = {x in D : g(x) = y}`, always
  contains exactly `k` distinct points.

  The property, `_is_injective` is set to `False` to indicate that the bijector
  is not injective, yet satisfies the above condition.

  The usual bijector API is modified in the case `_is_injective is False` (see
  method docstrings for specifics).  Here we show by example the `AbsoluteValue`
  bijector.  In this case, the domain `D = (-inf, inf)`, can be partitioned
  into `D1 = (-inf, 0)`, `D2 = {0}`, and `D3 = (0, inf)`.  Let `gi` be the
  restriction of `g` to `Di`, then both `g1` and `g3` are bijections onto
  `(0, inf)`, with `g1^{-1}(y) = -y`, and `g3^{-1}(y) = y`.  We will use
  `g1` and `g3` to define bijector methods over `D1` and `D3`.  `D2 = {0}` is
  an oddball in that `g2` is one to one, and the derivative is not well defined.
  Fortunately, when considering transformations of probability densities
  (e.g. in `TransformedDistribution`), sets of measure zero have no effect in
  theory, and only a small effect in 32 or 64 bit precision.  For that reason,
  we define `inverse(0)` and `inverse_log_det_jacobian(0)` both as `[0, 0]`,
  which is convenient and results in a left-semicontinuous pdf.


  ```python
  abs = tfp.bijectors.AbsoluteValue()

  abs.forward(-1.)
  ==> 1.

  abs.forward(1.)
  ==> 1.

  abs.inverse(1.)
  ==> (-1., 1.)

  # The |dX/dY| is constant, == 1.  So Log|dX/dY| == 0.
  abs.inverse_log_det_jacobian(1., event_ndims=0)
  ==> (0., 0.)

  # Special case handling of 0.
  abs.inverse(0.)
  ==> (0., 0.)

  abs.inverse_log_det_jacobian(0., event_ndims=0)
  ==> (0., 0.)
  ```

  """

  _TF_MODULE_IGNORED_PROPERTIES = tf.Module._TF_MODULE_IGNORED_PROPERTIES.union(
      (
          '_graph_parents',
          '_is_constant_jacobian',
          '_cache',
      ))

  @abc.abstractmethod
  def __init__(self,
               graph_parents=None,
               is_constant_jacobian=False,
               validate_args=False,
               dtype=None,
               forward_min_event_ndims=None,
               inverse_min_event_ndims=None,
               parameters=None,
               name=None):
    """Constructs Bijector.

    A `Bijector` transforms random variables into new random variables.

    Examples:

    ```python
    # Create the Y = g(X) = X transform.
    identity = Identity()

    # Create the Y = g(X) = exp(X) transform.
    exp = Exp()
    ```

    See `Bijector` subclass docstring for more details and specific examples.

    Args:
      graph_parents: Python list of graph prerequisites of this `Bijector`.
      is_constant_jacobian: Python `bool` indicating that the Jacobian matrix is
        not a function of the input.
      validate_args: Python `bool`, default `False`. Whether to validate input
        with asserts. If `validate_args` is `False`, and the inputs are invalid,
        correct behavior is not guaranteed.
      dtype: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
        enforced.
      forward_min_event_ndims: Python `integer` indicating the minimum number of
        dimensions `forward` operates on.
      inverse_min_event_ndims: Python `integer` indicating the minimum number of
        dimensions `inverse` operates on. Will be set to
        `forward_min_event_ndims` by default, if no value is provided.
      parameters: Python `dict` of parameters used to instantiate this
        `Bijector`.
      name: The name to give Ops created by the initializer.

    Raises:
      ValueError:  If neither `forward_min_event_ndims` and
        `inverse_min_event_ndims` are specified, or if either of them is
        negative.
      ValueError:  If a member of `graph_parents` is not a `Tensor`.
    """
    if not name:
      name = type(self).__name__
      name = name_util.camel_to_lower_snake(name)
    name = name_util.get_name_scope_name(name)
    name = name_util.strip_invalid_chars(name)
    super(Bijector, self).__init__(name=name)
    self._name = name
    self._parameters = self._no_dependency(parameters)

    self._graph_parents = self._no_dependency(graph_parents or [])

    self._is_constant_jacobian = is_constant_jacobian
    self._validate_args = validate_args
    self._dtype = dtype

    self._initial_parameter_control_dependencies = tuple(
        d for d in self._parameter_control_dependencies(is_init=True)
        if d is not None)
    if self._initial_parameter_control_dependencies:
      self._initial_parameter_control_dependencies = (
          tf.group(*self._initial_parameter_control_dependencies),)

    if forward_min_event_ndims is None and inverse_min_event_ndims is None:
      raise ValueError('Must specify at least one of `forward_min_event_ndims` '
                       'and `inverse_min_event_ndims`.')
    elif inverse_min_event_ndims is None:
      inverse_min_event_ndims = forward_min_event_ndims
    elif forward_min_event_ndims is None:
      forward_min_event_ndims = inverse_min_event_ndims

    if not isinstance(forward_min_event_ndims, int):
      raise TypeError('Expected forward_min_event_ndims to be of '
                      'type int, got {}'.format(
                          type(forward_min_event_ndims).__name__))

    if not isinstance(inverse_min_event_ndims, int):
      raise TypeError('Expected inverse_min_event_ndims to be of '
                      'type int, got {}'.format(
                          type(inverse_min_event_ndims).__name__))

    if forward_min_event_ndims < 0:
      raise ValueError('forward_min_event_ndims must be a non-negative '
                       'integer.')
    if inverse_min_event_ndims < 0:
      raise ValueError('inverse_min_event_ndims must be a non-negative '
                       'integer.')

    self._forward_min_event_ndims = forward_min_event_ndims
    self._inverse_min_event_ndims = inverse_min_event_ndims

    for i, t in enumerate(self._graph_parents):
      if t is None or not tf.is_tensor(t):
        raise ValueError('Graph parent item %d is not a Tensor; %s.' % (i, t))

    # Setup caching after everything else is done.
    self._cache = self._setup_cache()

  def _setup_cache(self):
    """Defines the cache for this bijector."""
    # Wrap forward/inverse with getters so instance methods can be patched.
    return cache_util.BijectorCache(
        forward_impl=(lambda x, **kwargs: self._forward(x, **kwargs)),  # pylint: disable=unnecessary-lambda
        inverse_impl=(lambda y, **kwargs: self._inverse(y, **kwargs)),  # pylint: disable=unnecessary-lambda
        cache_type=cache_util.CachedDirectedFunction)

  @property
  def graph_parents(self):
    """Returns this `Bijector`'s graph_parents as a Python list."""
    return self._graph_parents

  @property
  def forward_min_event_ndims(self):
    """Returns the minimal number of dimensions bijector.forward operates on."""
    return self._forward_min_event_ndims

  @property
  def inverse_min_event_ndims(self):
    """Returns the minimal number of dimensions bijector.inverse operates on."""
    return self._inverse_min_event_ndims

  @property
  def is_constant_jacobian(self):
    """Returns true iff the Jacobian matrix is not a function of x.

    Note: Jacobian matrix is either constant for both forward and inverse or
    neither.

    Returns:
      is_constant_jacobian: Python `bool`.
    """
    return self._is_constant_jacobian

  @property
  def _is_injective(self):
    """Returns true iff the forward map `g` is injective (one-to-one function).

    **WARNING** This hidden property and its behavior are subject to change.

    Note:  Non-injective maps `g` are supported, provided their domain `D` can
    be partitioned into `k` disjoint subsets, `Union{D1, ..., Dk}`, such that,
    ignoring sets of measure zero, the restriction of `g` to each subset is a
    differentiable bijection onto `g(D)`.

    Returns:
      is_injective: Python `bool`.
    """
    return True

  @property
  def validate_args(self):
    """Returns True if Tensor arguments will be validated."""
    return self._validate_args

  @property
  def dtype(self):
    """dtype of `Tensor`s transformable by this distribution."""
    return self._dtype

  @property
  def name(self):
    """Returns the string name of this `Bijector`."""
    return self._name

  @property
  def parameters(self):
    """Dictionary of parameters used to instantiate this `Bijector`."""
    # Remove "self", "__class__", or other special variables. These can appear
    # if the subclass used:
    # `parameters = dict(locals())`.
    return {k: v for k, v in self._parameters.items()
            if not k.startswith('__') and k != 'self'}

  def __call__(self, value, name=None, **kwargs):
    """Applies or composes the `Bijector`, depending on input type.

    This is a convenience function which applies the `Bijector` instance in
    three different ways, depending on the input:

    1. If the input is a `tfd.Distribution` instance, return
       `tfd.TransformedDistribution(distribution=input, bijector=self)`.
    2. If the input is a `tfb.Bijector` instance, return
       `tfb.Chain([self, input])`.
    3. Otherwise, return `self.forward(input)`

    Args:
      value: A `tfd.Distribution`, `tfb.Bijector`, or a `Tensor`.
      name: Python `str` name given to ops created by this function.
      **kwargs: Additional keyword arguments passed into the created
        `tfd.TransformedDistribution`, `tfb.Bijector`, or `self.forward`.

    Returns:
      composition: A `tfd.TransformedDistribution` if the input was a
        `tfd.Distribution`, a `tfb.Chain` if the input was a `tfb.Bijector`, or
        a `Tensor` computed by `self.forward`.

    #### Examples

    ```python
    sigmoid = tfb.Reciprocal()(
        tfb.AffineScalar(shift=1.)(
          tfb.Exp()(
            tfb.AffineScalar(scale=-1.))))
    # ==> `tfb.Chain([
    #         tfb.Reciprocal(),
    #         tfb.AffineScalar(shift=1.),
    #         tfb.Exp(),
    #         tfb.AffineScalar(scale=-1.),
    #      ])`  # ie, `tfb.Sigmoid()`

    log_normal = tfb.Exp()(tfd.Normal(0, 1))
    # ==> `tfd.TransformedDistribution(tfd.Normal(0, 1), tfb.Exp())`

    tfb.Exp()([-1., 0., 1.])
    # ==> tf.exp([-1., 0., 1.])
    ```

    """

    # To avoid circular dependencies and keep the implementation local to the
    # `Bijector` class, we violate PEP8 guidelines and import here rather than
    # at the top of the file.
    from tensorflow_probability.python.bijectors import chain  # pylint: disable=g-import-not-at-top
    from tensorflow_probability.python.distributions import distribution  # pylint: disable=g-import-not-at-top
    from tensorflow_probability.python.distributions import transformed_distribution  # pylint: disable=g-import-not-at-top

    # TODO(b/128841942): Handle Conditional distributions and bijectors.
    if type(value) is transformed_distribution.TransformedDistribution:  # pylint: disable=unidiomatic-typecheck
      # We cannot accept subclasses with different constructors here, because
      # subclass constructors may accept constructor arguments TD doesn't know
      # how to handle. e.g. `TypeError: __init__() got an unexpected keyword
      # argument 'allow_nan_stats'` when doing
      # `tfb.Identity()(tfd.Chi(df=1., allow_nan_stats=True))`.
      new_kwargs = value.parameters
      new_kwargs.update(kwargs)
      new_kwargs['name'] = name or new_kwargs.get('name', None)
      new_kwargs['bijector'] = self(value.bijector)
      return transformed_distribution.TransformedDistribution(**new_kwargs)

    if isinstance(value, distribution.Distribution):
      return transformed_distribution.TransformedDistribution(
          distribution=value,
          bijector=self,
          name=name,
          **kwargs)

    if isinstance(value, chain.Chain):
      new_kwargs = kwargs.copy()
      new_kwargs['bijectors'] = [self] + ([] if value.bijectors is None
                                          else list(value.bijectors))
      if 'validate_args' not in new_kwargs:
        new_kwargs['validate_args'] = value.validate_args
      new_kwargs['name'] = name or value.name
      return chain.Chain(**new_kwargs)

    if isinstance(value, Bijector):
      return chain.Chain([self, value], name=name, **kwargs)

    return self.forward(value, name=name or 'forward', **kwargs)

  def _forward_event_shape_tensor(self, input_shape):
    """Subclass implementation for `forward_event_shape_tensor` function."""
    # By default, we assume event_shape is unchanged.
    return input_shape

  def forward_event_shape_tensor(self,
                                 input_shape,
                                 name='forward_event_shape_tensor'):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      input_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `forward` function.
      name: name to give to the op

    Returns:
      forward_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `forward`.
    """
    with self._name_and_control_scope(name):
      input_shape = tf.convert_to_tensor(
          input_shape, dtype_hint=tf.int32, name='input_shape')
      return tf.identity(
          tf.convert_to_tensor(self._forward_event_shape_tensor(input_shape),
                               dtype_hint=tf.int32),
          name='forward_event_shape')

  def _forward_event_shape(self, input_shape):
    """Subclass implementation for `forward_event_shape` public function."""
    # By default, we assume event_shape is unchanged.
    return input_shape

  def forward_event_shape(self, input_shape):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `forward_event_shape_tensor`. May be only partially defined.

    Args:
      input_shape: `TensorShape` indicating event-portion shape passed into
        `forward` function.

    Returns:
      forward_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `forward`. Possibly unknown.
    """
    input_shape = tf.TensorShape(input_shape)
    return tf.TensorShape(self._forward_event_shape(input_shape))

  def _inverse_event_shape_tensor(self, output_shape):
    """Subclass implementation for `inverse_event_shape_tensor` function."""
    # By default, we assume event_shape is unchanged.
    return output_shape

  def inverse_event_shape_tensor(self,
                                 output_shape,
                                 name='inverse_event_shape_tensor'):
    """Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      output_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `inverse` function.
      name: name to give to the op

    Returns:
      inverse_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `inverse`.
    """
    with self._name_and_control_scope(name):
      output_shape = tf.convert_to_tensor(
          output_shape, dtype_hint=tf.int32, name='output_shape')
      return tf.identity(
          tf.convert_to_tensor(self._inverse_event_shape_tensor(output_shape),
                               dtype_hint=tf.int32),
          name='inverse_event_shape')

  def _inverse_event_shape(self, output_shape):
    """Subclass implementation for `inverse_event_shape` public function."""
    # By default, we assume event_shape is unchanged.
    return output_shape

  def inverse_event_shape(self, output_shape):
    """Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

    Args:
      output_shape: `TensorShape` indicating event-portion shape passed into
        `inverse` function.

    Returns:
      inverse_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `inverse`. Possibly unknown.
    """
    output_shape = tf.TensorShape(output_shape)
    return tf.TensorShape(self._inverse_event_shape(output_shape))

  def _forward(self, x):
    """Subclass implementation for `forward` public function."""
    raise NotImplementedError('forward not implemented.')

  def _call_forward(self, x, name, **kwargs):
    """Wraps call to _forward, allowing extra shared logic."""
    with self._name_and_control_scope(name):
      x = tf.convert_to_tensor(x, dtype_hint=self.dtype, name='x')
      self._maybe_assert_dtype(x)
      if not self._is_injective:  # No caching for non-injective
        return self._forward(x, **kwargs)
      return self._cache.forward(x, **kwargs)

  def forward(self, x, name='forward', **kwargs):
    """Returns the forward `Bijector` evaluation, i.e., X = g(Y).

    Args:
      x: `Tensor`. The input to the 'forward' evaluation.
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`.

    Raises:
      TypeError: if `self.dtype` is specified and `x.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_forward` is not implemented.
    """
    return self._call_forward(x, name, **kwargs)

  @classmethod
  def _is_increasing(cls, **kwargs):
    """Subclass implementation for `is_increasing` public function."""
    raise NotImplementedError('`_is_increasing` not implemented.')

  def _call_is_increasing(self, name, **kwargs):
    """Wraps call to _is_increasing, allowing extra shared logic."""
    with self._name_and_control_scope(name):
      return tf.identity(self._is_increasing(**kwargs))

  def _internal_is_increasing(self, name='is_increasing', **kwargs):
    """For scalar bijectors, returns True where `d forward(x) / d x > 0`.

    This method, like `_is_injective`, is part of a contract with
    `TransformedDistribution`. This method supports the correctness of scalar
    `quantile` / `cdf` / `survival_function` for transformed distributions.

    Args:
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      A python `bool` or a `tf.bool` `Tensor`.
    """
    return self._call_is_increasing(name, **kwargs)

  def _inverse(self, y):
    """Subclass implementation for `inverse` public function."""
    raise NotImplementedError('inverse not implemented')

  def _call_inverse(self, y, name, **kwargs):
    """Wraps call to _inverse, allowing extra shared logic."""
    with self._name_and_control_scope(name):
      y = tf.convert_to_tensor(y, dtype_hint=self.dtype, name='y')
      self._maybe_assert_dtype(y)
      if not self._is_injective:  # No caching for non-injective
        return self._inverse(y, **kwargs)
      return self._cache.inverse(y, **kwargs)

  def inverse(self, y, name='inverse', **kwargs):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: `Tensor`. The input to the 'inverse' evaluation.
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective, returns the k-tuple containing the unique
        `k` points `(x1, ..., xk)` such that `g(xi) = y`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse` is not implemented.
    """
    return self._call_inverse(y, name, **kwargs)

  def _compute_inverse_log_det_jacobian_with_caching(
      self, x, y, prefer_inverse_ldj_fn, event_ndims, kwargs):
    """Compute ILDJ by the best available means, and ensure it is cached.

    Sub-classes of Bijector may implement either `_forward_log_det_jacobian` or
    `_inverse_log_det_jacobian`, or both, and may prefer one or the other for
    reasons of computational efficiency and/or numerics. In general, to compute
    the [I]LDJ, we need one of `x` or `y` and one of `_forward_log_det_jacobian`
    or `_inverse_log_det_jacobian` (all bijectors implement `forward` and
    `inverse`, so either of `x` and `y` may always computed from the other).

    This method encapsulates the logic of selecting the best possible method of
    computing the inverse log det jacobian, and executing it. Possible avenues
    to obtaining this value are:
      - recovering it from the cache,
      - computing it as `-_forward_log_det_jacobian(x)`, where `x` may need to
        be computed as `inverse(y)`, or
      - computing it as `_inverse_log_det_jacobian(y)`, where `y` may need to
        be computed as `forward(x)`.

    To make things more interesting, we may be able to take advantage of the
    knowledge that the jacobian is constant, for given `event_ndims`, and may
    even cache that result.

    To make things even more interesting still, we may need to perform an
    additional `reduce_sum` over the event dims of an input after computing the
    ILDJ (see `_reduce_jacobian_det_over_event` for the reduction logic). In
    order to prevent spurious TF graph dependencies on past inputs in cached
    results, we need to take care to do this reduction after the cache lookups.

    This method takes care of all the above concerns.

    Args:
      x: a `Tensor`, the pre-Bijector transform value at whose post-Bijector
        transform value the ILDJ is to be computed. Can be `None` as long as
        `y` is not `None`.
      y: a `Tensor`, a point in the output space of the Bijector's `forward`
        transformation, at whose value the ILDJ is to be computed. Can be
        `None` as long as `x` is not `None`.
      prefer_inverse_ldj_fn: Python `bool`, if `True`, will strictly prefer to
        use the `_inverse_log_det_jacobian` to compute ILDJ; else, will strictly
        prefer to use `_forward_log_det_jacobian`. The switching behavior allows
        the caller to communicate that one of the inverse or forward LDJ
        computations may be more efficient (usually because it can avoid an
        extra call to `inverse` or `forward`). It's only a "preference" because
        it may not always be possible, namely if the underlying implementation
        only has one of `_inverse_log_det_jacobian` or
        `_forward_log_det_jacobian` defined.
      event_ndims: int-like `Tensor`, the number of dims of an event (in the
        pre- or post-transformed space, as appropriate). These need to be summed
        over to compute the total ildj.
      kwargs: dictionary of keyword args that will be passed to calls to
        `self.forward` or `self.inverse` (if either of those are required), and
        to `self._compute_unreduced_ildj_with_caching` (which uses the
        conditioning kwargs for caching and for their underlying computations).

    Returns:
      ildj: a Tensor of ILDJ['s] at the given input (as specified by the args).
        Also updates the cache as needed.
    """
    # Ensure at least one of _inverse/_forward_log_det_jacobian is defined.
    if not (hasattr(self, '_inverse_log_det_jacobian') or
            hasattr(self, '_forward_log_det_jacobian')):
      raise NotImplementedError(
          'Neither _forward_log_det_jacobian nor _inverse_log_det_jacobian '
          'is implemented. One or the other is required.')

    # Use inverse_log_det_jacobian if either
    #   1. it is preferred to *and* we are able, or
    #   2. forward ldj fn isn't implemented (so we have no choice).
    use_inverse_ldj_fn = (
        (prefer_inverse_ldj_fn and hasattr(self, '_inverse_log_det_jacobian'))
        or not hasattr(self, '_forward_log_det_jacobian'))

    if use_inverse_ldj_fn:
      tensor_to_use = y if y is not None else self.forward(x, **kwargs)
      min_event_ndims = self.inverse_min_event_ndims
    else:
      tensor_to_use = x if x is not None else self.inverse(y, **kwargs)
      min_event_ndims = self.forward_min_event_ndims

    unreduced_ildj = self._compute_unreduced_ildj_with_caching(
        x, y, tensor_to_use, use_inverse_ldj_fn, kwargs)

    return self._reduce_jacobian_det_over_event(
        ps.shape(tensor_to_use), unreduced_ildj, min_event_ndims, event_ndims)

  def _compute_unreduced_ildj_with_caching(
      self, x, y, tensor_to_use, use_inverse_ldj_fn, kwargs):
    """Helper for computing ILDJ, with caching.

    Does not do the 'reduce' step which is necessary in some cases; this is left
    to the caller.

    Args:
      x: a `Tensor`, the pre-Bijector transform value at whose post-Bijector
        transform value the ILDJ is to be computed. Can be `None` as long as
        `y` is not `None`. This method only uses the value for cache
        lookup/updating.
      y: a `Tensor`, a point in the output space of the Bijector's `forward`
        transformation, at whose value the ILDJ is to be computed. Can be
        `None` as long as `x` is not `None`. This method only uses the value
        for cache lookup/updating.
      tensor_to_use: a `Tensor`, the one to actually pass to the chosen compute
        function (`_inverse_log_det_jacobian` or `_forward_log_det_jacobian`).
        It is presumed that the caller has already figured out what input to use
        (it will either be the x or y value corresponding to the location where
        we are computing the ILDJ).
      use_inverse_ldj_fn: Python `bool`, if `True`, will use the
        `_inverse_log_det_jacobian` to compute ILDJ; else, will use
        `_forward_log_det_jacobian`.
      kwargs: dictionary of keyword args that will be passed to calls to to
        `_inverse_log_det_jacobian` or `_forward_log_det_jacobian`, as well as
        for lookup/updating of the result in the cache.

    Returns:
      ildj: the (un-reduce_sum'ed) value of the ILDJ at the specified input
        location. Also updates the cache as needed.
    """
    if use_inverse_ldj_fn:
      attrs = self._cache.inverse.attributes(tensor_to_use, **kwargs)
      if 'ildj' not in attrs:
        attrs['ildj'] = self._inverse_log_det_jacobian(tensor_to_use, **kwargs)
    else:
      attrs = self._cache.forward.attributes(tensor_to_use, **kwargs)
      if 'ildj' not in attrs:
        attrs['ildj'] = -self._forward_log_det_jacobian(tensor_to_use, **kwargs)

    return attrs['ildj']

  def _call_inverse_log_det_jacobian(self, y, event_ndims, name, **kwargs):
    """Wraps call to _inverse_log_det_jacobian, allowing extra shared logic.

    Specifically, this method
      - adds a name scope,
      - performs validations,
      - handles the special case of non-injective Bijector (skip caching and
        reduce_sum over the values at the multiple points in the preimage of `y`
        under the non-injective transformation)

    so that sub-classes don't have to worry about this stuff.

    Args:
      y: same as in `inverse_log_det_jacobian`
      event_ndims: same as in `inverse_log_det_jacobian`
      name: same as in `inverse_log_det_jacobian`
      **kwargs: same as in `inverse_log_det_jacobian`

    Returns:
      ildj: the inverse log det jacobian at `y`. Also updates the cache as
        needed.
    """
    with self._name_and_control_scope(name), tf.control_dependencies(
        self._check_valid_event_ndims(
            min_event_ndims=self.inverse_min_event_ndims,
            event_ndims=event_ndims)):
      y = tf.convert_to_tensor(y, name='y')
      self._maybe_assert_dtype(y)

      if not self._is_injective:
        ildjs = self._inverse_log_det_jacobian(y, **kwargs)
        return tuple(
            self._reduce_jacobian_det_over_event(  # pylint: disable=g-complex-comprehension
                ps.shape(y), ildj, self.inverse_min_event_ndims, event_ndims)
            for ildj in ildjs)

      return self._compute_inverse_log_det_jacobian_with_caching(
          x=None,
          y=y,
          prefer_inverse_ldj_fn=True,
          event_ndims=event_ndims,
          kwargs=kwargs)

  def inverse_log_det_jacobian(self,
                               y,
                               event_ndims,
                               name='inverse_log_det_jacobian',
                               **kwargs):
    """Returns the (log o det o Jacobian o inverse)(y).

    Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

    Note that `forward_log_det_jacobian` is the negative of this function,
    evaluated at `g^{-1}(y)`.

    Args:
      y: `Tensor`. The input to the 'inverse' Jacobian determinant evaluation.
      event_ndims: Number of dimensions in the probabilistic events being
        transformed. Must be greater than or equal to
        `self.inverse_min_event_ndims`. The result is summed over the final
        dimensions to produce a scalar Jacobian determinant for each event, i.e.
        it has shape `rank(y) - event_ndims` dimensions.
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      ildj: `Tensor`, if this bijector is injective.
        If not injective, returns the tuple of local log det
        Jacobians, `log(det(Dg_i^{-1}(y)))`, where `g_i` is the restriction
        of `g` to the `ith` partition `Di`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse_log_det_jacobian` is not implemented.
    """
    return self._call_inverse_log_det_jacobian(y, event_ndims, name, **kwargs)

  def _call_forward_log_det_jacobian(self, x, event_ndims, name, **kwargs):
    """Wraps call to _forward_log_det_jacobian, allowing extra shared logic.

    Specifically, this method
      - adds a name scope,
      - performs validations,
      - handles the special case of non-injective Bijector (forward jacobian is
        ill-defined in this case and we raise an exception)

    so that sub-classes don't have to worry about this stuff.

    Args:
      x: same as in `forward_log_det_jacobian`
      event_ndims: same as in `forward_log_det_jacobian`
      name: same as in `forward_log_det_jacobian`
      **kwargs: same as in `forward_log_det_jacobian`

    Returns:
      fldj: the forward log det jacobian at `x`. Also updates the cache as
      needed.
    """
    if not self._is_injective:
      raise NotImplementedError(
          'forward_log_det_jacobian cannot be implemented for non-injective '
          'transforms.')

    with self._name_and_control_scope(name), tf.control_dependencies(
        self._check_valid_event_ndims(
            min_event_ndims=self.forward_min_event_ndims,
            event_ndims=event_ndims)):
      x = tf.convert_to_tensor(x, name='x')
      self._maybe_assert_dtype(x)

      return -self._compute_inverse_log_det_jacobian_with_caching(
          x=x,
          y=None,
          prefer_inverse_ldj_fn=False,
          event_ndims=event_ndims,
          kwargs=kwargs)

  def forward_log_det_jacobian(self,
                               x,
                               event_ndims,
                               name='forward_log_det_jacobian',
                               **kwargs):
    """Returns both the forward_log_det_jacobian.

    Args:
      x: `Tensor`. The input to the 'forward' Jacobian determinant evaluation.
      event_ndims: Number of dimensions in the probabilistic events being
        transformed. Must be greater than or equal to
        `self.forward_min_event_ndims`. The result is summed over the final
        dimensions to produce a scalar Jacobian determinant for each event, i.e.
        it has shape `rank(x) - event_ndims` dimensions.
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective this is not implemented.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if neither `_forward_log_det_jacobian`
        nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented, or
        this is a non-injective bijector.
    """
    return self._call_forward_log_det_jacobian(x, event_ndims, name, **kwargs)

  def _forward_dtype(self, dtype):
    # TODO(emilyaf): Raise an error if not implemented for bijectors with
    # multipart forward or inverse event shapes.
    return dtype

  def _inverse_dtype(self, dtype):
    # TODO(emilyaf): Raise an error if not implemented for bijectors with
    # multipart forward or inverse event shapes.
    return dtype

  def forward_dtype(self,
                    dtype,
                    name='forward_dtype',
                    **kwargs):
    """Returns the dtype of the output of the forward transformation.

    Args:
      dtype: `tf.dtype`, or nested structure of `tf.dtype`s, of the input to
        `forward`.
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `tf.dtype` or nested structure of `tf.dtype`s of the output of `forward`.
    """
    with self._name_and_control_scope(name):
      return self._forward_dtype(dtype, **kwargs)

  def inverse_dtype(self,
                    dtype,
                    name='inverse_dtype',
                    **kwargs):
    """Returns the dtype of the output of the inverse transformation.

    Args:
      dtype: `tf.dtype`, or nested structure of `tf.dtype`s, of the input to
        `inverse`.
      name: The name to give this op.
      **kwargs: Named arguments forwarded to subclass implementation.

    Returns:
      `tf.dtype` or nested structure of `tf.dtype`s of the output of `inverse`.
    """
    with self._name_and_control_scope(name):
      return self._inverse_dtype(dtype, **kwargs)

  @contextlib.contextmanager
  def _name_and_control_scope(self, name=None):
    """Helper function to standardize op scope."""
    with tf.name_scope(self.name):
      with tf.name_scope(name) as name_scope:
        deps = tuple(
            d for d in (  # pylint: disable=g-complex-comprehension
                tuple(self._initial_parameter_control_dependencies) +
                tuple(self._parameter_control_dependencies(is_init=False)))
            if d is not None)
        if not deps:
          yield name_scope
          return
        with tf.control_dependencies(deps) as deps_scope:
          yield deps_scope

  def _maybe_assert_dtype(self, x):
    """Helper to check dtype when self.dtype is known."""
    if SKIP_DTYPE_CHECKS:
      return
    if (self.dtype is not None and
        not dtype_util.base_equal(self.dtype, x.dtype)):
      raise TypeError(
          'Input had dtype %s but expected %s.' % (x.dtype, self.dtype))

  def _reduce_jacobian_det_over_event(
      self, shape_tensor, ildj, min_event_ndims, event_ndims):
    """Reduce jacobian over event_ndims - min_event_ndims."""
    # In this case, we need to tile the Jacobian over the event and reduce.
    shape_tensor = ps.convert_to_shape_tensor(shape_tensor)
    rank = ps.rank_from_shape(shape_tensor)
    shape_tensor = shape_tensor[rank - event_ndims:rank - min_event_ndims]

    ones = tf.ones(shape_tensor, ildj.dtype)
    reduced_ildj = tf.reduce_sum(
        ones * ildj,
        axis=self._get_event_reduce_dims(min_event_ndims, event_ndims))

    return reduced_ildj

  def _get_event_reduce_dims(self, min_event_ndims, event_ndims):
    """Compute the reduction dimensions given event_ndims."""
    event_ndims_ = self._maybe_get_static_event_ndims(event_ndims)

    if event_ndims_ is not None:
      return [-index for index in range(1, event_ndims_ - min_event_ndims + 1)]
    else:
      reduce_ndims = event_ndims - min_event_ndims
      return tf.range(-reduce_ndims, 0)

  def _check_valid_event_ndims(self, min_event_ndims, event_ndims):
    """Check whether event_ndims is atleast min_event_ndims."""
    event_ndims = tf.convert_to_tensor(event_ndims, name='event_ndims')
    event_ndims_ = tf.get_static_value(event_ndims)
    assertions = []

    if not dtype_util.is_integer(event_ndims.dtype):
      raise ValueError('Expected integer dtype, got dtype {}'.format(
          event_ndims.dtype))

    if event_ndims_ is not None:
      if tensorshape_util.rank(event_ndims.shape) != 0:
        raise ValueError('Expected scalar event_ndims, got shape {}'.format(
            event_ndims.shape))
      if min_event_ndims > event_ndims_:
        raise ValueError('event_ndims ({}) must be larger than '
                         'min_event_ndims ({})'.format(event_ndims_,
                                                       min_event_ndims))
    elif self.validate_args:
      assertions += [
          assert_util.assert_greater_equal(event_ndims, min_event_ndims)
      ]

    if tensorshape_util.is_fully_defined(event_ndims.shape):
      if tensorshape_util.rank(event_ndims.shape) != 0:
        raise ValueError('Expected scalar shape, got ndims {}'.format(
            tensorshape_util.rank(event_ndims.shape)))

    elif self.validate_args:
      assertions += [
          assert_util.assert_rank(event_ndims, 0, message='Expected scalar.')
      ]
    return assertions

  def _maybe_get_static_event_ndims(self, event_ndims):
    """Helper which returns tries to return an integer static value."""
    event_ndims_ = distribution_util.maybe_get_static_value(event_ndims)

    if isinstance(event_ndims_, (np.generic, np.ndarray)):
      if event_ndims_.dtype not in (np.int32, np.int64):
        raise ValueError('Expected integer dtype, got dtype {}'.format(
            event_ndims_.dtype))

      if isinstance(event_ndims_, np.ndarray) and len(event_ndims_.shape):
        raise ValueError(
            'Expected a scalar integer, got {}'.format(event_ndims_))
      event_ndims_ = int(event_ndims_)

    return event_ndims_

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
