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
"""PositiveSemidefiniteKernel base."""

import abc
import contextlib
import functools
import operator
import tensorflow.compat.v2 as tf


from tensorflow_probability.python.internal import auto_composite_tensor
from tensorflow_probability.python.internal import batch_shape_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import slicing
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.psd_kernels.internal import util


__all__ = [
    'AutoCompositeTensorPsdKernel',
    'PositiveSemidefiniteKernel',
]


class PositiveSemidefiniteKernel(tf.Module, metaclass=abc.ABCMeta):
  """Abstract base class for positive semi-definite kernel functions.

  #### Background

  For any set `S`, a real- (or complex-valued) function `k` on the Cartesian
  product `S x S` is called positive semi-definite if we have

  ```none
  sum_i sum_j (c[i]*) c[j] k(x[i], x[j]) >= 0
  ```

  for any finite collections `{x[1], ..., x[N]}` in S and `{c[1], ..., c[N]}` in
  the reals (or the complex plane). '*' denotes the complex conjugate, in the
  complex case.

  Some examples:
    - `S` is R, and `k(s, t) = (s - a) (t - b)`, where a, b are in R. This
      corresponds to a linear kernel.
    - `S` is R^+ U {0}, and `k(s, t) = min(s, t)`. This corresponds to a kernel
      for a Wiener process.
    - `S` is the set of strings over an alphabet `A = {c1, ... cC}`, and
      `k(s, t)` is defined via some similarity metric over strings.

  We model positive semi-definite functions (*kernels*, in common machine
  learning parlance) as classes with 3 primary public methods: `apply`,
  `matrix`, and `tensor`.

  `apply` computes the value of the kernel function at a pair of (batches of)
  input locations. It is the more "low-level" operation: `matrix` and `tensor`
  are implemented in terms of `apply`.

  `matrix` computes the value of the kernel *pairwise* on two (batches of)
  lists of input examples. When the two collections are the same the result is
  called the Gram (or Gramian) matrix
  (https://en.wikipedia.org/wiki/Gramian_matrix).

  `tensor` generalizes `matrix`, taking rank `k1` and `k2` collections of
  input examples to a rank `k1 + k2` collection of kernel values.

  #### Kernel Parameter Shape Semantics

  PositiveSemidefiniteKernel implementations support batching of kernel
  parameters and broadcasting of these parameters across batches of inputs. This
  allows, for example, creating a single kernel object which acts like a
  collection of kernels with different parameters. This might be useful for,
  e.g., for exploring multiple random initializations in parallel during a
  kernel parameter optimization procedure.

  The interaction between kernel parameter shapes and input shapes (see below)
  is somewhat subtle. The semantics are designed to make the most common use
  cases easy, while not ruling out more intricate control. The overarching
  principle is that kernel parameter batch shapes must be broadcastable with
  input batch shapes (see below). Examples are provided in the method-level
  documentation.

  #### Input Shape Semantics

  PositiveSemidefiniteKernel methods each support a notion of batching inputs;
  see the method-level documentation for full details; here we describe the
  overall semantics of input shapes. Inputs to PositiveSemidefiniteKernel
  methods partition into 3 pieces:

  ```none
  [b1, ..., bB, e1, ..., eE, f1, ..., fF]
  '----------'  '---------'  '---------'
       |             |            '-- Feature dimensions
       |             '-- Example dimensions
       '-- Batch dimensions
  ```

  - Feature dimensions correspond to the space over which the kernel is defined;
    in typical applications inputs are vectors and this part of the shape is
    rank-1. For example, if our kernel is defined over R^2 x R^2, each input is
    a 2-D vector (a rank-1 tensor of shape `[2,]`) so that
    `F = 1, [f1, ..., fF] = [2]`. If we defined a kernel over DxD matrices, its
    domain would be R^(DxD) x R^(DxD), we would have `F = 2` and
    `[f1, ..., fF] = [D, D]`. Feature shapes of inputs should be the same, but
    no exception will be raised unless they are broadcast-incompatible.
  - Batch dimensions describe collections of inputs which in some sense have
    nothing to do with each other, but may be coupled to batches of kernel
    parameters. It's required that batch dimensions of inputs broadcast with
    each other, and with the kernel's overall batch shape.
  - Example dimensions are shape elements which represent a collection of inputs
    that in some sense "go together" (whereas batches are "independent"). The
    exact semantics are different for the `apply`, `matrix` and `tensor` methods
    (see method-level doc strings for more details). `apply` combines examples
    together pairwise, much like the python built-in `zip`. `matrix` combines
    examples pairwise for *all* pairs of elements from two rank-1 input
    collections (lists), ie, it applies the kernel to all elements in the
    cross-product of two lists of examples. `tensor` further generalizes
    `matrix` to higher rank collections of inputs. Only `matrix` strictly
    requires example dimensions to be present (and to be exactly rank 1),
    although the typical usage of `apply` (eg, building a matrix diagonal) will
    also have `example_ndims` 1.

  Inputs may also be nested structures, in which case the batch and example
  dimensions of all elements must broadcast. The number of feature dimensions of
  each element must be equal to the corresponding element of the `feature_ndims`
  structure.

  ##### Examples

    ```python
    import tensorflow_probability as tfp

    # Suppose `SomeKernel` acts on vectors (rank-1 tensors), ie number of
    # feature dimensions is 1.
    scalar_kernel = tfp.math.psd_kernels.SomeKernel(param=.5)
    scalar_kernel.batch_shape
    # ==> []

    # `x` and `y` are batches of five 3-D vectors:
    x = np.ones([5, 3], np.float32)
    y = np.ones([5, 3], np.float32)
    scalar_kernel.apply(x, y).shape
    # ==> [5]

    scalar_kernel.matrix(x, y).shape
    # ==> [5, 5]
    ```

    Now we can consider a kernel with batched parameters:

    ```python
    batch_kernel = tfp.math.psd_kernels.SomeKernel(param=[.2, .5])
    batch_kernel.batch_shape
    # ==> [2]

    # `x` and `y` are batches of five 3-D vectors:
    x = np.ones([5, 3], np.float32)
    y = np.ones([5, 3], np.float32)

    batch_kernel.apply(x, y).shape
    # ==> Error! [2] and [5] can't broadcast.
    # We could solve this by telling `apply` to treat the 5 as an example dim:

    batch_kernel.apply(x, y, example_ndims=1).shape
    # ==> [2, 5]

    # Note that example_ndims is implicitly 1 for a call to `matrix`, so the
    # following just works:
    batch_kernel.matrix(x, y).shape
    # ==> [2, 5, 5]
    ```

  """

  def __init__(self,
               feature_ndims,
               dtype=None,
               name=None,
               validate_args=False,
               parameters=None):
    """Construct a PositiveSemidefiniteKernel (subclass) instance.

    Args:
      feature_ndims: Python `integer`, or nested structure of integers,
        indicating the number of dims (the rank) of the feature space this
        kernel acts on.
      dtype: `DType` on which this kernel operates. Must have the same nested
        structure as `feature_ndims`.
      name: Python `str` name prefixed to Ops created by this class. Default:
        subclass name.
      validate_args: Python `bool`, default `False`. When `True` kernel
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      parameters: Python `dict` of constructor arguments.

    Raises:
      ValueError: if `feature_ndims` (or any element, if nested) is not an
        integer greater than or equal to 0.

    Inputs to PositiveSemidefiniteKernel methods (or each element, if nested)
    partition into 3 pieces:

    ```none
    [b1, ..., bB, e1, ..., eE, f1, ..., fF]
    '----------'  '---------'  '---------'
         |             |            '-- Feature dimensions
         |             '-- Example dimensions
         '-- Batch dimensions
    ```

    The `feature_ndims` argument declares how many of the right-most shape
    dimensions belong to the feature dimensions. This enables us to predict
    which shape dimensions will be 'reduced' away during kernel computation.
    """
    if not all(isinstance(ndim, int) and ndim >= 0
               for ndim in tf.nest.flatten(feature_ndims)):
      raise ValueError(
          '`feature_ndims` must contain only Python `integer`s greater than '
          f'or equal to zero. Got: {feature_ndims}')
    self._feature_ndims = feature_ndims

    # If dtype is not provided, ensure that it has the same nested structure as
    # the kernel's inputs.
    if dtype is None:
      dtype = nest_util.broadcast_structure(feature_ndims, None)
    self._dtype = dtype
    if not name or name[-1] != '/':  # `name` is not a name scope
      name = tf.name_scope(name or type(self).__name__).name
    self._name = name
    self._validate_args = validate_args
    if parameters is not None:
      # Ensure no `self` references.
      parameters = {k: v for k, v in parameters.items()
                    if v is not self and not k.startswith('__')}
    self._parameters = self._no_dependency(parameters)
    self._initial_parameter_control_dependencies = tuple(
        d for d in self._parameter_control_dependencies(is_init=True)
        if d is not None)
    if self._initial_parameter_control_dependencies:
      self._initial_parameter_control_dependencies = (
          tf.group(*self._initial_parameter_control_dependencies),)

  @classmethod
  def _parameter_properties(cls, dtype):
    raise NotImplementedError(
        f'_parameter_properties` is not implemented: {cls.__name__}')

  @property
  def parameters(self):
    """Dictionary of parameters used to instantiate this `PSDKernel`."""
    # Remove 'self', '__class__', or other special variables. These can appear
    # if the subclass used: `parameters = dict(locals())`.
    if (not hasattr(self, '_parameters_sanitized') or
        not self._parameters_sanitized):
      p = self._parameters() if callable(self._parameters) else self._parameters
      self._parameters = self._no_dependency({
          k: v for k, v in p.items()
          if not k.startswith('__') and v is not self})
      self._parameters_sanitized = True
    # In some situations, the PSDKernel metaclass logic defers the evaluation
    # of parameters, but at this point we actually want to evaluate the
    # parameters.
    return dict(
        self._parameters() if callable(self._parameters) else self._parameters)

  @classmethod
  def parameter_properties(cls, dtype=tf.float32):
    """Returns a dict mapping constructor arg names to property annotations.

    This dict should include an entry for each of the kernel's
    `Tensor`-valued constructor arguments.

    Args:
      dtype: Optional float `dtype` to assume for continuous-valued parameters.
        Some constraining bijectors require advance knowledge of the dtype
        because certain constants (e.g., `tfb.Softplus.low`) must be
        instantiated with the same dtype as the values to be transformed.
    Returns:
      parameter_properties: A
        `str -> `tfp.python.internal.parameter_properties.ParameterProperties`
        dict mapping constructor argument names to `ParameterProperties`
        instances.
    """
    with tf.name_scope('parameter_properties'):
      return cls._parameter_properties(dtype)

  def _params_event_ndims(self):
    """Returns a dict mapping constructor argument names to per-event rank.

    The ranks are pulled from `cls.parameter_properties()`; this is a
    convenience wrapper.

    Returns:
      params_event_ndims: Per-event parameter ranks, a `str->int dict`.
    """
    try:
      properties = type(self).parameter_properties()
    except NotImplementedError:
      raise NotImplementedError(
          '{type(self)} does not support batch slicing; must implement '
          '_parameter_properties.')
    params_event_ndims = {}
    for (k, param) in properties.items():
      ndims = param.instance_event_ndims(self)
      if param.is_tensor and ndims is not None:
        params_event_ndims[k] = ndims
    return params_event_ndims

  def __getitem__(self, slices):
    """Slices the batch axes of this kernel, returning a new instance.

    ```python
    k = tfpk.ExponentiatedQuadratic(
      amplitude=tf.ones([3, 5, 7, 9]),
      length_scale=tf.ones([3, 5, 7, 9]))
    k.batch_shape  # => [3, 5, 7, 9]
    k2 = k[:, tf.newaxis, ..., -2:, 1::2]
    k2.batch_shape  # => [3, 1, 5, 2, 4]
    ```

    Args:
      slices: slices from the [] operator

    Returns:
      dist: A new `PositiveSemidefiniteKernel` instance with sliced parameters.
    """
    return slicing.batch_slice(self, {}, slices)

  def __iter__(self):
    raise TypeError(f'{type(self).__name__} object is not iterable')

  @property
  def feature_ndims(self):
    """The number of feature dimensions.

    Kernel functions generally act on pairs of inputs from some space like

    ```none
    R^(d1 x ... x dD)
    ```

    or, in words: rank-`D` real-valued tensors of shape `[d1, ..., dD]`. Inputs
    can be vectors in some `R^N`, but are not restricted to be. Indeed, one
    might consider kernels over matrices, tensors, or even more general spaces,
    like strings or graphs. Inputs may also be nested structures, in which case
    `feature_ndims` is a parallel nested structure containing the feature rank
    of each component.

    Returns:
      The (possibly nested) number of feature dimensions (feature rank) of this
      kernel.
    """
    return self._feature_ndims

  @property
  def dtype(self):
    """(Nested) dype over which the kernel operates."""
    return self._dtype

  @property
  def name(self):
    """Name prepended to all ops created by this class."""
    return self._name

  @property
  def validate_args(self):
    """Python `bool` indicating possibly expensive checks are enabled."""
    return self._validate_args

  def copy(self, **override_parameters_kwargs):
    """Creates a copy of the kernel.

    Note: the copy may continue to depend on the original initialization
    arguments.

    Args:
      **override_parameters_kwargs: String/value dictionary of initialization
        arguments to override with new values.
    Returns:
      copied_kernel: A new instance of `type(self)` initialized from the union
        of self.parameters and override_parameters_kwargs, i.e.,
        `dict(self.parameters, **override_parameters_kwargs)`.
    """
    parameters = dict(self.parameters, **override_parameters_kwargs)
    copied_kernel = type(self)(**parameters)
    # pylint: disable=protected-access
    copied_kernel._parameters = self._no_dependency(parameters)
    # pylint: enable=protected-access
    return copied_kernel

  def _batch_shape_tensor(self, **parameter_kwargs):
    """Infers batch shape from parameters.

    The overall batch shape is inferred by broadcasting the batch shapes of
    all parameters,

    ```python
    parameter_batch_shapes = []
    for name, properties in self.parameter_properties.items():
      parameter = self.parameters[name]
      parameter_batch_shapes.append(
        base_shape(parameter)[:-properties.instance_event_ndims(parameter)])
    ```

    where a parameter's `base_shape` is its batch shape if it
    defines one (e.g., if it is a PSDKernel, LinearOperator, etc.), and its
    Tensor shape otherwise.

    Args:
      **parameter_kwargs: Optional keyword arguments overriding the parameter
        values in `self.parameters`. Typically this is used to avoid multiple
        Tensor conversions of the same value.
    Returns:
      batch_shape_tensor: `Tensor` broadcast batch shape of all parameters.
    """
    try:
      return batch_shape_lib.inferred_batch_shape_tensor(
          self, **parameter_kwargs)
    except NotImplementedError:
      raise NotImplementedError('Cannot compute batch shape of PSDKernel '
                                '{self}: you must implement at least one of '
                                '`_batch_shape_tensor` or '
                                '`_parameter_properties`.')

  def batch_shape_tensor(self, name='batch_shape_tensor'):
    """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this `PositiveSemidefiniteKernel`.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
    with self._name_and_control_scope(name):
      # Try to get the static batch shape.
      batch_shape = self.batch_shape
      if not tensorshape_util.is_fully_defined(batch_shape):
        batch_shape = self._batch_shape_tensor()
        batch_shape = ps.convert_to_shape_tensor(
            self._batch_shape_tensor())
      return ps.convert_to_shape_tensor(batch_shape)

  def _batch_shape(self):
    """Infers static batch shape from parameters.

    The overall batch shape is inferred by broadcasting the batch shapes of
    all parameters

    ```python
    parameter_batch_shapes = []
    for name, properties in self.parameter_properties.items():
      parameter = self.parameters[name]
      parameter_batch_shapes.append(
        base_shape(parameter)[:-properties.instance_event_ndims(parameter)])
    ```

    where a parameter's `base_shape` is its batch shape if it
    defines one (e.g., if it is a PSDKernel, LinearOperator, etc.), and its
    Tensor shape otherwise.

    Returns:
      batch_shape: `tf.TensorShape` broadcast batch shape of all parameters; may
        be partially defined or unknown.
    """
    try:
      return batch_shape_lib.inferred_batch_shape(self)
    except NotImplementedError:
      # If a kernel doesn't implement `_parameter_properties` or its own
      # `_batch_shape` method, we can only return the most general shape.
      return tf.TensorShape(None)

  @property
  def batch_shape(self):
    """Shape of a single sample from a single event index as a `TensorShape`.

    May be partially defined or unknown.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this `PositiveSemidefiniteKernel`.

    Returns:
      batch_shape: `TensorShape`, possibly unknown.
    """
    if not hasattr(self, '__cached_batch_shape'):
      self.__cached_batch_shape = self._no_dependency(self._batch_shape())
    return self.__cached_batch_shape

  def _broadcast_parameters_with_batch_shape(self, batch_shape):
    """Broadcasts each parameter's batch shape with the given `batch_shape`.

    This can be understood as a pseudo-inverse operation to batch slicing.

    Args:
      batch_shape: Integer `Tensor` batch shape.
    Returns:
      broadcast_kernel: copy of this kernel in which each parameter's
        batch shape is determined by broadcasting its current batch shape with
        the given `batch_shape`.
    """
    return self.copy(
        **batch_shape_lib.broadcast_parameters_with_batch_shape(
            self, batch_shape))

  @contextlib.contextmanager
  def _name_and_control_scope(self, name=None):
    """Helper function to standardize op scope."""
    with tf.name_scope(self.name):
      with tf.name_scope(name) as name_scope:
        deps = tuple(
            d for d in (  # pylint: disable=g-complex-comprehension
                tuple(self._initial_parameter_control_dependencies) +
                tuple(self._parameter_control_dependencies(is_init=False))))
        if not deps:
          yield name_scope
          return
        with tf.control_dependencies(deps) as deps_scope:
          yield deps_scope

  def apply(self, x1, x2, example_ndims=0, name='apply'):
    """Apply the kernel function pairs of inputs.

    Args:
      x1: (Nested) `Tensor` input to the kernel, of shape `B1 + E1 + F`, where
        `B1` and `E1` may be empty (ie, no batch/example dims, resp.). If
        nested, `B1` and `E1` must broadcast across elements of the
        structure. `F` (the feature shape) must have rank equal to the
        kernel's `feature_ndims` property, or to the corresponding element of
        the `feature_ndims` nested structure. Batch shape must broadcast with
        the batch shape of `x2` and with the kernel's batch shape. Example
        shape must broadcast with example shape of `x2`. `x1` and `x2` must
        have the same *number* of example dims (ie, same rank).
      x2: (Nested) `Tensor` input to the kernel, of shape `B2 + E2 + F`, where
        `B2` and `E2` may be empty (ie, no batch/example dims, resp.). If
        nested, `B1` and `E1` must broadcast across elements of the
        structure. `F` (the feature shape) must have rank equal to the
        kernel's `feature_ndims` property, or to the corresponding element of
        the `feature_ndims` nested structure. Batch shape must broadcast with
        the batch shape of `x2` and with the kernel's batch shape. Example
        shape must broadcast with example shape of `x2`. `x1` and `x2` must
        have the same *number* of example dims (ie, same rank).
      example_ndims: A python integer, the number of example dims in the inputs.
        In essence, this parameter controls how broadcasting of the kernel's
        batch shape with input batch shapes works. The kernel batch shape will
        be broadcast against everything to the left of the combined example and
        feature dimensions in the input shapes.
      name: name to give to the op

    Returns:
      `Tensor` containing the results of applying the kernel function to inputs
      `x1` and `x2`. If the kernel parameters' batch shape is `Bk` then the
      shape of the `Tensor` resulting from this method call is
      `broadcast(Bk, B1, B2) + broadcast(E1, E2)`.

    Given an index set `S`, a kernel function is mathematically defined as a
    real- or complex-valued function on `S` satisfying the
    positive semi-definiteness constraint:

    ```none
    sum_i sum_j (c[i]*) c[j] k(x[i], x[j]) >= 0
    ```

    for any finite collections `{x[1], ..., x[N]}` in `S` and
    `{c[1], ..., c[N]}` in the reals (or the complex plane). '*' is the complex
    conjugate, in the complex case.

    This method most closely resembles the function described in the
    mathematical definition of a kernel. Given a PositiveSemidefiniteKernel `k`
    with scalar parameters and inputs `x` and `y` in `S`, `apply(x, y)` yields a
    single scalar value.

    #### Examples

    ```python
    import tensorflow_probability as tfp

    # Suppose `SomeKernel` acts on vectors (rank-1 tensors)
    scalar_kernel = tfp.math.psd_kernels.SomeKernel(param=.5)
    scalar_kernel.batch_shape
    # ==> []

    # `x` and `y` are batches of five 3-D vectors:
    x = np.ones([5, 3], np.float32)
    y = np.ones([5, 3], np.float32)
    scalar_kernel.apply(x, y).shape
    # ==> [5]
    ```

    The above output is the result of vectorized computation of the five values

    ```none
    [k(x[0], y[0]), k(x[1], y[1]), ..., k(x[4], y[4])]
    ```

    Now we can consider a kernel with batched parameters:

    ```python
    batch_kernel = tfp.math.psd_kernels.SomeKernel(param=[.2, .5])
    batch_kernel.batch_shape
    # ==> [2]
    batch_kernel.apply(x, y).shape
    # ==> Error! [2] and [5] can't broadcast.
    ```

    The parameter batch shape of `[2]` and the input batch shape of `[5]` can't
    be broadcast together. We can fix this in either of two ways:

    ##### Fix #1
    Give the parameter a shape of `[2, 1]` which will correctly broadcast with
    `[5]` to yield `[2, 5]`:

    ```python
    batch_kernel = tfp.math.psd_kernels.SomeKernel(
        param=[[.2], [.5]])
    batch_kernel.batch_shape
    # ==> [2, 1]
    batch_kernel.apply(x, y).shape
    # ==> [2, 5]
    ```

    ##### Fix #2
    By specifying `example_ndims`, which tells the kernel to treat the `5` in
    the input shape as part of the "example shape", and "pushing" the kernel
    batch shape to the left:

    ```python
    batch_kernel = tfp.math.psd_kernels.SomeKernel(param=[.2, .5])
    batch_kernel.batch_shape
    # ==> [2]
    batch_kernel.apply(x, y, example_ndims=1).shape
    # ==> [2, 5]
    ```

    """
    with self._name_and_control_scope(name):
      x1 = nest_util.convert_to_nested_tensor(
          x1, name='x1', dtype_hint=self.dtype, allow_packing=True)
      x2 = nest_util.convert_to_nested_tensor(
          x2, name='x2', dtype_hint=self.dtype, allow_packing=True)
      return self._call_apply(x1, x2, example_ndims=example_ndims)

  def _call_apply(self, x1, x2, example_ndims):
    should_expand_dims = (example_ndims == 0)

    if should_expand_dims:
      example_ndims += 1
      expand_x_dims = lambda x, n: tf.expand_dims(x, -(n + 1))
      x1 = tf.nest.map_structure(expand_x_dims, x1, self.feature_ndims)
      x2 = tf.nest.map_structure(expand_x_dims, x2, self.feature_ndims)

    result = self._apply(x1, x2, example_ndims=example_ndims)

    if should_expand_dims:
      result = tf.squeeze(result, axis=-1)

    return result

  def _apply(self, x1, x2, example_ndims=1):
    """Apply the kernel function to a pair of (batches of) inputs.

    Subclasses must implement this method. It will always be called with
    example_ndims >= 1. Implementations should take care to respect
    example_ndims, by padding parameters on the right with 1's example_ndims
    times. See tests and existing subclasses for examples.

    Args:
      x1: (Nested) `Tensor` input to the first positional parameter of the
        kernel, of shape `B1 + E1 + F`, where `B1` may be empty (ie, no batch
        dims, resp.), `E1` is a shape of rank at least 1, and `F` (the
        feature shape) must have rank equal to the kernel's `feature_ndims`
        property (or to the corresponding element of `feature_ndims`, if
        nested). Batch shape must broadcast with the batch shape of `x2` and
        with the kernel's batch shape. Example shape must broadcast with
        example shape of `x2` (They don't strictly need to be equal, e.g.,
        when `apply` is called from `matrix`, `x1` and `x2` each have 1's in
        opposing positions in their example shapes). `x1` and `x2` must have
        the same *number* of example dims (ie, same rank).
      x2: (Nested) `Tensor` input to the second positional parameter of the
        kernel, shape `B2 + E2 + F`, where `B2` may be empty (ie, no batch
        dims, resp.), `E2` is a shape of rank at least 1, and `F` (the
        feature shape) must have rank equal to the kernel's `feature_ndims`
        property (or to the corresponding element of `feature_ndims`, if
        nested). Batch shape must broadcast with the batch shape of `x1` and
        with the kernel's batch shape. Example shape must broadcast with
        example shape of `x1` (They don't strictly need to be equal, e.g.,
        when `apply` is called from `matrix`, `x1` and `x2` each have 1's in
        opposing positions in their example shapes). `x1` and `x2` must have
        the same *number* of example dims (ie, same rank).
      example_ndims: A python integer greater than or equal to 1, the number of
        example dims in the inputs. In essence, this parameter controls how
        broadcasting of the kernel's batch shape with input batch shapes works.
        The kernel batch shape will be broadcast against everything to the left
        of the combined example and feature dimensions in the input shapes.

    Returns:
      `Tensor` containing the results of applying the kernel function to inputs
      `x1` and `x2`. If the kernel parameters' batch shape is `Bk` then the
      shape of the `Tensor` resulting from this method call is
      `broadcast(Bk, B1, B2) + broadcast(E1, E2)`.
    """
    raise NotImplementedError(
        'Subclasses must provide `_apply` implementation.')

  def matrix(self, x1, x2, name='matrix'):
    """Construct (batched) matrices from (batches of) collections of inputs.

    Args:
      x1: (Nested) `Tensor` input to the first positional parameter of the
        kernel, of shape `B1 + [e1] + F`, where `B1` may be empty (ie, no
        batch dims, resp.), `e1` is a single integer (ie, `x1` has example
        ndims exactly 1), and `F` (the feature shape) must have rank equal to
        the kernel's `feature_ndims` property (or to the corresponding
        element of `feature_ndims`, if nested). Batch shape must broadcast
        with the batch shape of `x2` and with the kernel's batch shape.
      x2: (Nested) `Tensor` input to the second positional parameter of the
        kernel, shape `B2 + [e2] + F`, where `B2` may be empty (ie, no batch
        dims, resp.), `e2` is a single integer (ie, `x2` has example ndims
        exactly 1), and `F` (the feature shape) must have rank equal to the
        kernel's `feature_ndims` property (or to the corresponding element of
        `feature_ndims`, if nested). Batch shape must broadcast with the
        batch shape of `x1` and with the kernel's batch shape.
      name: name to give to the op

    Returns:
      `Tensor` containing the matrix (possibly batched) of kernel applications
      to pairs from inputs `x1` and `x2`. If the kernel parameters' batch shape
      is `Bk` then the shape of the `Tensor` resulting from this method call is
      `broadcast(Bk, B1, B2) + [e1, e2]` (note this differs from `apply`: the
      example dimensions are concatenated, whereas in `apply` the example dims
      are broadcast together).

    Given inputs `x1` and `x2` of shapes

    ```none
    [b1, ..., bB, e1, f1, ..., fF]
    ```

    and

    ```none
    [c1, ..., cC, e2, f1, ..., fF]
    ```

    This method computes the batch of `e1 x e2` matrices resulting from applying
    the kernel function to all pairs of inputs from `x1` and `x2`. The shape
    of the batch of matrices is the result of broadcasting the batch shapes of
    `x1`, `x2`, and the kernel parameters (see examples below). As such, it's
    required that these shapes all be broadcast compatible. However, the kernel
    parameter batch shapes need not broadcast against the 'example shapes' (`e1`
    and `e2` above).

    When the two inputs are the (batches of) identical collections, the
    resulting matrix is the so-called Gram (or Gramian) matrix
    (https://en.wikipedia.org/wiki/Gramian_matrix).

    #### Examples

    First, consider a kernel with a single scalar parameter.

    ```python
    import tensorflow_probability as tfp

    scalar_kernel = tfp.math.psd_kernels.SomeKernel(param=.5)
    scalar_kernel.batch_shape
    # ==> []

    # Our inputs are two lists of 3-D vectors
    x = np.ones([5, 3], np.float32)
    y = np.ones([4, 3], np.float32)
    scalar_kernel.matrix(x, y).shape
    # ==> [5, 4]
    ```

    The result comes from applying the kernel to the entries in `x` and `y`
    pairwise, across all pairs:

      ```none
      | k(x[0], y[0])    k(x[0], y[1])  ...  k(x[0], y[3]) |
      | k(x[1], y[0])    k(x[1], y[1])  ...  k(x[1], y[3]) |
      |      ...              ...                 ...      |
      | k(x[4], y[0])    k(x[4], y[1])  ...  k(x[4], y[3]) |
      ```

    Now consider a kernel with batched parameters with the same inputs

    ```python
    batch_kernel = tfp.math.psd_kernels.SomeKernel(param=[1., .5])
    batch_kernel.batch_shape
    # ==> [2]

    batch_kernel.matrix(x, y).shape
    # ==> [2, 5, 4]
    ```

    This results in a batch of 2 matrices, one computed from the kernel with
    `param = 1.` and the other with `param = .5`.

    We also support batching of the inputs. First, let's look at that with
    the scalar kernel again.

    ```python
    # Batch of 10 lists of 5 vectors of dimension 3
    x = np.ones([10, 5, 3], np.float32)

    # Batch of 10 lists of 4 vectors of dimension 3
    y = np.ones([10, 4, 3], np.float32)

    scalar_kernel.matrix(x, y).shape
    # ==> [10, 5, 4]
    ```

    The result is a batch of 10 matrices built from the batch of 10 lists of
    input vectors. These batch shapes have to be broadcastable. The following
    will *not* work:

    ```python
    x = np.ones([10, 5, 3], np.float32)
    y = np.ones([20, 4, 3], np.float32)
    scalar_kernel.matrix(x, y).shape
    # ==> Error! [10] and [20] can't broadcast.
    ```

    Now let's consider batches of inputs in conjunction with batches of kernel
    parameters. We require that the input batch shapes be broadcastable with
    the kernel parameter batch shapes, otherwise we get an error:

    ```python
    x = np.ones([10, 5, 3], np.float32)
    y = np.ones([10, 4, 3], np.float32)

    batch_kernel = tfp.math.psd_kernels.SomeKernel(params=[1., .5])
    batch_kernel.batch_shape
    # ==> [2]
    batch_kernel.matrix(x, y).shape
    # ==> Error! [2] and [10] can't broadcast.
    ```

    The fix is to make the kernel parameter shape broadcastable with `[10]` (or
    reshape the inputs to be broadcastable!):

    ```python
    x = np.ones([10, 5, 3], np.float32)
    y = np.ones([10, 4, 3], np.float32)

    batch_kernel = tfp.math.psd_kernels.SomeKernel(
        params=[[1.], [.5]])
    batch_kernel.batch_shape
    # ==> [2, 1]
    batch_kernel.matrix(x, y).shape
    # ==> [2, 10, 5, 4]

    # Or, make the inputs broadcastable:
    x = np.ones([10, 1, 5, 3], np.float32)
    y = np.ones([10, 1, 4, 3], np.float32)

    batch_kernel = tfp.math.psd_kernels.SomeKernel(
        params=[1., .5])
    batch_kernel.batch_shape
    # ==> [2]
    batch_kernel.matrix(x, y).shape
    # ==> [10, 2, 5, 4]
    ```

    Here, we have the result of applying the kernel, with 2 different
    parameters, to each of a batch of 10 pairs of input lists.

    """
    with self._name_and_control_scope(name):
      x1 = nest_util.convert_to_nested_tensor(
          x1, name='x1', dtype_hint=self.dtype, allow_packing=True)
      x2 = nest_util.convert_to_nested_tensor(
          x2, name='x2', dtype_hint=self.dtype, allow_packing=True)
      return self._matrix(x1, x2)

  def _matrix(self, x1, x2):
    x1 = tf.nest.map_structure(
        lambda x, n: util.pad_shape_with_ones(x, ndims=1, start=-(n + 1)),
        x1, self.feature_ndims)
    x2 = tf.nest.map_structure(
        lambda x, n: util.pad_shape_with_ones(x, ndims=1, start=-(n + 2)),
        x2, self.feature_ndims)

    return self._call_apply(x1, x2, example_ndims=2)

  def tensor(self, x1, x2, x1_example_ndims, x2_example_ndims, name='tensor'):
    """Construct (batched) tensors from (batches of) collections of inputs.

    Args:
      x1: (Nested) `Tensor` input to the first positional parameter of the
        kernel, of shape `B1 + E1 + F`, where `B1` and `E1` arbitrary shapes
        which may be empty (ie, no batch/example dims, resp.), and `F` (the
        feature shape) must have rank equal to the kernel's `feature_ndims`
        property (or to the corresponding element of `feature_ndims`, if
        nested). Batch shape must broadcast with the batch shape of `x2` and
        with the kernel's batch shape.
      x2: (Nested) `Tensor` input to the second positional parameter of the
        kernel, shape `B2 + E2 + F`, where `B2` and `E2` arbitrary shapes
        which may be empty (ie, no batch/example dims, resp.), and `F` (the
        feature shape) must have rank equal to the kernel's `feature_ndims`
        property (or to the corresponding element of `feature_ndims`, if
        nested). Batch shape must broadcast with the batch shape of `x1` and
        with the kernel's batch shape.
      x1_example_ndims: A python integer greater than or equal to 0, the number
        of example dims in the first input. This affects both the alignment of
        batch shapes and the shape of the final output of the function.
        Everything left of the feature shape and the example dims in `x1` is
        considered "batch shape", and must broadcast as specified above.
      x2_example_ndims: A python integer greater than or equal to 0, the number
        of example dims in the second input. This affects both the alignment of
        batch shapes and the shape of the final output of the function.
        Everything left of the feature shape and the example dims in `x1` is
        considered "batch shape", and must broadcast as specified above.
      name: name to give to the op

    Returns:
      `Tensor` containing (possibly batched) kernel applications to pairs from
      inputs `x1` and `x2`. If the kernel parameters' batch shape is `Bk` then
      the shape of the `Tensor` resulting from this method call is
      `broadcast(Bk, B1, B2) + E1 + E2`. Note this differs from `apply`: the
      example dimensions are concatenated, whereas in `apply` the example dims
      are broadcast together. It also differs from `matrix`: the example shapes
      are arbitrary here, and the result accrues a rank equal to the sum of the
      ranks of the input example shapes.

    #### Examples

    First, consider a kernel with a single scalar parameter.

    ```python
    import tensorflow_probability as tfp

    scalar_kernel = tfp.math.psd_kernels.SomeKernel(param=.5)
    scalar_kernel.batch_shape
    # ==> []

    # Our inputs are two rank-2 collections of 3-D vectors
    x = np.ones([5, 6, 3], np.float32)
    y = np.ones([7, 8, 3], np.float32)
    scalar_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
    # ==> [5, 6, 7, 8]

    # Empty example shapes work too!
    x = np.ones([3], np.float32)
    y = np.ones([5, 3], np.float32)
    scalar_kernel.tensor(x, y, x1_example_ndims=0, x2_example_ndims=1).shape
    # ==> [5]
    ```

    The result comes from applying the kernel to the entries in `x` and `y`
    pairwise, across all pairs:

      ```none
      | k(x[0], y[0])    k(x[0], y[1])  ...  k(x[0], y[3]) |
      | k(x[1], y[0])    k(x[1], y[1])  ...  k(x[1], y[3]) |
      |      ...              ...                 ...      |
      | k(x[4], y[0])    k(x[4], y[1])  ...  k(x[4], y[3]) |
      ```

    Now consider a kernel with batched parameters.

    ```python
    batch_kernel = tfp.math.psd_kernels.SomeKernel(param=[1., .5])
    batch_kernel.batch_shape
    # ==> [2]

    # Inputs are two rank-2 collections of 3-D vectors
    x = np.ones([5, 6, 3], np.float32)
    y = np.ones([7, 8, 3], np.float32)
    scalar_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
    # ==> [2, 5, 6, 7, 8]
    ```

    We also support batching of the inputs. First, let's look at that with
    the scalar kernel again.

    ```python
    # Batch of 10 lists of 5x6 collections of dimension 3
    x = np.ones([10, 5, 6, 3], np.float32)

    # Batch of 10 lists of 7x8 collections of dimension 3
    y = np.ones([10, 7, 8, 3], np.float32)

    scalar_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
    # ==> [10, 5, 6, 7, 8]
    ```

    The result is a batch of 10 tensors built from the batch of 10 rank-2
    collections of input vectors. The batch shapes have to be broadcastable.
    The following will *not* work:

    ```python
    x = np.ones([10, 5, 3], np.float32)
    y = np.ones([20, 4, 3], np.float32)
    scalar_kernel.tensor(x, y, x1_example_ndims=1, x2_example_ndims=1).shape
    # ==> Error! [10] and [20] can't broadcast.
    ```

    Now let's consider batches of inputs in conjunction with batches of kernel
    parameters. We require that the input batch shapes be broadcastable with
    the kernel parameter batch shapes, otherwise we get an error:

    ```python
    x = np.ones([10, 5, 6, 3], np.float32)
    y = np.ones([10, 7, 8, 3], np.float32)

    batch_kernel = tfp.math.psd_kernels.SomeKernel(params=[1., .5])
    batch_kernel.batch_shape
    # ==> [2]
    batch_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
    # ==> Error! [2] and [10] can't broadcast.
    ```

    The fix is to make the kernel parameter shape broadcastable with `[10]` (or
    reshape the inputs to be broadcastable!):

    ```python
    x = np.ones([10, 5, 6, 3], np.float32)
    y = np.ones([10, 7, 8, 3], np.float32)

    batch_kernel = tfp.math.psd_kernels.SomeKernel(
        params=[[1.], [.5]])
    batch_kernel.batch_shape
    # ==> [2, 1]
    batch_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
    # ==> [2, 10, 5, 6, 7, 8]

    # Or, make the inputs broadcastable:
    x = np.ones([10, 1, 5, 6, 3], np.float32)
    y = np.ones([10, 1, 7, 8, 3], np.float32)

    batch_kernel = tfp.math.psd_kernels.SomeKernel(
        params=[1., .5])
    batch_kernel.batch_shape
    # ==> [2]
    batch_kernel.tensor(x, y, x1_example_ndims=2, x2_example_ndims=2).shape
    # ==> [10, 2, 5, 6, 7, 8]
    ```

    """
    with self._name_and_control_scope(name):
      x1 = nest_util.convert_to_nested_tensor(
          x1, name='x1', dtype_hint=self.dtype, allow_packing=True)
      x2 = nest_util.convert_to_nested_tensor(
          x2, name='x2', dtype_hint=self.dtype, allow_packing=True)
      # Specialize to the matrix computation.
      if x1_example_ndims == 1 and x2_example_ndims == 1:
        return self._matrix(x1, x2)
      return self._tensor(x1, x2, x1_example_ndims, x2_example_ndims)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    x1 = tf.nest.map_structure(
        lambda x, n: util.pad_shape_with_ones(  # pylint: disable=g-long-lambda
            x, ndims=x2_example_ndims, start=-(n + 1)),
        x1, self.feature_ndims)

    x2 = tf.nest.map_structure(
        lambda x, n: util.pad_shape_with_ones(  # pylint: disable=g-long-lambda
            x, ndims=x1_example_ndims, start=-(n + 1 + x2_example_ndims)),
        x2, self.feature_ndims)

    return self._call_apply(
        x1, x2, example_ndims=(x1_example_ndims + x2_example_ndims))

  def __add__(self, k):
    if not isinstance(k, PositiveSemidefiniteKernel):
      raise ValueError(
          "Can't add non-kernel (of type '%s') to kernel" % type(k))
    return _SumKernel([self, k])

  def __radd__(self, lhs):
    # Special-case `0 + kernel` so that sum(list_of_kernels) works (it appears
    # to be implemented as a fold with initial value `0`.
    if lhs == 0:
      return self
    return self + lhs

  def __iadd__(self, k):
    return self.__add__(k)

  def __mul__(self, k):
    if not isinstance(k, PositiveSemidefiniteKernel):
      raise ValueError(
          "Can't multiply by non-kernel (of type '%s') to kernel" % type(k))
    return _ProductKernel([self, k])

  def __imul__(self, k):
    return self.__mul__(k)

  def __str__(self):
    return ('tfp.math.psd_kernels.{type_name}('
            '"{self_name}"'
            '{maybe_batch_shape}'
            ', feature_ndims={feature_ndims}'
            ', dtype={dtype})'.format(
                type_name=type(self).__name__,
                self_name=self.name,
                maybe_batch_shape=(
                    ', batch_shape={}'.format(self.batch_shape)
                    if tensorshape_util.rank(self.batch_shape) is not None
                    else ''),
                feature_ndims=self.feature_ndims,
                dtype=None if self.dtype is None
                else tf.nest.map_structure(dtype_util.name, self.dtype)))

  def __repr__(self):
    return ('<tfp.math.psd_kernels.{type_name} '
            '\'{self_name}\''
            ' batch_shape={batch_shape}'
            ' feature_ndims={feature_ndims}'
            ' dtype={dtype}>'.format(
                type_name=type(self).__name__,
                self_name=self.name,
                batch_shape=self.batch_shape,
                feature_ndims=self.feature_ndims,
                dtype=None if self.dtype is None
                else tf.nest.map_structure(dtype_util.name, self.dtype)))

  def _parameter_control_dependencies(self, is_init):
    """Returns a list of ops to be executed in members with graph deps.

    Typically subclasses override this function to return parameter specific
    assertions (eg, positivity of `amplitude`, etc.).

    Args:
      is_init: Python `bool` indicating that the call site is `__init__`.

    Returns:
      dependencies: `list`-like of ops to be executed in member functions with
        graph dependencies.
    """
    return ()

  @property
  def _composite_tensor_params(self):
    """A tuple describing which parameters are expected to be tensors.

    CompositeTensor requires us to partition dynamic (tensor) parts from static
    (metadata) parts like 'validate_args'.  This collects the keys of parameters
    which are expected to be tensors.
    """
    return (self._composite_tensor_nonshape_params +
            self._composite_tensor_shape_params)

  @property
  def _composite_tensor_nonshape_params(self):
    """A tuple describing which parameters are non-shape-related tensors.

    Flattening in JAX involves many of the same considerations with regards to
    identifying tensor arguments for the purposes of CompositeTensor, except
    that shape-related items will be considered metadata.  This property
    identifies the keys of parameters that are expected to be tensors, except
    those that are shape-related.
    """
    try:
      return tuple(k for k, v in self.parameter_properties().items()
                   if not v.specifies_shape)
    except NotImplementedError:
      # Attempt to find parameters heuristically.
      pnames = ()
      for p in self.parameters:
        if p in self._composite_tensor_shape_params:
          continue
        if tf.is_tensor(getattr(self, p, None)):
          pnames += (p,)
      return pnames

  @property
  def _composite_tensor_shape_params(self):
    """A tuple describing which parameters are shape-related tensors.

    Flattening in JAX involves many of the same considerations with regards to
    identifying tensor arguments for the purposes of CompositeTensor, except
    that shape-related items will be considered metadata.  This property
    identifies the keys of parameters that are expected to be shape-related
    tensors, so that they can be collected appropriately in CompositeTensor but
    not in JAX applications.
    """
    try:
      return tuple(k for k, v in self.parameter_properties().items()
                   if v.specifies_shape)
    except NotImplementedError:
      return ()


class _AutoCompositeTensorPsdKernelMeta(abc.ABCMeta):
  """Metaclass for `AutoCompositeTensorPsdKernel`."""

  def __new__(mcs, classname, baseclasses, attrs):  # pylint: disable=bad-mcs-classmethod-argument
    """Give subclasses their own type_spec, not an inherited one."""

    cls = super(_AutoCompositeTensorPsdKernelMeta, mcs).__new__(  # pylint: disable=too-many-function-args
        mcs, classname, baseclasses, attrs)
    return auto_composite_tensor.auto_composite_tensor(
        cls,
        omit_kwargs=('parameters',),
        non_identifying_kwargs=('name',),
        module_name='tfp.math._psdkernels')


class AutoCompositeTensorPsdKernel(PositiveSemidefiniteKernel,
                                   auto_composite_tensor.AutoCompositeTensor,
                                   metaclass=_AutoCompositeTensorPsdKernelMeta):
  pass


def _flatten_summand_list(kernels):
  """Flatten a list of kernels which may contain _SumKernel instances.

  Args:
    kernels: Python list of `PositiveSemidefiniteKernel` instances

  Returns:
    Python list containing the elements of kernels, with any _SumKernel
    instances replaced by their `kernels` property contents.
  """
  flattened = []
  for k in kernels:
    if isinstance(k, _NonCompositeTensorSumKernel):
      flattened += k.kernels
    else:
      flattened.append(k)
  return flattened


def _flatten_multiplicand_list(kernels):
  """Flatten a list of kernels which may contain _ProductKernel instances.

  Args:
    kernels: Python list of `PositiveSemidefiniteKernel` instances

  Returns:
    Python list containing the elements of kernels, with any _ProductKernel
    instances replaced by their `kernels` property contents.
  """
  flattened = []
  for k in kernels:
    if isinstance(k, _NonCompositeTensorProductKernel):
      flattened += k.kernels
    else:
      flattened.append(k)
  return flattened


class _NonCompositeTensorSumKernel(PositiveSemidefiniteKernel):
  """Kernel class representing summation over a list of kernels.

  Mathematically this class represents the pointwise sum of several kernels.
  Given two kernels, `k1` and `k2`, and `kp = _SumKernel([k1, k2])`, we have

    ```none
    kp.apply(x, y) = k1(x, y) + k2(x, y)
    ```

  for any `x`, `y` in the feature space (this presumes that the constituent
  kernels all act on the same feature space).

  That the sum is positive semi-definite follows simply from the definition of
  positive semi-definiteness of functions. If we have

    ```none
    sum_i sum_j (c[i]*) c[j] k1(x[i], x[j]) >= 0
    ```
  and

    ```none
    sum_i sum_j (c[i]*) c[j] k2(x[i], x[j]) >= 0
    ```

  for any finite collections `{x[1], ..., x[N]}` in S and `{c[1], ..., c[N]}` in
  the reals (or the complex plane), then we clearly also have the same for the
  sum of `k1` and `k2`.
  """

  def __init__(self, kernels, name=None):
    """Create a kernel which is the sum of `kernels`.

    The input list is 'flattened' in the sense that any entries which are also
    of type `_SumKernel` will have their list of kernels appended to this
    instance's list of kernels. This will reduce the stack depth when actually
    evaluating the sum over kernel applications.

    Args:
      kernels: Python `list` of `PositiveSemidefiniteKernel` instances.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      ValueError: `kernels` is an empty list, or `kernels` don't all have the
      same `feature_ndims`.
    """
    parameters = dict(locals())
    if not kernels:
      raise ValueError("Can't create _SumKernel over empty list.")
    ndims = kernels[0].feature_ndims
    for k in kernels[1:]:
      if k.feature_ndims != ndims:
        raise ValueError(
            "Can't sum kernels with different feature_ndims. Got:\n%s" %
            str([k.feature_ndims for k in kernels]))
    self._kernels = _flatten_summand_list(kernels)
    if name is None:
      name = 'SumKernel'
    # We have ensured the list is non-empty and all feature_ndims are the same.
    super(_NonCompositeTensorSumKernel, self).__init__(
        feature_ndims=kernels[0].feature_ndims,
        dtype=util.maybe_get_common_dtype(
            [None if k.dtype is None else k for k in kernels]),
        name=name,
        validate_args=any([k.validate_args for k in kernels]),
        parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        kernels=parameter_properties.BatchedComponentProperties(
            event_ndims=(
                lambda self: [0 for _ in self.kernels])))

  @property
  def kernels(self):
    """The list of kernels this _SumKernel sums over."""
    return self._kernels

  def _apply(self, x1, x2, example_ndims=0):
    return sum(k.apply(x1, x2, example_ndims) for k in self.kernels)

  def _matrix(self, x1, x2):
    return sum(k.matrix(x1, x2) for k in self.kernels)

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    return sum(
        k.tensor(
            x1, x2, x1_example_ndims, x2_example_ndims) for k in self.kernels)

  def _batch_shape(self):
    return functools.reduce(tf.broadcast_static_shape,
                            [k.batch_shape for k in self.kernels])

  def _batch_shape_tensor(self):
    return functools.reduce(tf.broadcast_dynamic_shape,
                            [k.batch_shape_tensor() for k in self.kernels])


class _SumKernel(_NonCompositeTensorSumKernel, AutoCompositeTensorPsdKernel):

  def __new__(cls, kernels, name=None):
    if cls is _SumKernel:
      if not all(isinstance(k, tf.__internal__.CompositeTensor)
                 for k in kernels):
        return _NonCompositeTensorSumKernel(kernels, name=name)
    return super(_SumKernel, cls).__new__(cls)

_SumKernel.__doc__ = _NonCompositeTensorSumKernel.__doc__ + '\n' + (
    'When an `_SumKernel` is constructed, if any element of its `kernels`'
    'list is not a `CompositeTensor` instance, a `_NonCompositeTensorSumKernel`'
    ' instance is returned instead.')


class _NonCompositeTensorProductKernel(PositiveSemidefiniteKernel):
  """Kernel class representing the product over a list of kernels.

  Mathematically this class represents the pointwise product of several kernels.
  Given two kernels, `k1` and `k2`, and `kp = _ProductKernel([k1, k2])`, we have

    ```none
    kp.apply(x, y) = k1(x, y) * k2(x, y)
    ```

  for any x, y in the feature space (this presumes that the constituent kernels
  all act on the same feature space).

  The fact that this product is still positive semi-definite can be shown in a
  variety of ways, many deep and all fascinating, but follows readily from the
  [Schur product theorem](https://en.wikipedia.org/wiki/Schur_product_theorem),
  which states that the Hadamard (element-wise) product of two PSD matrices is
  also PSD.
  """

  def __init__(self, kernels, name=None):
    """Create a kernel which is the product of `kernels`.

    The input list is 'flattened' in the sense that any entries which are also
    of type `_ProductKernel` will have their list of kernels appended to this
    instance's list of kernels. This will reduce the stack depth when actually
    evaluating the product over kernel applications.

    Args:
      kernels: Python `list` of `PositiveSemidefiniteKernel` instances.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      ValueError: `kernels` is an empty list, or `kernels` don't all have the
      same `feature_ndims`.
    """
    parameters = dict(locals())
    if not kernels:
      raise ValueError("Can't create _ProductKernel over empty list.")
    ndims = kernels[0].feature_ndims
    for k in kernels[1:]:
      if k.feature_ndims != ndims:
        raise ValueError(
            "Can't multiply kernels with different feature_ndims. Got:\n%s" %
            str([k.feature_ndims for k in kernels]))
    self._kernels = _flatten_multiplicand_list(kernels)
    if name is None:
      name = 'ProductKernel'
    # We have ensured the list is non-empty and all feature_ndims are the same.
    super(_NonCompositeTensorProductKernel, self).__init__(
        feature_ndims=kernels[0].feature_ndims,
        dtype=util.maybe_get_common_dtype(
            [None if k.dtype is None else k for k in kernels]),
        name=name,
        validate_args=any([k.validate_args for k in kernels]),
        parameters=parameters)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
    return dict(
        kernels=parameter_properties.BatchedComponentProperties(
            event_ndims=(
                lambda self: [0 for _ in self.kernels])))

  @property
  def kernels(self):
    """The list of kernels this _ProductKernel multiplies over."""
    return self._kernels

  def _apply(self, x1, x2, example_ndims=0):
    return functools.reduce(
        operator.mul,
        [k.apply(x1, x2, example_ndims) for k in self.kernels])

  def _matrix(self, x1, x2):
    return functools.reduce(
        operator.mul, [k.matrix(x1, x2) for k in self.kernels])

  def _tensor(self, x1, x2, x1_example_ndims, x2_example_ndims):
    return functools.reduce(
        operator.mul,
        [k.tensor(
            x1, x2, x1_example_ndims, x2_example_ndims) for k in self.kernels])

  def _batch_shape(self):
    return functools.reduce(tf.broadcast_static_shape,
                            [k.batch_shape for k in self.kernels])

  def _batch_shape_tensor(self):
    return functools.reduce(tf.broadcast_dynamic_shape,
                            [k.batch_shape_tensor() for k in self.kernels])


class _ProductKernel(_NonCompositeTensorProductKernel,
                     AutoCompositeTensorPsdKernel):

  def __new__(cls, kernels, name=None):
    if cls is _ProductKernel:
      if not all(isinstance(k, tf.__internal__.CompositeTensor)
                 for k in kernels):
        return _NonCompositeTensorProductKernel(kernels, name=name)
    return super(_ProductKernel, cls).__new__(cls)


_ProductKernel.__doc__ = _NonCompositeTensorProductKernel.__doc__ + '\n' + (
    'When an `_ProductKernel` is constructed, if any element of its `kernels`'
    'list is not a `CompositeTensor` instance, a '
    '`_NonCompositeTensorProductKernel` instance is returned instead.')
