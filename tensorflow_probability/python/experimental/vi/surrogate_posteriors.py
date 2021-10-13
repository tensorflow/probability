# Copyright 2019 The TensorFlow Probability Authors.
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
"""Utilities for constructing surrogate posteriors."""

import functools

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import chain
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import reshape
from tensorflow_probability.python.bijectors import restructure
from tensorflow_probability.python.bijectors import scale_matvec_linear_operator
from tensorflow_probability.python.bijectors import shift
from tensorflow_probability.python.bijectors import split
from tensorflow_probability.python.distributions import batch_broadcast
from tensorflow_probability.python.distributions import joint_distribution_util
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.experimental import util as tfe_util
from tensorflow_probability.python.experimental.vi.util import trainable_linear_operators
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import


def _get_event_shape_shallow_structure(event_shape):
  """Gets shallow structure, treating lists of ints at the leaves as atomic."""
  def _not_list_of_ints(s):
    if isinstance(s, list) or isinstance(s, tuple):
      return not all(isinstance(x, int) for x in s)
    return True

  return nest.get_traverse_shallow_structure(_not_list_of_ints, event_shape)


def build_factored_surrogate_posterior(  # pylint: disable=dangerous-default-value
    event_shape=None,
    bijector=None,
    batch_shape=(),
    base_distribution_cls=normal.Normal,
    initial_parameters={'scale': 1e-2},
    dtype=tf.float32,
    seed=None,
    validate_args=False,
    name=None):
  """Builds a joint variational posterior that factors over model variables.

  By default, this method creates an independent trainable Normal distribution
  for each variable, transformed using a bijector (if provided) to
  match the support of that variable. This makes extremely strong
  assumptions about the posterior: that it is approximately normal (or
  transformed normal), and that all model variables are independent.

  Args:
    event_shape: `Tensor` shape, or nested structure of `Tensor` shapes,
      specifying the event shape(s) of the posterior variables.
    bijector: Optional `tfb.Bijector` instance, or nested structure of such
      instances, defining support(s) of the posterior variables. The structure
      must match that of `event_shape` and may contain `None` values. A
      posterior variable will be modeled as
      `tfd.TransformedDistribution(underlying_dist, bijector)` if a
      corresponding constraining bijector is specified, otherwise it is modeled
      as supported on the unconstrained real line.
    batch_shape: The `batch_shape` of the output distribution.
      Default value: `()`.
    base_distribution_cls: Subclass of `tfd.Distribution` that is instantiated
      and optionally transformed by the bijector to define the component
      distributions. May optionally be a structure of such subclasses
      matching `event_shape`.
      Default value: `tfd.Normal`.
    initial_parameters: Optional `str : Tensor` dictionary specifying initial
      values for some or all of the base distribution's trainable parameters,
      or a Python `callable` with signature
      `value = parameter_init_fn(parameter_name, shape, dtype, seed,
      constraining_bijector)`, passed to `tfp.experimental.util.make_trainable`.
      May optionally be a structure matching `event_shape` of such dictionaries
      and/or callables. Dictionary entries that do not correspond to parameter
      names are ignored.
      Default value: `{'scale': 1e-2}` (ignored when `base_distribution` does
        not have a `scale` parameter).
    dtype: Optional float `dtype` for trainable parameters. May
      optionally be a structure of such `dtype`s matching `event_shape`.
      Default value: `tf.float32`.
    seed: PRNG seed; see `tfp.random.sanitize_seed` for details.
    validate_args: Python `bool`. Whether to validate input with asserts. This
      imposes a runtime cost. If `validate_args` is `False`, and the inputs are
      invalid, correct behavior is not guaranteed.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'build_factored_surrogate_posterior').

  Returns:
    surrogate_posterior: A `tfd.Distribution` instance whose samples have
      shape and structure matching that of `event_shape` or `initial_loc`.

  ### Examples

  Consider a Gamma model with unknown parameters, expressed as a joint
  Distribution:

  ```python
  Root = tfd.JointDistributionCoroutine.Root
  def model_fn():
    concentration = yield Root(tfd.Exponential(1.))
    rate = yield Root(tfd.Exponential(1.))
    y = yield tfd.Sample(tfd.Gamma(concentration=concentration, rate=rate),
                         sample_shape=4)
  model = tfd.JointDistributionCoroutine(model_fn)
  ```

  Let's use variational inference to approximate the posterior over the
  data-generating parameters for some observed `y`. We'll build a
  surrogate posterior distribution by specifying the shapes of the latent
  `rate` and `concentration` parameters, and that both are constrained to
  be positive.

  ```python
  surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
    event_shape=model.event_shape_tensor()[:-1],  # Omit the observed `y`.
    bijector=[tfb.Softplus(),   # Rate is positive.
              tfb.Softplus()])  # Concentration is positive.
  ```

  This creates a trainable joint distribution, defined by variables in
  `surrogate_posterior.trainable_variables`. We use `fit_surrogate_posterior`
  to fit this distribution by minimizing a divergence to the true posterior.

  ```python
  y = [0.2, 0.5, 0.3, 0.7]
  losses = tfp.vi.fit_surrogate_posterior(
    lambda rate, concentration: model.log_prob([rate, concentration, y]),
    surrogate_posterior=surrogate_posterior,
    num_steps=100,
    optimizer=tf.optimizers.Adam(0.1),
    sample_size=10)

  # After optimization, samples from the surrogate will approximate
  # samples from the true posterior.
  samples = surrogate_posterior.sample(100)
  posterior_mean = [tf.reduce_mean(x) for x in samples]     # mean ~= [1.1, 2.1]
  posterior_std = [tf.math.reduce_std(x) for x in samples]  # std  ~= [0.3, 0.8]
  ```

  If we wanted to initialize the optimization at a specific location, we can
  specify initial parameters when we build the surrogate posterior. Note that
  these parameterize the distribution(s) over unconstrained values,
  so we need to transform our desired constrained locations using the inverse
  of the constraining bijector(s).

  ```python
  surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
    event_shape=tf.nest.map_fn(tf.shape, initial_loc),
    bijector={'concentration': tfb.Softplus(),   # Rate is positive.
              'rate': tfb.Softplus()}   # Concentration is positive.
    initial_parameters={
      'concentration': {'loc': tfb.Softplus().inverse(0.4), 'scale': 1e-2},
      'rate': {'loc': tfb.Softplus().inverse(0.2), 'scale': 1e-2}})
  ```

  """
  with tf.name_scope(name or 'build_factored_surrogate_posterior'):
    # Convert event shapes to Tensors.
    shallow_structure = _get_event_shape_shallow_structure(event_shape)
    event_shape = nest.map_structure_up_to(
        shallow_structure, lambda s: tf.convert_to_tensor(s, dtype=tf.int32),
        event_shape)

    if nest.is_nested(bijector):
      event_space_bijector = joint_map.JointMap(
          nest.map_structure(lambda b: identity.Identity() if b is None else b,
                             nest_util.coerce_structure(event_shape, bijector)),
          validate_args=validate_args)
    else:
      event_space_bijector = bijector

    if event_space_bijector is None:
      unconstrained_event_shape = event_shape
    else:
      unconstrained_event_shape = (
          event_space_bijector.inverse_event_shape_tensor(event_shape))
    unconstrained_batch_and_event_shape = tf.nest.map_structure(
        lambda s: ps.concat([batch_shape, s], axis=0),
        unconstrained_event_shape)

    base_distribution_cls = nest_util.broadcast_structure(
        event_shape, base_distribution_cls)
    try:
      # Check that we have initial parameters for each event part.
      nest.assert_shallow_structure(event_shape, initial_parameters)
    except (ValueError, TypeError):
      # If not, broadcast the parameters to match the event structure.
      # We do this manually rather than using `nest_util.broadcast_structure`
      # because the initial parameters can themselves be structures (dicts).
      initial_parameters = nest.map_structure(lambda x: initial_parameters,
                                              event_shape)

    unconstrained_trainable_distributions = (
        nest_util.map_structure_with_named_args(
            tfe_util.make_trainable,
            cls=base_distribution_cls,
            initial_parameters=initial_parameters,
            batch_and_event_shape=unconstrained_batch_and_event_shape,
            parameter_dtype=nest_util.broadcast_structure(event_shape, dtype),
            seed=tf.nest.pack_sequence_as(
                event_shape,
                samplers.split_seed(seed,
                                    n=len(tf.nest.flatten(event_shape)))),
            _up_to=event_shape))
    unconstrained_trainable_distribution = (
        joint_distribution_util.independent_joint_distribution_from_structure(
            unconstrained_trainable_distributions,
            batch_ndims=ps.rank_from_shape(batch_shape),
            validate_args=validate_args))
    if event_space_bijector is None:
      return unconstrained_trainable_distribution
    return transformed_distribution.TransformedDistribution(
        unconstrained_trainable_distribution, event_space_bijector)


def build_affine_surrogate_posterior(
    event_shape,
    operators='diag',
    bijector=None,
    base_distribution=normal.Normal,
    dtype=tf.float32,
    batch_shape=(),
    seed=None,
    validate_args=False,
    name=None):
  """Builds a joint variational posterior with a given `event_shape`.

  This function builds a surrogate posterior by applying a trainable
  transformation to a standard base distribution and constraining the samples
  with `bijector`. The surrogate posterior has event shape equal to
  the input `event_shape`.

  This function is a convenience wrapper around
  `build_affine_surrogate_posterior_from_base_distribution` that allows the
  user to pass in the desired posterior `event_shape` instead of
  pre-constructed base distributions (at the expense of full control over the
  base distribution types and parameterizations).

  Args:
    event_shape: (Nested) event shape of the posterior.
    operators: Either a string or a list/tuple containing `LinearOperator`
      subclasses, `LinearOperator` instances, or callables returning
      `LinearOperator` instances. Supported string values are "diag" (to create
      a mean-field surrogate posterior) and "tril" (to create a full-covariance
      surrogate posterior). A list/tuple may be passed to induce other
      posterior covariance structures. If the list is flat, a
      `tf.linalg.LinearOperatorBlockDiag` instance will be created and applied
      to the base distribution. Otherwise the list must be singly-nested and
      have a first element of length 1, second element of length 2, etc.; the
      elements of the outer list are interpreted as rows of a lower-triangular
      block structure, and a `tf.linalg.LinearOperatorBlockLowerTriangular`
      instance is created. For complete documentation and examples, see
      `tfp.experimental.vi.util.build_trainable_linear_operator_block`, which
      receives the `operators` arg if it is list-like.
      Default value: `"diag"`.
    bijector: `tfb.Bijector` instance, or nested structure of `tfb.Bijector`
      instances, that maps (nested) values in R^n to the support of the
      posterior. (This can be the `experimental_default_event_space_bijector` of
      the distribution over the prior latent variables.)
      Default value: `None` (i.e., the posterior is over R^n).
    base_distribution: A `tfd.Distribution` subclass parameterized by `loc` and
      `scale`. The base distribution of the transformed surrogate has `loc=0.`
      and `scale=1.`.
      Default value: `tfd.Normal`.
    dtype: The `dtype` of the surrogate posterior.
      Default value: `tf.float32`.
    batch_shape: Batch shape (Python tuple, list, or int) of the surrogate
      posterior, to enable parallel optimization from multiple initializations.
      Default value: `()`.
    seed: Python integer to seed the random number generator for initial values.
      Default value: `None`.
    validate_args: Python `bool`. Whether to validate input with asserts. This
      imposes a runtime cost. If `validate_args` is `False`, and the inputs are
      invalid, correct behavior is not guaranteed.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'build_affine_surrogate_posterior').

  Returns:
    surrogate_distribution: Trainable `tfd.Distribution` with event shape equal
      to `event_shape`.

  #### Examples

  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  # Define a joint probabilistic model.
  Root = tfd.JointDistributionCoroutine.Root
  def model_fn():
    concentration = yield Root(tfd.Exponential(1.))
    rate = yield Root(tfd.Exponential(1.))
    y = yield tfd.Sample(
        tfd.Gamma(concentration=concentration, rate=rate),
        sample_shape=4)
  model = tfd.JointDistributionCoroutine(model_fn)

  # Assume the `y` are observed, such that the posterior is a joint distribution
  # over `concentration` and `rate`. The posterior event shape is then equal to
  # the first two components of the model's event shape.
  posterior_event_shape = model.event_shape_tensor()[:-1]

  # Constrain the posterior values to be positive using the `Exp` bijector.
  bijector = [tfb.Exp(), tfb.Exp()]

  # Build a full-covariance surrogate posterior.
  surrogate_posterior = (
    tfp.experimental.vi.build_affine_surrogate_posterior(
        event_shape=posterior_event_shape,
        operators='tril',
        bijector=bijector))

  # For an example defining `'operators'` as a list to express an alternative
  # covariance structure, see
  # `build_affine_surrogate_posterior_from_base_distribution`.

  # Fit the model.
  y = [0.2, 0.5, 0.3, 0.7]
  target_model = model.experimental_pin(y=y)
  losses = tfp.vi.fit_surrogate_posterior(
      target_model.unnormalized_log_prob,
      surrogate_posterior,
      num_steps=100,
      optimizer=tf.optimizers.Adam(0.1),
      sample_size=10)
  ```
  """
  with tf.name_scope(name or 'build_affine_surrogate_posterior'):

    event_shape = nest.map_structure_up_to(
        _get_event_shape_shallow_structure(event_shape),
        lambda s: tf.convert_to_tensor(s, dtype=tf.int32),
        event_shape)

    if nest.is_nested(bijector):
      bijector = joint_map.JointMap(
          nest.map_structure(
              lambda b: identity.Identity() if b is None else b,
              bijector), validate_args=validate_args)

    if bijector is None:
      unconstrained_event_shape = event_shape
    else:
      unconstrained_event_shape = (
          bijector.inverse_event_shape_tensor(event_shape))

    standard_base_distribution = nest.map_structure(
        lambda s: base_distribution(loc=tf.zeros([], dtype=dtype), scale=1.),
        unconstrained_event_shape)
    standard_base_distribution = nest.map_structure(
        lambda d, s: (  # pylint: disable=g-long-lambda
            sample.Sample(d, sample_shape=s, validate_args=validate_args)
            if distribution_util.shape_may_be_nontrivial(s)
            else d),
        standard_base_distribution,
        unconstrained_event_shape)
    if distribution_util.shape_may_be_nontrivial(batch_shape):
      standard_base_distribution = nest.map_structure(
          lambda d: batch_broadcast.BatchBroadcast(  # pylint: disable=g-long-lambda
              d, to_shape=batch_shape, validate_args=validate_args),
          standard_base_distribution)

    return build_affine_surrogate_posterior_from_base_distribution(
        standard_base_distribution,
        operators=operators,
        bijector=bijector,
        seed=seed,
        validate_args=validate_args)


# Default constructors for
# `build_affine_surrogate_posterior_from_base_distribution`.
_sample_uniform_initial_loc = functools.partial(
    samplers.uniform, minval=-2., maxval=2., dtype=tf.float32)


def build_affine_surrogate_posterior_from_base_distribution(
    base_distribution,
    operators='diag',
    bijector=None,
    initial_unconstrained_loc_fn=_sample_uniform_initial_loc,
    seed=None,
    validate_args=False,
    name=None):
  """Builds a variational posterior by linearly transforming base distributions.

  This function builds a surrogate posterior by applying a trainable
  transformation to a base distribution (typically a `tfd.JointDistribution`) or
  nested structure of base distributions, and constraining the samples with
  `bijector`. Note that the distributions must have event shapes corresponding
  to the *pretransformed* surrogate posterior -- that is, if `bijector` contains
  a shape-changing bijector, then the corresponding base distribution event
  shape is the inverse event shape of the bijector applied to the desired
  surrogate posterior shape. The surrogate posterior is constucted as follows:

  1. Flatten the base distribution event shapes to vectors, and pack the base
     distributions into a `tfd.JointDistribution`.
  2. Apply a trainable blockwise LinearOperator bijector to the joint base
     distribution.
  3. Apply the constraining bijectors and return the resulting trainable
     `tfd.TransformedDistribution` instance.

  Args:
    base_distribution: `tfd.Distribution` instance (typically a
      `tfd.JointDistribution`), or a nested structure of `tfd.Distribution`
      instances.
    operators: Either a string or a list/tuple containing `LinearOperator`
      subclasses, `LinearOperator` instances, or callables returning
      `LinearOperator` instances. Supported string values are "diag" (to create
      a mean-field surrogate posterior) and "tril" (to create a full-covariance
      surrogate posterior). A list/tuple may be passed to induce other
      posterior covariance structures. If the list is flat, a
      `tf.linalg.LinearOperatorBlockDiag` instance will be created and applied
      to the base distribution. Otherwise the list must be singly-nested and
      have a first element of length 1, second element of length 2, etc.; the
      elements of the outer list are interpreted as rows of a lower-triangular
      block structure, and a `tf.linalg.LinearOperatorBlockLowerTriangular`
      instance is created. For complete documentation and examples, see
      `tfp.experimental.vi.util.build_trainable_linear_operator_block`, which
      receives the `operators` arg if it is list-like.
      Default value: `"diag"`.
    bijector: `tfb.Bijector` instance, or nested structure of `tfb.Bijector`
      instances, that maps (nested) values in R^n to the support of the
      posterior. (This can be the `experimental_default_event_space_bijector` of
      the distribution over the prior latent variables.)
      Default value: `None` (i.e., the posterior is over R^n).
    initial_unconstrained_loc_fn: Optional Python `callable` with signature
      `initial_loc = initial_unconstrained_loc_fn(shape, dtype, seed)` used to
      sample real-valued initializations for the unconstrained location of
      each variable.
      Default value: `functools.partial(tf.random.stateless_uniform,
        minval=-2., maxval=2., dtype=tf.float32)`.
    seed: Python integer to seed the random number generator for initial values.
      Default value: `None`.
    validate_args: Python `bool`. Whether to validate input with asserts. This
      imposes a runtime cost. If `validate_args` is `False`, and the inputs are
      invalid, correct behavior is not guaranteed.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e.,
      'build_affine_surrogate_posterior_from_base_distribution').

  Returns:
    surrogate_distribution: Trainable `tfd.JointDistribution` instance.
  Raises:
    NotImplementedError: Base distributions with mixed dtypes are not supported.

  #### Examples
  ```python
  tfd = tfp.distributions
  tfb = tfp.bijectors

  # Fit a multivariate Normal surrogate posterior on the Eight Schools model
  # [1].

  treatment_effects = [28., 8., -3., 7., -1., 1., 18., 12.]
  treatment_stddevs = [15., 10., 16., 11., 9., 11., 10., 18.]

  def model_fn():
    avg_effect = yield tfd.Normal(loc=0., scale=10., name='avg_effect')
    log_stddev = yield tfd.Normal(loc=5., scale=1., name='log_stddev')
    school_effects = yield tfd.Sample(
        tfd.Normal(loc=avg_effect, scale=tf.exp(log_stddev)),
        sample_shape=[8],
        name='school_effects')
    treatment_effects = yield tfd.Independent(
        tfd.Normal(loc=school_effects, scale=treatment_stddevs),
        reinterpreted_batch_ndims=1,
        name='treatment_effects')
  model = tfd.JointDistributionCoroutineAutoBatched(model_fn)

  # Pin the observed values in the model.
  target_model = model.experimental_pin(treatment_effects=treatment_effects)

  # Define a lower triangular structure of `LinearOperator` subclasses that
  # models full covariance among latent variables except for the 8 dimensions
  # of `school_effect`, which are modeled as independent (using
  # `LinearOperatorDiag`).
  operators = [
    [tf.linalg.LinearOperatorLowerTriangular],
    [tf.linalg.LinearOperatorFullMatrix, LinearOperatorLowerTriangular],
    [tf.linalg.LinearOperatorFullMatrix, LinearOperatorFullMatrix,
     tf.linalg.LinearOperatorDiag]]


  # Constrain the posterior values to the support of the prior.
  bijector = target_model.experimental_default_event_space_bijector()

  # Build a full-covariance surrogate posterior.
  surrogate_posterior = (
    tfp.experimental.vi.build_affine_surrogate_posterior_from_base_distribution(
        base_distribution=base_distribution,
        operators=operators,
        bijector=bijector))

  # Fit the model.
  losses = tfp.vi.fit_surrogate_posterior(
      target_model.unnormalized_log_prob,
      surrogate_posterior,
      num_steps=100,
      optimizer=tf.optimizers.Adam(0.1),
      sample_size=10)
  ```

  #### References

  [1] Andrew Gelman, John Carlin, Hal Stern, David Dunson, Aki Vehtari, and
      Donald Rubin. Bayesian Data Analysis, Third Edition.
      Chapman and Hall/CRC, 2013.

  """
  with tf.name_scope(
      name or 'build_affine_surrogate_posterior_from_base_distribution'):

    if nest.is_nested(base_distribution):
      base_distribution = (
          joint_distribution_util.independent_joint_distribution_from_structure(
              base_distribution, validate_args=validate_args))

    if nest.is_nested(bijector):
      bijector = joint_map.JointMap(
          nest.map_structure(
              lambda b: identity.Identity() if b is None else b, bijector),
          validate_args=validate_args)

    batch_shape = base_distribution.batch_shape_tensor()
    if tf.nest.is_nested(batch_shape):  # Base is a classic JointDistribution.
      batch_shape = functools.reduce(ps.broadcast_shape,
                                     tf.nest.flatten(batch_shape))
    event_shape = base_distribution.event_shape_tensor()
    flat_event_size = nest.flatten(
        nest.map_structure(ps.reduce_prod, event_shape))

    base_dtypes = set(nest.flatten(base_distribution.dtype))
    if len(base_dtypes) > 1:
      raise NotImplementedError(
          'Base distributions with mixed dtype are not supported. Saw '
          'components of dtype {}'.format(base_dtypes))
    base_dtype = list(base_dtypes)[0]

    num_components = len(flat_event_size)
    if operators == 'diag':
      operators = [tf.linalg.LinearOperatorDiag] * num_components
    elif operators == 'tril':
      operators = [
          [tf.linalg.LinearOperatorFullMatrix] * i
          + [tf.linalg.LinearOperatorLowerTriangular]
          for i in range(num_components)]
    elif isinstance(operators, str):
      raise ValueError(
          'Unrecognized operator type {}. Valid operators are "diag", "tril", '
          'or a structure that can be passed to '
          '`tfp.experimental.vi.util.build_trainable_linear_operator_block` as '
          'the `operators` arg.'.format(operators))

    if nest.is_nested(operators):
      seed, operators_seed = samplers.split_seed(seed)
      operators = (
          trainable_linear_operators.build_trainable_linear_operator_block(
              operators,
              block_dims=flat_event_size,
              dtype=base_dtype,
              batch_shape=batch_shape,
              seed=operators_seed))

    linop_bijector = (
        scale_matvec_linear_operator.ScaleMatvecLinearOperatorBlock(
            scale=operators, validate_args=validate_args))
    loc_bijector = joint_map.JointMap(
        tf.nest.map_structure(
            lambda s, seed: shift.Shift(  # pylint: disable=g-long-lambda
                tf.Variable(
                    initial_unconstrained_loc_fn(
                        ps.concat([batch_shape, [s]], axis=0),
                        dtype=base_dtype,
                        seed=seed))),
            flat_event_size,
            samplers.split_seed(seed, n=len(flat_event_size))),
        validate_args=validate_args)

    unflatten_and_reshape = chain.Chain(
        [joint_map.JointMap(
            nest.map_structure(reshape.Reshape, event_shape),
            validate_args=validate_args),
         restructure.Restructure(
             nest.pack_sequence_as(event_shape, range(num_components)))],
        validate_args=validate_args)

    bijectors = [] if bijector is None else [bijector]
    bijectors.extend(
        [unflatten_and_reshape,
         loc_bijector,  # Allow the mean of the standard dist to shift from 0.
         linop_bijector])  # Apply LinOp to scale the standard dist.
    bijector = chain.Chain(bijectors, validate_args=validate_args)

    flat_base_distribution = invert.Invert(
        unflatten_and_reshape)(base_distribution)

    return transformed_distribution.TransformedDistribution(
        flat_base_distribution, bijector=bijector, validate_args=validate_args)


def build_split_flow_surrogate_posterior(
    event_shape,
    trainable_bijector,
    constraining_bijector=None,
    base_distribution=normal.Normal,
    batch_shape=(),
    dtype=tf.float32,
    validate_args=False,
    name=None):
  """Builds a joint variational posterior by splitting a normalizing flow.

  Args:
    event_shape: (Nested) event shape of the surrogate posterior.
    trainable_bijector: A trainable `tfb.Bijector` instance that operates on
      `Tensor`s (not structures), e.g. `tfb.MaskedAutoregressiveFlow` or
      `tfb.RealNVP`. This bijector transforms the base distribution before it is
      split.
    constraining_bijector: `tfb.Bijector` instance, or nested structure of
      `tfb.Bijector` instances, that maps (nested) values in R^n to the support
      of the posterior. (This can be the
      `experimental_default_event_space_bijector` of the distribution over the
      prior latent variables.)
      Default value: `None` (i.e., the posterior is over R^n).
    base_distribution: A `tfd.Distribution` subclass parameterized by `loc` and
      `scale`. The base distribution for the transformed surrogate has `loc=0.`
      and `scale=1.`.
      Default value: `tfd.Normal`.
    batch_shape: The `batch_shape` of the output distribution.
      Default value: `()`.
    dtype: The `dtype` of the surrogate posterior.
      Default value: `tf.float32`.
    validate_args: Python `bool`. Whether to validate input with asserts. This
      imposes a runtime cost. If `validate_args` is `False`, and the inputs are
      invalid, correct behavior is not guaranteed.
      Default value: `False`.
    name: Python `str` name prefixed to ops created by this function.
      Default value: `None` (i.e., 'build_split_flow_surrogate_posterior').

  Returns:
    surrogate_distribution: Trainable `tfd.TransformedDistribution` with event
      shape equal to `event_shape`.

  ### Examples
  ```python

  # Train a normalizing flow on the Eight Schools model [1].

  treatment_effects = [28., 8., -3., 7., -1., 1., 18., 12.]
  treatment_stddevs = [15., 10., 16., 11., 9., 11., 10., 18.]
  model = tfd.JointDistributionNamed({
      'avg_effect':
          tfd.Normal(loc=0., scale=10., name='avg_effect'),
      'log_stddev':
          tfd.Normal(loc=5., scale=1., name='log_stddev'),
      'school_effects':
          lambda log_stddev, avg_effect: (
              tfd.Independent(
                  tfd.Normal(
                      loc=avg_effect[..., None] * tf.ones(8),
                      scale=tf.exp(log_stddev[..., None]) * tf.ones(8),
                      name='school_effects'),
                  reinterpreted_batch_ndims=1)),
      'treatment_effects': lambda school_effects: tfd.Independent(
          tfd.Normal(loc=school_effects, scale=treatment_stddevs),
          reinterpreted_batch_ndims=1)
  })

  # Pin the observed values in the model.
  target_model = model.experimental_pin(treatment_effects=treatment_effects)

  # Create a Masked Autoregressive Flow bijector.
  net = tfb.AutoregressiveNetwork(2, hidden_units=[16, 16], dtype=tf.float32)
  maf = tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=net)

  # Build and fit the surrogate posterior.
  surrogate_posterior = (
      tfp.experimental.vi.build_split_flow_surrogate_posterior(
          event_shape=target_model.event_shape_tensor(),
          trainable_bijector=maf,
          constraining_bijector=(
              target_model.experimental_default_event_space_bijector())))

  losses = tfp.vi.fit_surrogate_posterior(
      target_model.unnormalized_log_prob,
      surrogate_posterior,
      num_steps=100,
      optimizer=tf.optimizers.Adam(0.1),
      sample_size=10)
  ```

  #### References

  [1] Andrew Gelman, John Carlin, Hal Stern, David Dunson, Aki Vehtari, and
      Donald Rubin. Bayesian Data Analysis, Third Edition.
      Chapman and Hall/CRC, 2013.

  """
  with tf.name_scope(name or 'build_split_flow_surrogate_posterior'):

    shallow_structure = _get_event_shape_shallow_structure(event_shape)
    event_shape = nest.map_structure_up_to(
        shallow_structure, ps.convert_to_shape_tensor, event_shape)

    if nest.is_nested(constraining_bijector):
      constraining_bijector = joint_map.JointMap(
          nest.map_structure(
              lambda b: identity.Identity() if b is None else b,
              constraining_bijector), validate_args=validate_args)

    if constraining_bijector is None:
      unconstrained_event_shape = event_shape
    else:
      unconstrained_event_shape = (
          constraining_bijector.inverse_event_shape_tensor(event_shape))

    flat_base_event_shape = nest.flatten(unconstrained_event_shape)
    flat_base_event_size = nest.map_structure(
        tf.reduce_prod, flat_base_event_shape)
    event_size = tf.reduce_sum(flat_base_event_size)

    base_distribution = sample.Sample(
        base_distribution(tf.zeros(batch_shape, dtype=dtype), scale=1.),
        [event_size])

    # After transforming base distribution samples with `trainable_bijector`,
    # split them into vector-valued components.
    split_bijector = split.Split(
        flat_base_event_size, validate_args=validate_args)

    # Reshape the vectors to the correct posterior event shape.
    event_reshape = joint_map.JointMap(
        nest.map_structure(reshape.Reshape, unconstrained_event_shape),
        validate_args=validate_args)

    # Restructure the flat list of components to the correct posterior
    # structure.
    event_unflatten = restructure.Restructure(
        nest.pack_sequence_as(
            unconstrained_event_shape, range(len(flat_base_event_shape))))

    bijectors = [] if constraining_bijector is None else [constraining_bijector]
    bijectors.extend(
        [event_reshape, event_unflatten, split_bijector, trainable_bijector])
    bijector = chain.Chain(bijectors, validate_args=validate_args)

    return transformed_distribution.TransformedDistribution(
        base_distribution, bijector=bijector, validate_args=validate_args)
