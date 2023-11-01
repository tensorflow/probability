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
"""Tests for surrogate posteriors."""

# Dependency imports

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import exp
from tensorflow_probability.python.bijectors import fill_triangular
from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import masked_autoregressive as maf_lib
from tensorflow_probability.python.bijectors import sigmoid
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import exponential
from tensorflow_probability.python.distributions import gamma
from tensorflow_probability.python.distributions import independent
from tensorflow_probability.python.distributions import joint_distribution_coroutine as jdc
from tensorflow_probability.python.distributions import joint_distribution_named as jdn
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import laplace
from tensorflow_probability.python.distributions import logistic
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import sample
from tensorflow_probability.python.distributions import student_t
from tensorflow_probability.python.experimental.vi import surrogate_posteriors
from tensorflow_probability.python.internal import nest_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.vi import optimization

from tensorflow.python.util import nest  # pylint: disable=g-direct-tensorflow-import

JAX_MODE = False


def _as_concrete_instance(d):
  """Forces evaluation of a DeferredModule object."""
  if hasattr(d, '_build_module'):
    return d._build_module()
  return d


class _SurrogatePosterior(object):
  """Common methods for testing ADVI surrogate posteriors."""

  def _initialize_surrogate(
      self, family_str, seed, is_stateless=JAX_MODE, **kwargs):
    """Initializes a stateless (JAX) or stateful (TF) surrogate posterior."""
    # TODO(davmre): refactor to support testing gradients of stateless
    # surrogates.
    if is_stateless:
      build_fn = getattr(surrogate_posteriors, family_str + '_stateless')
      init_fn, apply_fn = build_fn(**kwargs)
      return apply_fn(init_fn(seed=seed))
    else:
      build_fn = getattr(surrogate_posteriors, family_str)
      surrogate_posterior = build_fn(seed=seed, **kwargs)
      self.evaluate([v.initializer
                     for v in surrogate_posterior.trainable_variables])
      return surrogate_posterior

  def _test_shapes(self, surrogate_posterior, batch_shape, event_shape, seed):

    posterior_sample = surrogate_posterior.sample(seed=seed)

    # Test that the posterior has the specified event shape(s).
    posterior_event_shape = self.evaluate(
        surrogate_posterior.event_shape_tensor())
    event_shape = nest.map_structure_up_to(
        surrogate_posterior.event_shape_tensor(),
        ps.convert_to_shape_tensor,
        event_shape,
        check_types=False)
    self.assertAllEqualNested(event_shape, posterior_event_shape)

    # Test that all sample Tensors have the expected shapes.
    sample_shape = nest.map_structure(
        lambda s: ps.concat([batch_shape, s], axis=0), event_shape)
    self.assertAllEqualNested(
        nest.map_structure(ps.shape, posterior_sample),
        sample_shape)

    # Test that the posterior components have the specified batch shape, which
    # is also the shape of `log_prob` of a sample.
    self.assertAllAssertsNested(
        lambda s: self.assertAllEqual(s, batch_shape),
        self.evaluate(surrogate_posterior.batch_shape_tensor()))
    self.assertAllEqual(
        batch_shape, ps.shape(surrogate_posterior.log_prob(posterior_sample)))

  def _test_gradients(self, surrogate_posterior, seed):

    # Test that gradients are available wrt the variational parameters.
    self.assertNotEmpty(surrogate_posterior.trainable_variables)

    posterior_sample = self.evaluate(surrogate_posterior.sample(seed=seed))
    with tf.GradientTape() as tape:
      posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    grad = tape.gradient(posterior_logprob,
                         surrogate_posterior.trainable_variables)
    self.assertTrue(all(g is not None for g in grad))

  def _test_dtype(self, surrogate_posterior, dtype, seed):
    self.assertAllAssertsNested(
        lambda t: self.assertDTypeEqual(t, dtype),
        surrogate_posterior.sample(seed=seed))

  def _test_fitting(self, model, surrogate_posterior):

    # Fit model.
    y = [0.2, 0.5, 0.3, 0.7]
    losses = optimization.fit_surrogate_posterior(
        lambda rate, concentration: model.log_prob((rate, concentration, y)),
        surrogate_posterior,
        num_steps=5,  # Don't optimize to completion.
        optimizer=tf.keras.optimizers.Adam(0.1),
        sample_size=10)

    # Compute posterior statistics.
    with tf.control_dependencies([losses]):
      posterior_samples = surrogate_posterior.sample(100)
      posterior_mean = [tf.reduce_mean(x) for x in posterior_samples]
      posterior_stddev = [tf.math.reduce_std(x) for x in posterior_samples]

    self.evaluate(tf1.global_variables_initializer())
    self.evaluate([losses, posterior_mean, posterior_stddev])

  def _make_gamma_model(self):
    Root = jdc.JointDistributionCoroutine.Root  # pylint: disable=invalid-name

    def model_fn():
      concentration = yield Root(exponential.Exponential(1.))
      rate = yield Root(exponential.Exponential(1.))
      y = yield sample.Sample(  # pylint: disable=unused-variable
          gamma.Gamma(concentration=concentration, rate=rate),
          sample_shape=4)

    return jdc.JointDistributionCoroutine(model_fn)


@test_util.test_all_tf_execution_regimes
class FactoredSurrogatePosterior(test_util.TestCase, _SurrogatePosterior):

  @parameterized.named_parameters(
      {
          'testcase_name': 'TensorEvent',
          'event_shape': tf.TensorShape([4]),
          'bijector': sigmoid.Sigmoid(),
          'dtype': np.float64,
          'is_static': True
      },
      {
          'testcase_name': 'ListEventStateless',
          'event_shape':
              [tf.TensorShape([3]),
               tf.TensorShape([]),
               tf.TensorShape([2, 2])],
          'bijector':
              [softplus.Softplus(), None,
               fill_triangular.FillTriangular()],
          'dtype': np.float32,
          'is_static': False,
          'is_stateless': True
      },
      {
          'testcase_name': 'DictEvent',
          'event_shape': {
              'x': tf.TensorShape([1]),
              'y': tf.TensorShape([])
          },
          'bijector': None,
          'dtype': np.float64,
          'is_static': True
      },
      {
          'testcase_name': 'NestedEvent',
          'event_shape': {
              'x': [tf.TensorShape([1]),
                    tf.TensorShape([1, 2])],
              'y': tf.TensorShape([])
          },
          'bijector': {
              'x': [identity.Identity(),
                    softplus.Softplus()],
              'y': sigmoid.Sigmoid()
          },
          'dtype': np.float32,
          'is_static': True
      },
  )
  def test_specifying_event_shape(
      self, event_shape, bijector, dtype, is_static, is_stateless=JAX_MODE):
    init_seed, grads_seed, shapes_seed, dtype_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'),
        n=4)

    surrogate_posterior = self._initialize_surrogate(
        'build_factored_surrogate_posterior',
        is_stateless=is_stateless,
        seed=init_seed,
        event_shape=tf.nest.map_structure(
            lambda s: self.maybe_static(  # pylint: disable=g-long-lambda
                np.array(s, dtype=np.int32), is_static=is_static),
            event_shape),
        bijector=bijector,
        dtype=dtype,
        validate_args=True)

    self._test_shapes(
        surrogate_posterior, batch_shape=[], event_shape=event_shape,
        seed=shapes_seed)
    self._test_dtype(surrogate_posterior, dtype, seed=dtype_seed)
    if not is_stateless:
      self._test_gradients(surrogate_posterior, seed=grads_seed)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TensorEventStateless',
          'event_shape': [4],
          'initial_loc': np.array([[[0.9, 0.1, 0.5, 0.7]]]),
          'batch_shape': [1, 1],
          'bijector': sigmoid.Sigmoid(),
          'dtype': np.float32,
          'is_static': False,
          'is_stateless': True
      },
      {
          'testcase_name': 'ListEvent',
          'event_shape': [[3], [], [2, 2]],
          'initial_loc':
              [np.array([0.1, 7., 3.]), 0.1,
               np.array([[1., 0], [-4., 2.]])],
          'batch_shape': [],
          'bijector':
              [softplus.Softplus(), None,
               fill_triangular.FillTriangular()],
          'dtype': np.float64,
          'is_static': True
      },
      {
          'testcase_name': 'DictEvent',
          'event_shape': {
              'x': [2],
              'y': []
          },
          'initial_loc': {
              'x': np.array([[0.9, 1.2]]),
              'y': np.array([-4.1])
          },
          'batch_shape': [1],
          'bijector': None,
          'dtype': np.float32,
          'is_static': False
      },
      {
          'testcase_name': 'ExplicitBatchShape',
          'event_shape': [[3], [4]],
          'initial_loc': [0., 0.],
          'batch_shape': [5, 1],
          'bijector': None,
          'dtype': np.float32,
          'is_static': False
      },
  )
  def test_specifying_initial_loc(
      self, event_shape, initial_loc, batch_shape, bijector, dtype,
      is_static, is_stateless=JAX_MODE):

    init_seed, grads_seed, shapes_seed, dtype_seed, sample_seed = (
        samplers.split_seed(test_util.test_seed(sampler_type='stateless'), n=5))

    initial_loc = tf.nest.map_structure(
        lambda s: self.maybe_static(  # pylint: disable=g-long-lambda
            np.array(s, dtype=dtype), is_static=is_static),
        initial_loc)

    if bijector is not None:
      initial_unconstrained_loc = tf.nest.map_structure(
          lambda x, b: x if b is None else b.inverse(x),
          initial_loc, bijector)
    else:
      initial_unconstrained_loc = initial_loc

    surrogate_posterior = self._initialize_surrogate(
        'build_factored_surrogate_posterior',
        is_stateless=is_stateless,
        seed=init_seed,
        event_shape=event_shape,
        initial_parameters=tf.nest.map_structure(
            lambda x: {'loc': x, 'scale': 1e-6},
            initial_unconstrained_loc),
        batch_shape=batch_shape,
        dtype=dtype,
        bijector=bijector,
        validate_args=True)

    self._test_shapes(
        surrogate_posterior, batch_shape=batch_shape,
        event_shape=event_shape, seed=shapes_seed)
    self._test_dtype(surrogate_posterior, dtype, seed=dtype_seed)
    if not is_stateless:
      self._test_gradients(surrogate_posterior, seed=grads_seed)

    # Check that the sampled values are close to the initial locs.
    posterior_sample_ = self.evaluate(
        surrogate_posterior.sample(seed=sample_seed))
    self.assertAllCloseNested(
        tf.nest.map_structure(lambda x, y: tf.broadcast_to(x, ps.shape(y)),
                              initial_loc, posterior_sample_),
        posterior_sample_,
        atol=1e-4)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TensorEventAllDeterministic',
          'event_shape': [4],
          'base_distribution_cls': deterministic.Deterministic
      },
      {
          'testcase_name': 'ListEventSingleDeterministicStateless',
          'event_shape': [[3], [], [2, 2]],
          'base_distribution_cls': deterministic.Deterministic,
          'is_stateless': True
      },
      {
          'testcase_name':
              'ListEventDeterministicNormalStudentT',
          'event_shape': [[3], [], [2, 2]],
          'base_distribution_cls':
              [deterministic.Deterministic, normal.Normal, student_t.StudentT]
      },
  )
  def test_specifying_distribution_type(
      self, event_shape, base_distribution_cls, is_stateless=JAX_MODE):
    init_seed, sample_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'), n=2)
    surrogate_posterior = self._initialize_surrogate(
        'build_factored_surrogate_posterior',
        is_stateless=is_stateless,
        seed=init_seed,
        event_shape=event_shape,
        base_distribution_cls=base_distribution_cls,
        validate_args=True)

    # Test that the surrogate uses the expected distribution types.
    if tf.nest.is_nested(surrogate_posterior.event_shape):
      ds, _ = surrogate_posterior.sample_distributions(seed=sample_seed)
    else:
      ds = [surrogate_posterior]
    for cls, d in zip(
        nest_util.broadcast_structure(ds, base_distribution_cls), ds):
      d = _as_concrete_instance(d)
      while isinstance(d, independent.Independent):
        d = _as_concrete_instance(d.distribution)
      self.assertIsInstance(d, cls)

  @test_util.jax_disable_variable_test
  def test_that_gamma_fitting_example_runs(self):
    model = self._make_gamma_model()
    surrogate_posterior = (
        surrogate_posteriors.build_factored_surrogate_posterior(
            event_shape=model.event_shape_tensor()[:-1],
            bijector=[softplus.Softplus(),
                      softplus.Softplus()]))
    self._test_fitting(model, surrogate_posterior)

  def test_can_jit_creation_of_stateless_surrogate_posterior(self):
    seed = test_util.test_seed(sampler_type='stateless')
    if not JAX_MODE:
      self.skipTest('Test is specific to JAX.')
    import jax  # pylint: disable=g-import-not-at-top

    @jax.jit
    def init(seed):
      model = normal.Normal(0., 1.)
      init_fn, _ = (
          surrogate_posteriors.build_factored_surrogate_posterior_stateless(
              event_shape=model.event_shape))
      return init_fn(seed)
    init(seed)

  def test_multipart_bijector(self):
    dist = jdn.JointDistributionNamed({
        'a': exponential.Exponential(1.),
        'b': normal.Normal(0., 1.),
        'c': lambda b, a: sample.Sample(normal.Normal(b, a), sample_shape=[5])
    })

    surrogate_posterior = self._initialize_surrogate(
        'build_factored_surrogate_posterior',
        seed=test_util.test_seed(),
        event_shape=dist.event_shape,
        bijector=(
            dist.experimental_default_event_space_bijector()),
        validate_args=True)
    # Test that the posterior has the specified event shape(s).
    self.assertAllEqualNested(
        self.evaluate(dist.event_shape_tensor()),
        self.evaluate(surrogate_posterior.event_shape_tensor()))

    posterior_sample_ = self.evaluate(surrogate_posterior.sample(
        seed=test_util.test_seed()))
    posterior_logprob_ = self.evaluate(
        surrogate_posterior.log_prob(posterior_sample_))

    # Test that all sample Tensors have the expected shapes.
    check_shape = lambda s, x: self.assertAllEqual(s, x.shape)
    self.assertAllAssertsNested(
        check_shape, dist.event_shape, posterior_sample_)

    # Test that samples are finite and not NaN.
    self.assertAllAssertsNested(self.assertAllFinite, posterior_sample_)

    # Test that logprob is scalar, finite, and not NaN.
    self.assertEmpty(posterior_logprob_.shape)
    self.assertAllFinite(posterior_logprob_)


@test_util.test_all_tf_execution_regimes
class AffineSurrogatePosterior(test_util.TestCase, _SurrogatePosterior):

  @parameterized.named_parameters(
      {
          'testcase_name': 'TensorEvent',
          'dist_classes': normal.Normal,
          'event_shape': [4],
          'operators': [tf.linalg.LinearOperatorLowerTriangular],
          'initial_loc': np.array([[[0.9, 0.1, 0.5, 0.7]]]),
          'implicit_batch_shape': [1, 1],
          'bijector': sigmoid.Sigmoid(),
          'dtype': np.float32,
          'is_static': False
      },
      {
          'testcase_name': 'ListEvent',
          'dist_classes': [laplace.Laplace, normal.Normal, logistic.Logistic],
          'event_shape': [[3], [], [2, 2]],
          'operators': [
              tf.linalg.LinearOperatorDiag, tf.linalg.LinearOperatorDiag,
              tf.linalg.LinearOperatorLowerTriangular
          ],
          'initial_loc':
              [np.array([0.1, 7., 3.]), 0.1,
               np.array([[1., 0], [-4., 2.]])],
          'implicit_batch_shape': [],
          'bijector':
              [fill_triangular.FillTriangular(), None,
               softplus.Softplus()],
          'dtype': np.float64,
          'is_static': True
      },
      {
          'testcase_name': 'DictEventStateless',
          'dist_classes': {
              'x': logistic.Logistic,
              'y': normal.Normal
          },
          'event_shape': {
              'x': [2],
              'y': []
          },
          'operators': 'tril',
          'initial_loc': {
              'x': np.array([[0.9, 1.2]]),
              'y': np.array([-4.1])
          },
          'implicit_batch_shape': [1],
          'bijector': None,
          'dtype': np.float32,
          'is_static': False,
          'is_stateless': True
      },
  )
  def test_constrained_affine_from_distributions(
      self, dist_classes, event_shape, operators, initial_loc,
      implicit_batch_shape, bijector, dtype, is_static, is_stateless=JAX_MODE):
    if not tf.executing_eagerly() and not is_static:
      self.skipTest('tfb.Reshape requires statically known shapes in graph'
                    ' mode.')

    init_seed, grads_seed, shapes_seed, dtype_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'),
        n=4)

    # pylint: disable=g-long-lambda
    initial_loc = tf.nest.map_structure(
        lambda s: self.maybe_static(np.array(s, dtype=dtype),
                                    is_static=is_static),
        initial_loc)
    distributions = nest.map_structure_up_to(
        dist_classes, lambda d, loc, s: independent.Independent(
            d(loc=loc, scale=1.),
            reinterpreted_batch_ndims=ps.rank_from_shape(s)), dist_classes,
        initial_loc, event_shape)
    # pylint: enable=g-long-lambda

    surrogate_posterior = self._initialize_surrogate(
        'build_affine_surrogate_posterior_from_base_distribution',
        is_stateless=is_stateless,
        seed=init_seed,
        base_distribution=distributions,
        operators=operators,
        bijector=bijector,
        validate_args=True)

    event_shape = nest.map_structure(
        lambda d: d.event_shape_tensor(), distributions)
    if bijector is not None:
      event_shape = nest.map_structure(
          lambda b, s: s if b is None else b.forward_event_shape_tensor(s),
          bijector, event_shape)

    self._test_shapes(
        surrogate_posterior, batch_shape=implicit_batch_shape,
        event_shape=event_shape, seed=shapes_seed)
    self._test_dtype(surrogate_posterior, dtype, dtype_seed)
    if not is_stateless:
      self._test_gradients(surrogate_posterior, seed=grads_seed)

  @parameterized.named_parameters(
      {
          'testcase_name': 'TensorEvent',
          'event_shape': tf.TensorShape([3]),
          'operators': [tf.linalg.LinearOperatorDiag],
          'batch_shape': (),
          'bijector': exp.Exp(),
          'dtype': np.float32,
          'is_static': True
      },
      {
          'testcase_name': 'ListEvent',
          'event_shape':
              [tf.TensorShape([3]),
               tf.TensorShape([]),
               tf.TensorShape([2, 2])],
          'operators': 'diag',
          'batch_shape': (),
          'bijector':
              [softplus.Softplus(), None,
               fill_triangular.FillTriangular()],
          'dtype': np.float32,
          'is_static': False
      },
      {
          'testcase_name': 'DictEvent',
          'event_shape': {
              'x': tf.TensorShape([3]),
              'y': tf.TensorShape([])
          },
          'operators': [[tf.linalg.LinearOperatorDiag],
                        [
                            tf.linalg.LinearOperatorFullMatrix,
                            tf.linalg.LinearOperatorLowerTriangular
                        ]],
          'batch_shape': (),
          'bijector': None,
          'dtype': np.float64,
          'is_static': True
      },
      {
          'testcase_name': 'BatchShapeStateless',
          'event_shape': [tf.TensorShape([3]),
                          tf.TensorShape([])],
          'operators': 'tril',
          'batch_shape': (2, 1),
          'bijector': [
              softplus.Softplus(),
              None,
          ],
          'dtype': np.float32,
          'is_static': True,
          'is_stateless': True
      },
      {
          'testcase_name': 'DynamicBatchShape',
          'event_shape': [tf.TensorShape([3]),
                          tf.TensorShape([])],
          'operators': 'tril',
          'batch_shape': (2, 1),
          'bijector': [
              softplus.Softplus(),
              None,
          ],
          'dtype': np.float32,
          'is_static': False
      },
  )
  def test_constrained_affine_from_event_shape(
      self, event_shape, operators, bijector, batch_shape, dtype, is_static,
      is_stateless=JAX_MODE):

    init_seed, grads_seed, shapes_seed, dtype_seed = samplers.split_seed(
        test_util.test_seed(sampler_type='stateless'),
        n=4)

    if not is_static:
      event_shape = tf.nest.map_structure(
          lambda s: tf1.placeholder_with_default(  # pylint: disable=g-long-lambda
              np.array(s, dtype=np.int32),
              # Reshape bijector needs to know event_ndims statically.
              shape=[len(s)]),
          event_shape)

    surrogate_posterior = self._initialize_surrogate(
        'build_affine_surrogate_posterior',
        is_stateless=is_stateless,
        seed=init_seed,
        event_shape=event_shape,
        operators=operators,
        bijector=bijector,
        batch_shape=self.maybe_static(np.int32(batch_shape),
                                      is_static=is_static),
        dtype=dtype,
        validate_args=True)

    self._test_shapes(
        surrogate_posterior, batch_shape=batch_shape, event_shape=event_shape,
        seed=shapes_seed)
    self._test_dtype(surrogate_posterior, dtype, seed=dtype_seed)
    if not is_stateless:
      self._test_gradients(surrogate_posterior, seed=grads_seed)

  def test_constrained_affine_from_joint_inputs(self):
    base_distribution = jds.JointDistributionSequential([
        sample.Sample(normal.Normal(0., 1.), sample_shape=[3, 2]),
        logistic.Logistic(0., 1.),
        sample.Sample(normal.Normal(0., 1.), sample_shape=[4])
    ],
                                                        validate_args=True)
    operator = tf.linalg.LinearOperatorBlockDiag(
        operators=[
            tf.linalg.LinearOperatorScaledIdentity(
                6, multiplier=tf.Variable(1.)),
            tf.linalg.LinearOperatorDiag(tf.Variable([1.])),
            tf.linalg.LinearOperatorDiag(tf.Variable(tf.ones([4])))],
        is_non_singular=True)
    bijector = joint_map.JointMap(
        [exp.Exp(), identity.Identity(),
         softplus.Softplus()],
        validate_args=True)
    surrogate_posterior = self._initialize_surrogate(
        'build_affine_surrogate_posterior_from_base_distribution',
        seed=test_util.test_seed(),
        base_distribution=base_distribution,
        operators=operator,
        bijector=bijector,
        validate_args=True)
    posterior_sample = surrogate_posterior.sample(seed=test_util.test_seed())
    posterior_logprob = surrogate_posterior.log_prob(posterior_sample)
    sample_, log_prob_ = self.evaluate([posterior_sample, posterior_logprob])
    self.assertEmpty(ps.shape(log_prob_))
    self.assertAllFinite(log_prob_)
    self.assertAllEqualNested(
        self.evaluate(base_distribution.event_shape_tensor()),
        nest.map_structure(ps.shape, sample_))

  def test_mixed_dtypes_raises(self):
    base_distribution = {
        'a': normal.Normal(tf.zeros([], dtype=tf.float32), 1.),
        'b': logistic.Logistic(tf.zeros([], dtype=tf.float64), 1.)
    }
    operators = [tf.linalg.LinearOperatorDiag] * 2
    with self.assertRaisesRegexp(NotImplementedError, 'mixed dtype'):
      init_fn, apply_fn = (
          surrogate_posteriors
          .build_affine_surrogate_posterior_from_base_distribution_stateless(
              base_distribution, operators=operators, validate_args=True))
      apply_fn(init_fn(seed=test_util.test_seed(sampler_type='stateless')))

  def test_deterministic_initialization_from_seed(self):
    initial_samples = []
    seed = test_util.test_seed(sampler_type='stateless')
    init_seed, sample_seed = samplers.split_seed(seed)
    for _ in range(2):
      surrogate_posterior = self._initialize_surrogate(
          'build_affine_surrogate_posterior',
          event_shape={'x': tf.TensorShape([3]), 'y': tf.TensorShape([])},
          operators='tril',
          bijector=None,
          dtype=tf.float32,
          seed=init_seed,
          validate_args=True)
      initial_samples.append(
          surrogate_posterior.sample([5], seed=sample_seed))
    self.assertAllEqualNested(initial_samples[0], initial_samples[1])

  @test_util.jax_disable_variable_test
  def test_that_gamma_fitting_example_runs(self):
    model = self._make_gamma_model()
    surrogate_posterior = (
        surrogate_posteriors.build_affine_surrogate_posterior(
            model.event_shape_tensor()[:-1],
            operators=[[tf.linalg.LinearOperatorLowerTriangular],
                       [None, tf.linalg.LinearOperatorDiag]],
            bijector=[softplus.Softplus(),
                      softplus.Softplus()],
            base_distribution=logistic.Logistic,
            validate_args=True))
    self._test_fitting(model, surrogate_posterior)


@test_util.test_all_tf_execution_regimes
class SplitFlowSurrogatePosterior(
    test_util.TestCase, _SurrogatePosterior):

  @parameterized.named_parameters(
      {
          'testcase_name': 'TensorEvent',
          'event_shape': [6],
          'constraining_bijector': softplus.Softplus(),
          'batch_shape': [2],
          'dtype': np.float32,
          'is_static': True
      },
      {
          'testcase_name': 'ListEvent',
          'event_shape':
              [tf.TensorShape([3]),
               tf.TensorShape([]),
               tf.TensorShape([2, 2])],
          'constraining_bijector':
              [softplus.Softplus(), None,
               fill_triangular.FillTriangular()],
          'batch_shape': tf.TensorShape([2, 2]),
          'dtype': np.float32,
          'is_static': False
      },
      {
          'testcase_name': 'DictEvent',
          'event_shape': {
              'x': tf.TensorShape([3]),
              'y': tf.TensorShape([])
          },
          'constraining_bijector': None,
          'batch_shape': tf.TensorShape([]),
          'dtype': np.float64,
          'is_static': True
      },
  )
  @test_util.jax_disable_variable_test
  def test_shapes_and_gradients(
      self, event_shape, constraining_bijector, batch_shape, dtype, is_static):
    if not tf.executing_eagerly() and not is_static:
      self.skipTest('tfb.Reshape requires statically known shapes in graph'
                    ' mode.')
    net = maf_lib.AutoregressiveNetwork(2, hidden_units=[4, 4], dtype=dtype)
    maf = maf_lib.MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=net, validate_args=True)

    seed = test_util.test_seed_stream()
    surrogate_posterior = (
        surrogate_posteriors.build_split_flow_surrogate_posterior(
            event_shape=tf.nest.map_structure(
                lambda s: self.maybe_static(  # pylint: disable=g-long-lambda
                    np.array(s, dtype=np.int32),
                    is_static=is_static),
                event_shape),
            constraining_bijector=constraining_bijector,
            batch_shape=batch_shape,
            trainable_bijector=maf,
            dtype=dtype,
            validate_args=True))

    # Add an op to the graph so tf.Variables get created in graph mode.
    _ = surrogate_posterior.sample(seed=seed())
    self.evaluate(
        [v.initializer for v in surrogate_posterior.trainable_variables])

    self._test_shapes(
        surrogate_posterior, batch_shape=batch_shape, event_shape=event_shape,
        seed=seed())
    self._test_gradients(surrogate_posterior, seed=seed())
    self._test_dtype(surrogate_posterior, dtype, seed())

if __name__ == '__main__':
  test_util.main()
