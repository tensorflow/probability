# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Tests for trainable distributions and bijectors."""

import functools
from absl.testing import parameterized

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import identity
from tensorflow_probability.python.bijectors import joint_map
from tensorflow_probability.python.bijectors import softplus
from tensorflow_probability.python.distributions import deterministic
from tensorflow_probability.python.distributions import joint_distribution_auto_batched as jdab
from tensorflow_probability.python.distributions import joint_distribution_sequential as jds
from tensorflow_probability.python.distributions import lognormal
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import tf_keras
from tensorflow_probability.python.internal import trainable_state_util
from tensorflow_probability.python.math import gradient
from tensorflow_probability.python.math.minimize import minimize
from tensorflow_probability.python.math.minimize import minimize_stateless

JAX_MODE = False


def seed_generator():
  # Seed must be passed as kwarg.
  a = yield trainable_state_util.Parameter(
      functools.partial(samplers.normal, shape=[5]))
  # Seed must be passed positionally.
  b = yield trainable_state_util.Parameter(
      lambda my_seed: samplers.normal([], seed=my_seed))
  # Seed not accepted.
  c = yield trainable_state_util.Parameter(lambda: tf.zeros([3]))
  # Bare value in place of callable.
  d = yield trainable_state_util.Parameter(tf.ones([1, 1]))
  # Distribution sample method.
  e = yield trainable_state_util.Parameter(
      lognormal.LogNormal([-1., 1.], 1.).sample)
  return jds.JointDistributionSequential([
      deterministic.Deterministic(a),
      deterministic.Deterministic(b),
      deterministic.Deterministic(c),
      deterministic.Deterministic(d),
      deterministic.Deterministic(e)
  ])


def normal_generator(shape):
  shape = ps.convert_to_shape_tensor(shape, dtype=np.int32)
  loc = yield trainable_state_util.Parameter(
      init_fn=functools.partial(samplers.normal, shape=shape),
      name='loc')
  bij = softplus.Softplus()
  scale = yield trainable_state_util.Parameter(
      init_fn=lambda seed: bij.forward(samplers.normal(shape, seed=seed)),
      constraining_bijector=bij,
      name='scale')
  return normal.Normal(loc=loc, scale=scale, validate_args=True)


def joint_normal_nested_generator(shapes):
  dists = []
  for shape in shapes:
    dist = yield from normal_generator(shape)
    dists.append(dist)
  return jdab.JointDistributionSequentialAutoBatched(dists)


def yields_structured_parameter():
  dict_loc_scale = yield trainable_state_util.Parameter(
      init_fn=lambda: {  # pylint:disable=g-long-lambda
          'scale': tf.ones([2]),
          'loc': tf.zeros([2])
      },
      name='dict_loc_scale',
      constraining_bijector=joint_map.JointMap({
          'scale': softplus.Softplus(),
          'loc': identity.Identity()
      }))
  return normal.Normal(**dict_loc_scale)


def yields_never():
  return
  yield  # pylint: disable=unreachable


def yields_none():
  yield


def yields_distribution():
  yield normal.Normal(0., 1.)


def yields_non_callable_init_fn():
  yield trainable_state_util.Parameter(0.)


def generator_with_docstring(a, b):
  """Test generator with a docstring.

  Args:
    a: an arg.
    b: another arg.
  Yields:
    some_stuff: THIS SHOULD BE REMOVED.
    some_more_stuff: ALSO THIS.

  """
  yield lambda seed: (a, b)


@test_util.test_graph_and_eager_modes
class TestWrapGeneratorAsStateless(test_util.TestCase):

  def test_init_supports_arg_or_kwarg_seed(self):

    seed = test_util.test_seed(sampler_type='stateless')
    init_fn, _ = trainable_state_util.as_stateless_builder(
        seed_generator)()
    self.assertLen(init_fn(seed=seed), 5)
    seed = test_util.clone_seed(seed)
    seed2 = test_util.clone_seed(seed)
    # Check that we can invoke init_fn with an arg or kwarg seed,
    # regardless of how the inner functions are parameterized.
    self.assertAllCloseNested(init_fn(seed), init_fn(seed=seed2))

    if not JAX_MODE:
      # Check that we can initialize with no seed.
      self.assertLen(init_fn(), 5)

  @parameterized.named_parameters(
      ('_normal', normal_generator, 2, np.array([5])),
      ('_joint_normal',
       joint_normal_nested_generator, 6, [np.array([5]),
                                          np.array([]),
                                          np.array([2, 1])]))
  def test_distribution_init_apply(self, generator, expected_num_params, shape):
    # Test passing arguments to the wrapper.
    init_fn, apply_fn = trainable_state_util.as_stateless_builder(
        generator)(shape)

    seed = test_util.test_seed(sampler_type='stateless')
    params = init_fn(seed)
    self.assertLen(params, expected_num_params)

    # Check that the distribution's samples have the expected shape.
    dist = apply_fn(params)
    seed = test_util.clone_seed(seed)
    x = dist.sample(seed=seed)
    self.assertAllEqualNested(shape, tf.nest.map_structure(ps.shape, x))

    # Check that gradients are defined.
    _, grad = gradient.value_and_gradient(
        lambda *params: apply_fn(*params).log_prob(x), params)
    self.assertLen(grad, expected_num_params)
    self.assertAllNotNone(grad)

  def test_basic_parameter_names(self):
    init_fn, _ = trainable_state_util.as_stateless_builder(
        normal_generator)([2])
    params = init_fn(test_util.test_seed(sampler_type='stateless'))
    param_keys = params._asdict().keys()  # Params is a namedtuple / structuple.
    self.assertLen(param_keys, 2)
    self.assertIn('loc', param_keys)
    self.assertIn('scale', param_keys)

  def test_assigns_unique_names(self):
    init_fn, _ = trainable_state_util.as_stateless_builder(
        joint_normal_nested_generator)([[1], [2], [3]])
    params = init_fn(test_util.test_seed(sampler_type='stateless'))
    param_keys = params._asdict().keys()  # Params is a namedtuple / structuple.
    self.assertLen(param_keys, 6)
    self.assertIn('loc', param_keys)
    self.assertIn('scale', param_keys)
    self.assertIn('loc_0001', param_keys)
    self.assertIn('scale_0001', param_keys)
    self.assertIn('loc_0002', param_keys)
    self.assertIn('scale_0002', param_keys)

  def test_assigns_default_names(self):
    init_fn, _ = trainable_state_util.as_stateless_builder(
        seed_generator)()
    params = init_fn(test_util.test_seed(sampler_type='stateless'))
    param_keys = params._asdict().keys()  # Params is a namedtuple / structuple.
    self.assertLen(param_keys, 5)
    self.assertIn('parameter', param_keys)
    for i in range(1, 5):
      self.assertIn('parameter_{:04d}'.format(i), param_keys)

  def test_respects_constraining_bijector(self):
    init_fn, apply_fn = trainable_state_util.as_stateless_builder(
        normal_generator)([50])
    params = init_fn(test_util.test_seed(sampler_type='stateless'))
    dist = apply_fn(params)
    self.assertAllGreater(dist.scale, 0)

  def test_structured_parameters(self):
    init_fn, apply_fn = trainable_state_util.as_stateless_builder(
        yields_structured_parameter)()
    params = init_fn(test_util.test_seed(sampler_type='stateless'))
    self.assertAllEqualNested(params.dict_loc_scale, {
        'scale': softplus.Softplus().inverse(tf.ones([2])),
        'loc': tf.zeros([2])
    })
    dist = apply_fn(params)
    self.assertAllEqual(dist.loc, tf.zeros([2]))
    self.assertAllEqual(dist.scale, tf.ones([2]))

  def test_trivial_generator(self):
    init_fn, apply_fn = trainable_state_util.as_stateless_builder(
        yields_never)()
    params = init_fn(seed=test_util.test_seed())
    self.assertEmpty(params)
    result = apply_fn(params)
    self.assertIsNone(result)

  def test_raises_when_generator_is_not_a_generator(self):
    init_fn, apply_fn = trainable_state_util.as_stateless_builder(
        lambda: normal.Normal(0., 1.))()
    error_msg = 'must contain at least one `yield` statement'
    with self.assertRaisesRegex(ValueError, error_msg):
      init_fn()
    with self.assertRaisesRegex(ValueError, error_msg):
      apply_fn([])

  @parameterized.named_parameters(
      ('_yields_none', yields_none),
      ('_yields_distribution', yields_distribution))
  def test_raises_when_non_callable_yielded(self, generator):
    init_fn, _ = trainable_state_util.as_stateless_builder(
        generator)()
    with self.assertRaisesRegex(ValueError, 'Expected generator to yield'):
      init_fn()

  def test_apply_raises_on_bad_parameters(self):
    init_fn, apply_fn = trainable_state_util.as_stateless_builder(
        normal_generator)(shape=[2])
    good_params = init_fn(seed=test_util.test_seed(sampler_type='stateless'))
    # Check that both calling styles are supported.
    self.assertIsInstance(apply_fn(good_params), normal.Normal)
    self.assertIsInstance(apply_fn(*good_params), normal.Normal)

    with self.assertRaisesRegex(ValueError, 'Insufficient parameters'):
      apply_fn()
    with self.assertRaisesRegex(ValueError, 'Insufficient parameters'):
      apply_fn(None)
    apply_fn(list(good_params) + [np.array(2.)])

  def test_rewrites_yield_to_return_in_docstring(self):
    wrapped = trainable_state_util.as_stateless_builder(
        generator_with_docstring)
    self.assertIn('Yields:', generator_with_docstring.__doc__)
    self.assertNotIn('Yields:', wrapped.__doc__)
    self.assertIn('Test generator with a docstring.', wrapped.__doc__)
    self.assertIn(
        trainable_state_util._STATELESS_RETURNS_DOCSTRING,
        wrapped.__doc__,
    )

  def test_fitting_example(self):
    if not JAX_MODE:
      self.skipTest('Requires JAX with optax.')
    import optax  # pylint: disable=g-import-not-at-top
    build_trainable_normal_stateless = (
        trainable_state_util.as_stateless_builder(
            normal_generator))
    init_fn, apply_fn = build_trainable_normal_stateless(shape=[])

    # Find the maximum likelihood distribution given observed data.
    x_observed = [3., -2., 1.7]
    mle_parameters, _ = minimize_stateless(
        loss_fn=lambda *params: -apply_fn(*params).log_prob(x_observed),
        init=init_fn(seed=test_util.test_seed(sampler_type='stateless')),
        optimizer=optax.adam(1.0),
        num_steps=400)
    mle_dist = apply_fn(mle_parameters)
    self.assertAllClose(mle_dist.mean(), np.mean(x_observed), atol=0.1)
    self.assertAllClose(mle_dist.stddev(), np.std(x_observed), atol=0.1)


class TestWrapGeneratorAsStateful(test_util.TestCase):

  @test_util.jax_disable_variable_test
  def test_initialization_is_deterministic_with_seed(self):
    seed = test_util.test_seed(sampler_type='stateless')
    make_trainable_jd = trainable_state_util.as_stateful_builder(
        seed_generator)

    trainable_jd1 = make_trainable_jd(seed=seed)
    variables1 = trainable_jd1.trainable_variables
    self.assertLen(variables1, 5)

    seed = test_util.clone_seed(seed)
    trainable_jd2 = make_trainable_jd(seed=seed)
    variables2 = trainable_jd2.trainable_variables
    self.evaluate([v.initializer for v in variables1 + variables2])
    vals1, vals2 = self.evaluate((variables1, variables2))
    self.assertAllCloseNested(vals1, vals2)

  @test_util.jax_disable_variable_test
  def test_structured_parameters(self):
    make_trainable_normal = trainable_state_util.as_stateful_builder(
        yields_structured_parameter)
    trainable_normal = make_trainable_normal()
    self.assertLen(trainable_normal.trainable_variables, 2)
    self.evaluate(tf1.global_variables_initializer())
    self.assertAllEqual(trainable_normal.loc, tf.zeros([2]))
    self.assertAllEqual(trainable_normal.scale, tf.ones([2]))

  @test_util.jax_disable_variable_test
  def test_rewrites_yield_to_return_in_docstring(self):
    wrapped = trainable_state_util.as_stateful_builder(
        generator_with_docstring)
    self.assertIn('Yields:', generator_with_docstring.__doc__)
    self.assertNotIn('Yields:', wrapped.__doc__)
    self.assertIn('Test generator with a docstring.', wrapped.__doc__)
    self.assertIn(
        trainable_state_util._STATEFUL_RETURNS_DOCSTRING,
        wrapped.__doc__,
    )

  @test_util.jax_disable_variable_test
  def test_fitting_example(self):
    build_trainable_normal = trainable_state_util.as_stateful_builder(
        normal_generator)
    trainable_dist = build_trainable_normal(
        shape=[],
        seed=test_util.test_seed(sampler_type='stateless'))
    optimizer = tf_keras.optimizers.Adam(1.0)
    # Find the maximum likelihood distribution given observed data.
    x_observed = [3., -2., 1.7]
    losses = minimize(
        loss_fn=lambda: -trainable_dist.log_prob(x_observed),
        optimizer=optimizer,
        num_steps=300)
    self.evaluate(tf1.global_variables_initializer())
    losses = self.evaluate(losses)
    self.assertAllClose(trainable_dist.mean(),
                        np.mean(x_observed), atol=0.1)
    self.assertAllClose(trainable_dist.stddev(),
                        np.std(x_observed), atol=0.1)

if __name__ == '__main__':
  test_util.main()
