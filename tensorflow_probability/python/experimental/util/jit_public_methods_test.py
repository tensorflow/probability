# Copyright 2021 The TensorFlow Probability Authors.
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
"""Tests for compiled distributions."""

import types

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.bijectors import invert
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.distributions import uniform
from tensorflow_probability.python.experimental.util import jit_public_methods
from tensorflow_probability.python.internal import test_util

from tensorflow.python.ops import control_flow_util  # pylint: disable=g-direct-tensorflow-import


JAX_MODE = False


class LogProbCanary(object):
  """Dummy class to verify that functions are only called once."""

  def __init__(self):
    self.log_prob_calls = 0

  def log_prob(self, x):
    self.log_prob_calls += 1
    return normal.Normal(0., 1.).log_prob(x)


class XLADetector(object):

  def in_xla_context(self):
    return control_flow_util.GraphOrParentsInXlaContext(tf1.get_default_graph())


@test_util.test_graph_and_eager_modes
class TracedPublicMethodsTest(test_util.TestCase):

  def test_distribution_methods(self):
    batch_shape = [4]
    sample_shape = [2, 3]
    sample_and_batch_shape = [2, 3, 4]
    d = jit_public_methods.JitPublicMethods(
        normal.Normal(loc=0., scale=tf.ones(batch_shape, dtype=tf.float32)))
    self.assertEqual(d.dtype, tf.float32)
    self.assertAllEqual(d.event_shape, [])
    self.assertAllEqual(d.event_shape_tensor(), [])
    self.assertAllEqual(d.batch_shape, batch_shape)
    self.assertAllEqual(d.batch_shape_tensor(), batch_shape)

    if JAX_MODE:
      # TODO(b/184565492): Enable passing shapes to jitted methods in JAX.
      x = tf.zeros(sample_and_batch_shape, dtype=tf.float32)
    else:
      x = d.sample(sample_shape,
                   seed=test_util.test_seed(sampler_type='stateless'))
    self.assertAllEqual(x.shape, sample_and_batch_shape)
    self.assertAllEqual(d.log_prob(x).shape, sample_and_batch_shape)

    self.assertAllEqual(d.mean().shape, batch_shape)
    self.assertAllEqual(d.stddev().shape, batch_shape)
    self.assertAllEqual(d.entropy().shape, batch_shape)

    d_copy = d.copy(loc=[1., 2., 3.], scale=1.)
    self.assertIsInstance(d_copy, jit_public_methods.JitPublicMethods)
    self.assertEqual(d_copy.batch_shape, [3])

    d_slice = d_copy[:2]
    self.assertIsInstance(d_slice, jit_public_methods.JitPublicMethods)
    self.assertEqual(d_slice.batch_shape, [2])

  def test_methods_are_wrapped(self):
    d = jit_public_methods.JitPublicMethods(
        normal.Normal(0., 1.),
        methods_to_exclude=('event_shape_tensor', 'batch_shape_tensor'))
    # Wrapped methods should be some sort of non-method type, e.g.:
    #   tensorflow.python.eager.def_function.Function (TF)
    #   types.FunctionType (JAX)
    self.assertNotIsInstance(d.sample, types.MethodType)
    self.assertNotIsInstance(d.log_prob, types.MethodType)

    # Excluded methods should not be wrapped.
    self.assertIsInstance(d.event_shape_tensor, types.MethodType)
    self.assertIsInstance(d.batch_shape_tensor, types.MethodType)

    # Private methods should not be wrapped.
    self.assertIsInstance(d._sample_n, types.MethodType)
    self.assertIsInstance(d._log_prob, types.MethodType)

  def test_works_with_transformed_distributions(self):
    dist = jit_public_methods.JitPublicMethods(uniform.Uniform(0., 1.))
    td = transformed_distribution.TransformedDistribution(
        distribution=dist,
        bijector=invert.Invert(
            dist.experimental_default_event_space_bijector()))
    if JAX_MODE:
      # TODO(b/184565492): Enable passing shapes to jitted methods in JAX.
      x = tf.zeros([], dtype=td.dtype)
    else:
      x = td.sample(seed=test_util.test_seed())
    td.log_prob(x)

  @test_util.jax_disable_test_missing_functionality('b/184565492')
  def test_kl_divergence(self):
    a, b = normal.Normal(1., 2.), normal.Normal(3., 1.)
    ta, tb = jit_public_methods.JitPublicMethods(
        a), jit_public_methods.JitPublicMethods(b)
    kl_a_b = a.kl_divergence(b)
    self.assertAllClose(kl_a_b, a.kl_divergence(tb))
    self.assertAllClose(kl_a_b, ta.kl_divergence(b))
    self.assertAllClose(kl_a_b, ta.kl_divergence(tb))

  def test_functions_are_cached(self):
    d = jit_public_methods.JitPublicMethods(LogProbCanary())
    d.log_prob(tf.constant(1.))
    d.log_prob(tf.constant(2.))
    self.assertEqual(d.log_prob_calls, 1)
    self.assertIs(d.log_prob, d.log_prob)

  @test_util.jax_disable_test_missing_functionality('trace_only')
  def test_trace_only_bypasses_xla(self):
    self.skip_if_no_xla()
    d = jit_public_methods.JitPublicMethods(XLADetector())
    d_trace_only = jit_public_methods.JitPublicMethods(
        XLADetector(), trace_only=True)
    self.assertTrue(self.evaluate(d.in_xla_context()))
    self.assertFalse(self.evaluate(d_trace_only.in_xla_context()))


if __name__ == '__main__':
  test_util.main()
