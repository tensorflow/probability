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
"""Tests TFP distribution compositionality with JAX transformations."""
import functools

from absl import flags
from absl.testing import parameterized
import hypothesis as hp
from hypothesis import strategies as hps
import jax
from jax import random
import jax.numpy as np

# pylint: disable=no-name-in-module

from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal.backend import jax as tf
from tensorflow_probability.substrates.jax.distributions import hypothesis_testlib as dhps
from tensorflow_probability.substrates.jax.internal import hypothesis_testlib as tfp_hps
from tensorflow_probability.substrates.jax.internal import test_util


flags.DEFINE_bool('execute_only', False,
                  'If specified, skip equality checks and only verify '
                  'execution of transforms works.')
# Allows us to easily evaluate which blocklisted items now pass.
flags.DEFINE_bool('blocklists_only', False,
                  'If specified, run tests only for blocklisted distributions.')
FLAGS = flags.FLAGS


JIT_SAMPLE_BLOCKLIST = ()
JIT_LOGPROB_BLOCKLIST = (
    'BatchReshape',  # http://b/161984806
    'Bates',
)

VMAP_SAMPLE_BLOCKLIST = ()
VMAP_LOGPROB_BLOCKLIST = (
    'BatchReshape',  # http://b/161984806
    'Bates',
    'NegativeBinomial',  # Times out.
)

JVP_SAMPLE_BLOCKLIST = ()
JVP_LOGPROB_SAMPLE_BLOCKLIST = ()
JVP_LOGPROB_PARAM_BLOCKLIST = ()

VJP_SAMPLE_BLOCKLIST = ()
VJP_LOGPROB_SAMPLE_BLOCKLIST = ()
VJP_LOGPROB_PARAM_BLOCKLIST = ()


DEFAULT_MAX_EXAMPLES = 3


test_all_distributions = parameterized.named_parameters(
    {'testcase_name': dname, 'dist_name': dname} for dname in
    sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())
           + list(d for d in dhps.INSTANTIABLE_META_DISTS if d != 'Mixture')))

test_base_distributions = parameterized.named_parameters(
    {'testcase_name': dname, 'dist_name': dname} for dname in
    sorted(list(dhps.INSTANTIABLE_BASE_DISTS.keys())))


class JitTest(test_util.TestCase):

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testSample(self, dist_name, data):
    if (dist_name in JIT_SAMPLE_BLOCKLIST) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        dist_name=dist_name))
    def _sample(seed):
      return dist.sample(seed=seed)
    seed = test_util.test_seed()
    result = jax.jit(_sample)(seed)
    if not FLAGS.execute_only:
      self.assertAllClose(_sample(seed), result, rtol=1e-6,
                          atol=1e-6)

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testLogProb(self, dist_name, data):
    if (dist_name in JIT_LOGPROB_BLOCKLIST) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(
        enable_vars=False,
        dist_name=dist_name,
        eligibility_filter=lambda dname: dname not in JIT_LOGPROB_BLOCKLIST))
    sample = dist.sample(seed=test_util.test_seed())
    result = jax.jit(dist.log_prob)(sample)
    if not FLAGS.execute_only:
      self.assertAllClose(dist.log_prob(sample), result,
                          rtol=1e-6, atol=1e-6)


class VmapTest(test_util.TestCase):

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testSample(self, dist_name, data):
    if (dist_name in VMAP_SAMPLE_BLOCKLIST) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')
    dist = data.draw(dhps.distributions(enable_vars=False,
                                        dist_name=dist_name))
    def _sample(seed):
      return dist.sample(seed=seed)
    seed = test_util.test_seed()
    jax.vmap(_sample)(random.split(seed, 10))

  @test_all_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testLogProb(self, dist_name, data):
    if (dist_name in VMAP_LOGPROB_BLOCKLIST) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')
    if dist_name == 'NegativeBinomial':
      self.skipTest('Skip never-terminating negative binomial vmap logprob.')
    dist = data.draw(dhps.distributions(
        enable_vars=False,
        dist_name=dist_name,
        eligibility_filter=lambda dname: dname not in VMAP_LOGPROB_BLOCKLIST))
    sample = dist.sample(seed=test_util.test_seed(), sample_shape=10)
    result = jax.vmap(dist.log_prob)(sample)
    if not FLAGS.execute_only:
      self.assertAllClose(result, dist.log_prob(sample),
                          rtol=1e-6, atol=1e-6)


class _GradTest(test_util.TestCase):

  def _make_distribution(self, dist_name, params,
                         batch_shape, override_params=None):
    override_params = override_params or {}
    all_params = dict(params)
    for param_name, override_param in override_params.items():
      all_params[param_name] = override_param
    all_params = dhps.constrain_params(all_params, dist_name)
    all_params = dhps.modify_params(all_params, dist_name, validate_args=False)
    return dhps.base_distributions(
        enable_vars=False, dist_name=dist_name, params=all_params,
        batch_shape=batch_shape, validate_args=False)

  def _param_func_generator(self, data, dist_name, params, batch_shape, func,
                            generate_sample_function=False):
    for param_name, param in params.items():
      if (not tf.is_tensor(param)
          or not np.issubdtype(param.dtype, np.floating)):
        continue
      def _dist_func(param_name, param):
        return data.draw(self._make_distribution(
            dist_name, params, batch_shape,
            override_params={param_name: param}))
      def _func(param_name, param):
        return func(_dist_func(param_name, param))
      yield param_name, param, _dist_func, _func

  @test_base_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testSample(self, dist_name, data):
    if (dist_name in self.sample_blocklist) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')

    def _sample(dist):
      return dist.sample(seed=random.PRNGKey(0))

    params_unconstrained, batch_shape = data.draw(
        dhps.base_distribution_unconstrained_params(
            enable_vars=False, dist_name=dist_name))

    for (param_name, unconstrained_param, dist_func,
         func) in self._param_func_generator(data, dist_name,
                                             params_unconstrained, batch_shape,
                                             _sample):
      dist = dist_func(param_name, unconstrained_param)
      if (dist.reparameterization_type !=
          reparameterization.FULLY_REPARAMETERIZED):
        # Skip distributions that don't support differentiable sampling.
        self.skipTest('{} is not reparameterized.'.format(dist_name))
      self._test_transformation(
          functools.partial(func, param_name), unconstrained_param,
          msg=param_name)

  @test_base_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testLogProbParam(self, dist_name, data):
    if (dist_name in self.logprob_param_blocklist) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')

    params, batch_shape = data.draw(
        dhps.base_distribution_unconstrained_params(
            enable_vars=False, dist_name=dist_name))
    constrained_params = dhps.constrain_params(params, dist_name)

    sampling_dist = data.draw(dhps.base_distributions(
        batch_shape=batch_shape, enable_vars=False, dist_name=dist_name,
        params=constrained_params))
    sample = sampling_dist.sample(seed=random.PRNGKey(0))
    def _log_prob(dist):
      return dist.log_prob(sample)
    for param_name, param, dist_func, func in self._param_func_generator(
        data, dist_name, params, batch_shape, _log_prob):
      del dist_func
      self._test_transformation(
          functools.partial(func, param_name), param, msg=param_name)

  @test_base_distributions
  @hp.given(hps.data())
  @tfp_hps.tfp_hp_settings(default_max_examples=DEFAULT_MAX_EXAMPLES)
  def testLogProbSample(self, dist_name, data):
    if (dist_name in self.logprob_sample_blocklist) != FLAGS.blocklists_only:
      self.skipTest('Distribution currently broken.')

    params, batch_shape = data.draw(
        dhps.base_distribution_unconstrained_params(
            enable_vars=False, dist_name=dist_name))
    constrained_params = dhps.constrain_params(params, dist_name)

    dist = data.draw(dhps.base_distributions(
        batch_shape=batch_shape, enable_vars=False, dist_name=dist_name,
        params=constrained_params))

    sample = dist.sample(seed=random.PRNGKey(0))
    if np.issubdtype(sample.dtype, np.integer):
      self.skipTest('{} has integer samples; no derivative.'.format(dist_name))
    def _log_prob(sample):
      return dist.log_prob(sample)
    self._test_transformation(_log_prob, sample)


class JVPTest(_GradTest):

  sample_blocklist = JVP_SAMPLE_BLOCKLIST
  logprob_param_blocklist = JVP_LOGPROB_PARAM_BLOCKLIST
  logprob_sample_blocklist = JVP_LOGPROB_SAMPLE_BLOCKLIST

  def _test_transformation(self, func, param, msg=None):
    primal, tangent = jax.jvp(func, (param,), (np.ones_like(param),))
    self.assertEqual(primal.shape, tangent.shape)
    if not FLAGS.execute_only:
      self.assertNotAllEqual(tangent, np.zeros_like(tangent), msg=msg)


class VJPTest(_GradTest):

  sample_blocklist = VJP_SAMPLE_BLOCKLIST
  logprob_param_blocklist = VJP_LOGPROB_PARAM_BLOCKLIST
  logprob_sample_blocklist = VJP_LOGPROB_SAMPLE_BLOCKLIST

  def _test_transformation(self, func, param, msg=None):
    out, f_vjp = jax.vjp(func, param)
    cotangent, = f_vjp(np.ones_like(out).astype(out.dtype))
    self.assertEqual(param.shape, cotangent.shape)
    if not FLAGS.execute_only:
      self.assertNotAllEqual(cotangent, np.zeros_like(cotangent), msg=msg)


del _GradTest  # not intended for standalone execution


if __name__ == '__main__':
  tf.test.main()
