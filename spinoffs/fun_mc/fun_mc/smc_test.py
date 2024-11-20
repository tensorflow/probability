# Copyright 2024 The TensorFlow Probability Authors.
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
# Dependency imports

from absl.testing import parameterized
import jax as real_jax
import tensorflow.compat.v2 as real_tf
from tensorflow_probability.python.internal import test_util as tfp_test_util
from fun_mc import backend
from fun_mc import fun_mc_lib as fun_mc
from fun_mc import smc
from fun_mc import test_util
from fun_mc import types

jax = backend.jax
jnp = backend.jnp
tfp = backend.tfp
util = backend.util
tfd = tfp.distributions
distribute_lib = backend.distribute_lib
Root = tfd.JointDistributionCoroutine.Root
Array = types.Array
Seed = types.Seed
Float = types.Float
Int = types.Int
Bool = types.Bool
BoolScalar = types.BoolScalar
IntScalar = types.IntScalar
FloatScalar = types.FloatScalar


real_tf.enable_v2_behavior()
real_tf.experimental.numpy.experimental_enable_numpy_behavior()
real_jax.config.update('jax_enable_x64', True)

BACKEND = None  # Rewritten by backends/rewrite.py.


def _test_seed() -> Seed:
  seed = tfp_test_util.test_seed() % (2**32 - 1)
  if BACKEND == 'backend_jax':
    return jax.random.PRNGKey(seed)
  else:
    return util.make_tensor_seed([seed, 0])


class SMCTest(tfp_test_util.TestCase):

  @property
  def _dtype(self):
    raise NotImplementedError()

  def _constant(self, value):
    return jnp.array(value, self._dtype)

  @parameterized.parameters(True, False)
  def test_systematic_resampling(self, permute):
    seed = _test_seed()

    num_replications = 10000
    weights = self._constant([0.0, 0.5, 0.2, 0.3, 0.0])
    log_weights = jnp.log(weights)

    def kernel(seed):
      seed, sample_seed = util.split_seed(seed, 2)
      parents = smc.systematic_resampling(
          log_weights, seed=sample_seed, permute=permute
      )
      return seed, parents

    _, parents = jax.jit(
        lambda seed: fun_mc.trace(seed, kernel, num_replications)
    )(seed)

    # [num_samples, parents, parents]
    freqs = jnp.mean(
        jnp.array(
            parents[..., jnp.newaxis] == jnp.arange(len(weights)), jnp.float32
        ),
        (0, 1),
    )

    self.assertAllClose(freqs, weights, atol=0.05)

    if permute:
      mean_index = jnp.sum(weights * jnp.arange(len(weights)))
      self.assertAllClose(
          jnp.mean(parents, 0), [mean_index] * len(weights), atol=0.05
      )

  def test_conditional_systematic_resampling(self):
    seed = _test_seed()

    num_replications = 10000
    weights = self._constant([0.2, 0.5, 0.2, 0.1, 0.0])
    log_weights = jnp.log(weights)

    def kernel(seed):
      seed, systematic_seed, cond_seed = util.split_seed(seed, 3)
      systematic_parents = smc.systematic_resampling(
          log_weights, seed=systematic_seed, permute=True
      )
      conditional_parents = smc.systematic_resampling(
          log_weights,
          seed=cond_seed,
      )
      return seed, (systematic_parents, conditional_parents)

    _, (systematic_parents, conditional_parents) = jax.jit(
        lambda seed: fun_mc.trace(seed, kernel, num_replications)
    )(seed)

    self.assertFalse(jnp.all(systematic_parents[:, 0] == 0))
    self.assertTrue(jnp.all(conditional_parents[:, 0] == 0))

    accepted_samples = jnp.array(systematic_parents[:, 0] == 0, jnp.float32)
    rejection_freqs = jnp.sum(
        jnp.mean(
            accepted_samples[:, jnp.newaxis, jnp.newaxis]
            * jnp.array(
                systematic_parents[..., jnp.newaxis]
                == jnp.arange(len(weights)),
                jnp.float32,
            ),
            1,
        ),
        0,
    ) / jnp.sum(accepted_samples)
    conditional_freqs = jnp.mean(
        jnp.array(
            conditional_parents[..., jnp.newaxis] == jnp.arange(len(weights)),
            jnp.float32,
        ),
        (0, 1),
    )
    self.assertAllClose(rejection_freqs, conditional_freqs, atol=0.05)


@test_util.multi_backend_test(globals(), 'smc_test')
class SMCTest32(SMCTest):

  @property
  def _dtype(self):
    return jnp.float32


@test_util.multi_backend_test(globals(), 'smc_test')
class SMCTest64(SMCTest):

  @property
  def _dtype(self):
    return jnp.float64


del SMCTest

if __name__ == '__main__':
  tfp_test_util.main()
