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
"""Integration test TFP+JAX."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

import jax
import jax.numpy as np

import tensorflow_probability as tfp
tfp = tfp.experimental.substrates.jax

tfb = tfp.bijectors
tfd = tfp.distributions


class JaxIntegrationTest(absltest.TestCase):

  def testBijector(self):

    def f(x):
      return tfb.GumbelCDF(loc=np.arange(3.).astype(np.float32)).forward(x)

    vecf = jax.vmap(f, in_axes=(0,))

    f(np.array(0., np.float32))
    f(np.array([1, 2, 3.], np.float32))
    vecf(np.linspace(-20, 20, 10).astype(np.float32))

    jax.jacfwd(f)(np.array([1, 2, 3.], np.float32))
    jax.jacfwd(vecf)(np.linspace(-20, 20, 10).astype(np.float32))

    f = jax.jit(f)
    vecf = jax.jit(vecf)
    f(np.array(0., np.float32))
    f(np.array([1, 2, 3.], np.float32))
    vecf(np.linspace(-20, 20, 10).astype(np.float32))

  def testDistribution(self):

    def f(s):
      return tfd.Normal(loc=np.arange(3.).astype(np.float32), scale=s).sample(
          seed=jax.random.PRNGKey(1))

    vecf = jax.vmap(f, in_axes=(0,))
    f(np.array(1., np.float32))
    f(np.array([1, 2, 3.], np.float32))
    vecf(np.linspace(1, 3, 10).astype(np.float32))
    vecf(np.linspace(1, 3, 30).reshape(10, 3).astype(np.float32))

    jax.jacfwd(f)(np.array(1., np.float32))
    jax.jacfwd(f)(np.array([1, 2, 3.], np.float32))
    jax.jacfwd(vecf)(np.linspace(1, 3, 10).astype(np.float32))


if __name__ == '__main__':
  absltest.main()
