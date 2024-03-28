# Copyright 2023 The TensorFlow Probability Authors.
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
"""Tests for training_util.py."""

import chex
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.internal import test_util
from autobnn import kernels
from autobnn import operators
from autobnn import training_util
from autobnn import util


class TrainingUtilTest(test_util.TestCase):

  def test__filter_stuck_chains_doesnt_overfilter(self):
    noise_scale = 0.001 * np.random.randn(64, 100, 1)
    filtered = training_util._filter_stuck_chains(
        {'params': {'noise_scale': noise_scale}}
    )
    self.assertEqual(filtered['params']['noise_scale'].shape, (64, 100, 1))

  def test__filter_stuck_chains(self):
    noise_scale = np.concatenate(
        (0.1 * np.random.randn(2, 100, 1), np.random.randn(62, 100, 1))
    )
    filtered = training_util._filter_stuck_chains(
        {'params': {'noise_scale': noise_scale}}
    )
    self.assertEqual(filtered['params']['noise_scale'].shape, (62, 100, 1))

  def test_train(self):
    seed = jax.random.PRNGKey(20231018)
    x_train, y_train = util.load_fake_dataset()

    leaf1 = kernels.PeriodicBNN(
        width=5, period=0.1, going_to_be_multiplied=True
    )
    leaf2 = kernels.LinearBNN(width=5, going_to_be_multiplied=True)
    bnn = operators.Multiply(bnns=[leaf1, leaf2])

    _, diagnostics = training_util.fit_bnn_map(
        bnn, seed, x_train, y_train, num_particles=8, num_iters=100
    )
    self.assertEqual(diagnostics['loss'].shape, (8, 100))

  def test_plot(self):
    seed = jax.random.PRNGKey(20231018)
    x_train, y_train = util.load_fake_dataset()

    leaf1 = kernels.PeriodicBNN(
        width=5, period=0.1, going_to_be_multiplied=True
    )
    leaf2 = kernels.LinearBNN(width=5, going_to_be_multiplied=True)
    bnn = operators.Multiply(bnns=[leaf1, leaf2])

    params, diagnostics = training_util.fit_bnn_map(
        bnn, seed, x_train, y_train, width=5, num_particles=8, num_iters=100
    )
    preds = training_util.make_predictions(params, bnn, x_train)
    fig = training_util.plot_results(
        x_train.squeeze(),
        preds,
        dates_train=x_train.squeeze(),
        y_train=y_train,
        dates_test=x_train.squeeze(),
        y_test=y_train,
        diagnostics=diagnostics,
    )
    self.assertLen(fig.axes, 2)

  def test_get_params_batch_length(self):
    self.assertEqual(
        10, training_util.get_params_batch_length({'noise_scale': jnp.ones(10)})
    )
    self.assertEqual(
        5,
        training_util.get_params_batch_length(
            {'noise_scale': jnp.ones((5, 7))}
        ),
    )
    self.assertEqual(
        20,
        training_util.get_params_batch_length(
            {'params': {'noise_scale': jnp.ones((20, 8))}}
        ),
    )

  def test_debatchify_params1(self):
    out = training_util.debatchify_params({'noise_scale': jnp.ones(10)})
    self.assertLen(out, 10)
    self.assertEqual(out[0], {'noise_scale': jnp.ones(1)})
    self.assertEqual(out[1], {'noise_scale': jnp.ones(1)})

  def test_debatchify_params2(self):
    out = training_util.debatchify_params(
        {'params': {'noise_scale': jnp.ones(10)}}
    )
    self.assertLen(out, 10)
    self.assertEqual(out[0], {'params': {'noise_scale': jnp.ones(1)}})
    self.assertEqual(out[1], {'params': {'noise_scale': jnp.ones(1)}})

  def test_debatchify_params3(self):
    out = training_util.debatchify_params({
        'params': {
            'noise_scale': jnp.ones(10),
            'amplitude': jnp.ones(10),
            'length_scale': jnp.zeros(10),
        }
    })
    self.assertLen(out, 10)
    self.assertEqual(
        out[0],
        {
            'params': {
                'noise_scale': jnp.ones(1),
                'amplitude': jnp.ones(1),
                'length_scale': jnp.zeros(1),
            }
        },
    )
    self.assertEqual(
        out[1],
        {
            'params': {
                'noise_scale': jnp.ones(1),
                'amplitude': jnp.ones(1),
                'length_scale': jnp.zeros(1),
            }
        },
    )

  def test_debatchify_params4(self):
    out = training_util.debatchify_params({
        'params': {
            'noise_scale': jnp.ones(20),
            'dense1': {
                'kernel': jnp.ones((20, 10)),
                'bias': jnp.ones((20, 10)),
            },
        }
    })
    self.assertLen(out, 20)
    print(f'{out[0]=}')
    chex.assert_trees_all_close(
        out[0],
        {
            'params': {
                'noise_scale': jnp.ones(1),
                'dense1': {'kernel': jnp.ones(10), 'bias': jnp.ones(10)},
            }
        },
    )
    chex.assert_trees_all_close(
        out[1],
        {
            'params': {
                'noise_scale': jnp.ones(1),
                'dense1': {'kernel': jnp.ones(10), 'bias': jnp.ones(10)},
            }
        },
    )

  def test_debatchify_real(self):
    k = kernels.OneLayerBNN(width=50)
    num_particles = 10
    init_seed = jax.random.PRNGKey(0)

    def _init(seed):
      return k.init(seed, jnp.ones(5))

    params = jax.vmap(_init)(jax.random.split(init_seed, num_particles))
    self.assertEqual(10, training_util.get_params_batch_length(params))
    debatched_params = training_util.debatchify_params(params)
    self.assertLen(debatched_params, 10)
    lp = k.log_prior(debatched_params[0])
    lp2 = k.log_prior(params)
    self.assertLess(lp2, lp)

  def test_debatchify_real_weighted_sum(self):
    k = operators.WeightedSum(
        bnns=(kernels.OneLayerBNN(width=50), kernels.OneLayerBNN(width=50))
    )
    num_particles = 10
    init_seed = jax.random.PRNGKey(0)

    def _init(seed):
      return k.init(seed, jnp.ones(5))

    params = jax.vmap(_init)(jax.random.split(init_seed, num_particles))
    debatched_params = training_util.debatchify_params(params)
    self.assertLen(debatched_params, 10)
    lp = k.log_prior(debatched_params[0])
    lp2 = k.log_prior(params)
    self.assertLess(lp2, lp)


if __name__ == '__main__':
  test_util.main()
