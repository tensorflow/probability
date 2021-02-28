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
"""Tests for `KernelBuilder`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v2 as tf
from tensorflow_probability.python.experimental.mcmc import kernel_builder
from tensorflow_probability.python.experimental.mcmc import sample_discarding_kernel
from tensorflow_probability.python.experimental.mcmc import with_reductions
from tensorflow_probability.python.internal import test_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.mcmc import hmc
from tensorflow_probability.python.mcmc import nuts
from tensorflow_probability.python.mcmc import simple_step_size_adaptation
from tensorflow_probability.python.mcmc import transformed_kernel


@test_util.test_all_tf_execution_regimes
class TestKernelBuilder(test_util.TestCase):

  def test_stack(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = (
        builder
        .hmc(num_leapfrog_steps=3)
        .simple_adaptation(adaptation_rate=.03)
        .transform([])
        .set_num_steps_between_results(10))
    kernel = builder.build(10)
    self.assertIsInstance(kernel, with_reductions.WithReductions)
    kernel = kernel.inner_kernel
    self.assertIsInstance(
        kernel, sample_discarding_kernel.SampleDiscardingKernel)
    kernel = kernel.inner_kernel
    self.assertIsInstance(
        kernel, transformed_kernel.TransformedTransitionKernel)
    kernel = kernel.inner_kernel
    self.assertIsInstance(
        kernel, simple_step_size_adaptation.SimpleStepSizeAdaptation)
    kernel = kernel.inner_kernel
    self.assertIsInstance(kernel, hmc.HamiltonianMonteCarlo)

  def test_change_core(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = builder.hmc(num_leapfrog_steps=3)
    self.assertIsInstance(builder.build().inner_kernel,
                          hmc.HamiltonianMonteCarlo)
    builder = builder.nuts()
    self.assertIsInstance(builder.build().inner_kernel, nuts.NoUTurnSampler)

  def test_change_core_params(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = builder.hmc(num_leapfrog_steps=3)
    kernel = builder.build()
    found_steps = unnest.get_innermost(kernel, 'num_leapfrog_steps')
    self.assertEqual(found_steps, 3)
    builder = builder.hmc(num_leapfrog_steps=5)
    kernel = builder.build()
    found_steps = unnest.get_innermost(kernel, 'num_leapfrog_steps')
    self.assertEqual(found_steps, 5)

  def test_transform_is_stateful(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = (
        builder
        .hmc(num_leapfrog_steps=3)
        .transform([]))
    self.assertIsInstance(
        builder.build().inner_kernel,
        transformed_kernel.TransformedTransitionKernel)
    builder = builder.nuts()
    self.assertIsInstance(
        builder.build().inner_kernel,
        transformed_kernel.TransformedTransitionKernel)

  def test_adaptation_needs_num_steps(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = (
        builder
        .hmc(num_leapfrog_steps=3)
        .simple_adaptation(adaptation_rate=.03))
    self.assertRaises(ValueError, builder.build)

  def test_clear_step_adapter(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = (
        builder
        .hmc(num_leapfrog_steps=3)
        .simple_adaptation(adaptation_rate=.03))
    kernel = builder.build(10).inner_kernel
    self.assertIsInstance(
        kernel, simple_step_size_adaptation.SimpleStepSizeAdaptation)

    builder = builder.clear_step_adapter()
    kernel = builder.build(10).inner_kernel
    self.assertIsInstance(kernel, hmc.HamiltonianMonteCarlo)

  def test_reducers(self):
    builder = kernel_builder.KernelBuilder.make(lambda x: x)
    builder = builder.hmc(num_leapfrog_steps=3)
    self.assertIsInstance(builder.build(), with_reductions.WithReductions)
    builder = builder.clear_tracing()
    self.assertIsInstance(builder.build(), hmc.HamiltonianMonteCarlo)
    builder = builder.set_reducer([])
    self.assertIsInstance(builder.build(), with_reductions.WithReductions)
    builder = builder.clear_reducer()
    self.assertIsInstance(builder.build(), hmc.HamiltonianMonteCarlo)
    builder = builder.set_show_progress()
    self.assertIsInstance(builder.build(10), with_reductions.WithReductions)
    builder = builder.set_show_progress(False)
    self.assertIsInstance(builder.build(), hmc.HamiltonianMonteCarlo)


if __name__ == '__main__':
  tf.test.main()
