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
"""XLA tests for TensorFlow Probability ODE solvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util

flags.DEFINE_string('test_device', None,
                    'TensorFlow device on which to place operators under test')
FLAGS = flags.FLAGS

_RTOL = 1e-8
_ATOL = 1e-12


def linear(solver, jacobian_diag_part, initial_state):
  ode_fn = lambda time, state: jacobian_diag_part * state
  initial_time = 0.
  jacobian = np.diag(jacobian_diag_part)
  solver_instance = solver(rtol=_RTOL, atol=_ATOL)
  results = solver_instance.solve(
      ode_fn,
      initial_time,
      initial_state,
      solution_times=[1.],
      jacobian_fn=jacobian)
  return results.times, results.states


@parameterized.named_parameters([('bdf', tfp.math.ode.BDF)])
class XLATest(test_util.TestCase):

  def test_linear(self, solver):
    jacobian_diag_part = np.float32([-0.5, -1.])
    initial_state = np.float32([1., 2.])
    fn = lambda: linear(solver, jacobian_diag_part, initial_state)
    fn = tf.function(fn, autograph=False, experimental_compile=True)
    with tf.device(FLAGS.test_device):
      times, states = self.evaluate(fn())
    states_exact = np.exp(jacobian_diag_part[np.newaxis, :] *
                          times[:, np.newaxis]) * initial_state
    self.assertAllClose(states, states_exact, rtol=1e-4)


if __name__ == '__main__':
  tf.test.main()
