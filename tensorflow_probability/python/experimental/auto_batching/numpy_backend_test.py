# Copyright 2018 The TensorFlow Probability Authors.
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
"""Tests for implementations of batched variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import hypothesis as hp
from hypothesis import strategies as hps
from hypothesis.extra import numpy as hpnp
import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.auto_batching import backend_test_lib as backend_test
from tensorflow_probability.python.experimental.auto_batching import instructions as inst
from tensorflow_probability.python.experimental.auto_batching import numpy_backend
from tensorflow_probability.python.internal import test_util as tfp_test_util

NP_BACKEND = numpy_backend.NumpyBackend()


def var_init(max_stack_depth, initial_value):
  type_ = inst.TensorType(initial_value.dtype, initial_value.shape[1:])
  var = NP_BACKEND.create_variable(
      None, inst.VariableAllocation.FULL, type_,
      max_stack_depth, batch_size=initial_value.shape[0])
  return var.update(
      initial_value, NP_BACKEND.full_mask(initial_value.shape[0]))


# A TF test case for self.assertAllEqual, but doesn't use TF so doesn't care
# about Eager vs Graph mode.
class NumpyVariableTest(tfp_test_util.TestCase, backend_test.VariableTestCase):

  def testNumpySmoke(self):
    """Test the property on specific example, without relying on Hypothesis."""
    init = (12, np.random.randn(3, 2, 2).astype(np.float32))
    ops = [('pop', [False, False, True]),
           ('push', [True, False, True]),
           ('update', np.ones((3, 2, 2), dtype=np.float32),
            [True, True, False]),
           ('pop', [True, False, True])]
    self.check_same_results(init, ops, var_init)

  @hp.given(hps.data())
  @hp.settings(
      deadline=None,
      max_examples=100)
  def testNumpyVariableRandomOps(self, data):
    # Hypothesis strategy:
    # Generate a random max stack depth and value shape
    # Deduce the batch size from the value shape
    # Make a random dtype
    # Generate a random initial value of that dtype and shape
    # Generate ops, some of which write random values of that dtype and shape
    max_stack_depth = data.draw(hps.integers(min_value=1, max_value=1000))
    value_shape = data.draw(hpnp.array_shapes(min_dims=1))
    batch_size = value_shape[0]
    dtype = data.draw(hpnp.scalar_dtypes())
    masks = hpnp.arrays(dtype=np.bool, shape=[batch_size])
    values = hpnp.arrays(dtype, value_shape)
    init_val = data.draw(values)
    ops = data.draw(
        hps.lists(
            hps.one_of(
                hps.tuples(hps.just('update'), values, masks),
                hps.tuples(hps.just('push'), masks),
                hps.tuples(hps.just('pop'), masks),  # preserve line break
                hps.tuples(hps.just('read')))))
    self.check_same_results((max_stack_depth, init_val), ops, var_init)


if __name__ == '__main__':
  tf.test.main()
