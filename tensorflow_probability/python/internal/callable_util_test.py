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
"""Tests for cache_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import callable_util
from tensorflow_probability.python.internal import test_util


def _return_args_from_infinite_loop(*loop_vars, additional_loop_vars=()):
  return tf.while_loop(
      cond=lambda *_: True,
      body=lambda *loop_vars: loop_vars,
      loop_vars=loop_vars + additional_loop_vars)


class CallableUtilTest(test_util.TestCase):

  @test_util.numpy_disable_test_missing_functionality('Tracing not supported')
  def test_get_output_spec_avoids_evaluating_fn(self):
    args = (np.array(0., dtype=np.float64),
            (tf.convert_to_tensor(0.),
             tf.convert_to_tensor([1., 1.], dtype=tf.float64)))
    additional_args = (tf.convert_to_tensor([[3], [4]], dtype=tf.int32),)
    # Trace using both positional and keyword args.
    results = callable_util.get_output_spec(
        _return_args_from_infinite_loop,
        *args,
        additional_loop_vars=additional_args)
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: tf.convert_to_tensor(x).shape,
                              args + additional_args),
        tf.nest.map_structure(lambda x: x.shape, results))
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: tf.convert_to_tensor(x).dtype,
                              args + additional_args),
        tf.nest.map_structure(lambda x: x.dtype, results))

  @test_util.numpy_disable_test_missing_functionality('Tracing not supported')
  @test_util.jax_disable_test_missing_functionality('b/174071016')
  def test_get_output_spec_from_tensor_specs(self):
    args = (tf.TensorSpec([], dtype=tf.float32),
            (tf.TensorSpec([1, 1], dtype=tf.float32),
             tf.TensorSpec([2], dtype=tf.float64)))
    additional_args = (tf.TensorSpec([2, 1], dtype=tf.int32),)
    # Trace using both positional and keyword args.
    results = callable_util.get_output_spec(
        _return_args_from_infinite_loop,
        *args,
        additional_loop_vars=additional_args)
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: x.shape, args + additional_args),
        tf.nest.map_structure(lambda x: x.shape, results))
    self.assertAllEqualNested(
        tf.nest.map_structure(lambda x: x.dtype, args + additional_args),
        tf.nest.map_structure(lambda x: x.dtype, results))

if __name__ == '__main__':
  tf.test.main()
