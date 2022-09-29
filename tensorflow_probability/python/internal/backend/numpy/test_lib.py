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
"""Numpy implementations of TensorFlow functions."""


JAX_MODE = False
NUMPY_MODE = not JAX_MODE

__all__ = [
    'is_gpu_available',
    'Benchmark',
]


# --- Begin Public Functions --------------------------------------------------


is_gpu_available = lambda: False


if not (JAX_MODE or NUMPY_MODE):
  import tensorflow.compat.v2 as tf  # pylint: disable=g-import-not-at-top

  class Benchmark(tf.test.Benchmark):
    pass

  main = tf.test.main
else:

  class Benchmark(object):
    pass

  main = None
