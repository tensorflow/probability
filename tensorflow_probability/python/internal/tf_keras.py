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
"""Utility for importing the correct version of Keras."""

import tensorflow.compat.v2 as tf

# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
# pylint: disable=wildcard-import
try:
  _keras_version_fn = getattr(tf.keras, "version", None)
  _use_tf_keras = _keras_version_fn and _keras_version_fn().startswith("3.")
  del _keras_version_fn
except ImportError:
  _use_tf_keras = True
if _use_tf_keras:
  from tf_keras import *
  from tf_keras import __internal__
  import tf_keras.api._v1.keras.__internal__.legacy.layers as tf1_layers
  import tf_keras.api._v1.keras as v1
else:
  from tensorflow.compat.v2.keras import *
  from tensorflow.compat.v2.keras import __internal__
  import tensorflow.compat.v1 as tf1
  v1 = tf1.keras
  tf1_layers = tf1.layers
  del tf1

del tf
del _use_tf_keras
