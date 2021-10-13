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
"""Numpy implementations of sparse functions."""

from tensorflow_probability.python.internal.backend.numpy import _utils as utils


__all__ = [
    'to_dense',
]


def _to_dense(sp_input, default_value=0, validate_indices=True, name=None):  # pylint: disable=unused-argument
  if default_value != 0:
    raise NotImplementedError(
        'Argument `default_value != 0` is currently unimplemented.')
  if not validate_indices:
    raise NotImplementedError(
        'Argument `validate_indices != True` is currently unimplemented.')
  return sp_input


# --- Begin Public Functions --------------------------------------------------


# TODO(b/136555907): Add unit test.
to_dense = utils.copy_docstring(
    'tf.sparse.to_dense',
    _to_dense)
