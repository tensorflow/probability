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
"""Jax implementation of deferred_tensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def DeferredTensor(pretransformed_input, transform_fn,
                   dtype=None, shape='None', name=None):  # pylint: disable=unused-argument
  # DeferredTensor is used to address tape-safety issues in TF2
  # which do not exist in the JAX backend
  # so it is safe to evaluate the function immediately
  return transform_fn(pretransformed_input)


def TransformedVariable(initial_value, bijector,  # pylint: disable=unused-argument
                        dtype=None, name=None, **kwargs):  # pylint: disable=unused-argument
  return DeferredTensor(initial_value, bijector)
