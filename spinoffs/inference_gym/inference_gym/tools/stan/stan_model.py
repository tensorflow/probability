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
"""Stan model."""

import collections

__all__ = [
    'StanModel',
]


class StanModel(
    collections.namedtuple('StanModel', [
        'extract_fns',
        'sample_fn',
    ])):
  """Describes a Stan model.

  See `logistic_regression.py` for an example.

  Attributes:
    sample_fn: A callable with the same signature as `CmdStanModel.sample`. It
      should generate samples and generated quantities from the model.
    extract_fns: A dictionary of strings to functions that extract transformed
      samples from a dataframe holding the outputs of a single Stan chain. The
      keys must be the same as the keys in the `sample_transformations` field of
      the corresponding member of `inference_gym/targets`, and the values must
      be commensurable to the return values of the those transformations.
  """
  __slots__ = ()
