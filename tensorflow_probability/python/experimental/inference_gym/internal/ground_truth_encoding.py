# Lint as: python2, python3
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
"""Functions to encode/decode ground truth into Python sources.

Sample transformation ground truth is composed of up to 4 quantities (mean,
standard deviation and standard errors of both), each of which can be a nested
structure of Numpy arrays. We encode them using global variables via
the `array_to_source` utility.  We derive the names of these global variables
as follows:

```none
constant_name ::= transformation_name [ "_" tuple_path ] "_"
                  quantity_name

tuple_path = path_component [ "_" path_component ]*

quantity_name ::= "MEAN" | "MEAN_STANDARD_ERROR" |
                  "STANDARD_DEVIATION" | "STANDARD_DEVIATION_STANDARD_ERROR"
```

Here `transfomation_name` is the name of the sample transfromation.
`tuple_path` is the tuple path (as defined by `nest.flatten_with_tuple_paths`)
of the relevant part of the nested ground truth quantity. `quantity_name` is one
of the 4 supported quantities. The global variable name is upper-cased to
conform to style guidelines. The ground truth file is allowed to omit unknown
quantities.

#### Examples

##### A non-nested ground truth.

```
name = "test"
mean_value = 3.
```

is encoded (modulo the encoding used by `array_to_source`) as:

```
TEST_MEAN = 3.
```

##### Nested ground truth.

```
name = "test"
mean_value = [1., {'b': 2.}]
```

is encoded (modulo the encoding used by `array_to_source`) as:

```
TEST_0_MEAN = 1.
TEST_1_B_MEAN = 2.
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_probability.python.experimental.inference_gym.internal import array_to_source

__all__ = [
    'get_ground_truth_module_source',
    'load_ground_truth_part',
    'save_ground_truth_part',
]


def _get_global_variable_names(name, tuple_path):
  """Get the global variables names for ground truth.

  Args:
    name: Python `str`. Name of the sample transformation.
    tuple_path: Tuple path of the part of the ground truth we're saving. See
      `nest.flatten_with_tuple_paths`.

  Returns:
    mean_name: Variable name to use for the mean.
    sem_name: Variable name to use for the standard error of the mean.
    std_name: Variable name to use for the standard deviation.
    sestd_name: Variable name to use for the standard deviation standard error.
  """
  if tuple_path:
    path_suffix = '_' + '_'.join(str(p).upper() for p in tuple_path)
  else:
    path_suffix = ''

  upper_name = name.upper()
  mean_name = '{}{}_MEAN'.format(upper_name, path_suffix)
  sem_name = '{}{}_MEAN_STANDARD_ERROR'.format(upper_name, path_suffix)
  std_name = '{}{}_STANDARD_DEVIATION'.format(upper_name, path_suffix)
  sestd_name = '{}{}_STANDARD_DEVIATION_STANDARD_ERROR'.format(
      upper_name, path_suffix)

  return mean_name, sem_name, std_name, sestd_name


def load_ground_truth_part(module, name, tuple_path):
  """Loads a ground truth part from a module.

  We assume the module was created by saving the output of
  `save_ground_truth_part` to disk.

  This is meant to be called with the outputs of
  `nest.flatten_with_tuple_paths(sample_transformation.dtype)`.

  Args:
    module: Python module to load from.
    name: Python `str`. Name of the sample transformation.
    tuple_path: Tuple path of the part of the ground truth we're loading. See
      `nest.flatten_with_tuple_paths`.

  Returns:
    mean: Ground truth mean, or `None` if it is absent.
    sem: Ground truth stadard error of the mean, or `None` if it is absent.
    std: Ground truth standard deviation, or `None` if it is absent.
    sestd: Ground truth mean, or `None` if it is absent.
  """

  mean_name, sem_name, std_name, sestd_name = _get_global_variable_names(
      name, tuple_path)

  return (getattr(module, mean_name, None), getattr(module, sem_name, None),
          getattr(module, std_name, None), getattr(module, sestd_name, None))


def save_ground_truth_part(name, tuple_path, mean, sem, std, sestd):
  """Saves a ground truth part to strings.

  This is meant to be called with outputs of
  `nest.flatten_with_tuple_paths(ground_truth_mean)`.

  Args:
    name: Python `str`. Name of the sample transformation.
    tuple_path: Tuple path of the part of the ground truth we're saving. See
      `nest.flatten_with_tuple_paths`.
    mean: Ground truth mean, or `None` if it is absent.
    sem: Ground truth stadard error of the mean, or `None` if it is absent.
    std: Ground truth standard deviation, or `None` if it is absent.
    sestd: Ground truth mean, or `None` if it is absent.

  Returns:
    array_strs: Python list of strings, representing the encoded arrays (that
      were present). Typically these would be joined with a newline and written
      out to a module, which can then be passed to `load_ground_truth_part`.
  """

  array_strs = []
  mean_name, sem_name, std_name, sestd_name = _get_global_variable_names(
      name, tuple_path)

  if mean is not None:
    array_strs.append(array_to_source.array_to_source(mean_name, mean))

  if sem is not None:
    array_strs.append(array_to_source.array_to_source(sem_name, sem))

  if std is not None:
    array_strs.append(array_to_source.array_to_source(std_name, std))

  if sestd is not None:
    array_strs.append(array_to_source.array_to_source(sestd_name, sestd))

  return array_strs


def get_ground_truth_module_source(target_name, command_str, array_strs):
  """Gets the source of a module that will contain the ground truth.

  Args:
    target_name: Name of the target.
    command_str: String of the command name used to generate the ground truth
      values.
    array_strs: List of strings that encode the ground truth arrays. See
      `save_ground_truth_part`.

  Returns:
    module_source: Source of the module.
  """
  array_str = '\n'.join(array_strs)

  module_source = r'''# Lint as: python2, python3
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
r"""Ground truth values for `{target_name}`.

Automatically generated using the command:

```
{command_str}
```
"""

import numpy as np

{array_str}'''.format(
    target_name=target_name, array_str=array_str, command_str=command_str)

  return module_source
