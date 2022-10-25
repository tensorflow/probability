# Copyright 2022 The TensorFlow Probability Authors.
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
r"""Grid search runner.

Usage:

```
grid_search_runner \
  --hparams='experiment.output_dir: /tmp, experiment.grid_index: [0, 1]'
```

See grid_search_runner_experiments.py for the hyperparameters.

"""

from collections.abc import Sequence
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from etils import epath
import gin
import numpy as np
from discussion.adaptive_malt import adaptive_malt
from discussion.adaptive_malt import utils

_HPARAMS = flags.DEFINE(utils.YAMLDictParser(), 'hparams', '',
                        'Hyperparameters to override.')


@gin.configurable
def experiment(output_dir: str,
               grid_index: np.ndarray,
               mean_trajectory_length_vals: Optional[np.ndarray] = None,
               damping_vals: Optional[np.ndarray] = None):
  """Runs an experiment."""
  epath.Path(output_dir).mkdir(parents=True, exist_ok=True)
  if mean_trajectory_length_vals is not None:
    for j, mean_trajectory_length in enumerate(mean_trajectory_length_vals):
      logging.info('Starting %d', j)
      whole_grid_index = [grid_index, j]
      res = adaptive_malt.run_grid_element(  # pytype: disable=missing-parameter
          mean_trajectory_length=mean_trajectory_length,
          seed=np.random.RandomState(list(whole_grid_index)).randint(1 << 32))
      utils.save_h5py(
          os.path.join(output_dir,
                       f'{whole_grid_index[0]}.{whole_grid_index[1]}.h5'), res)
  elif damping_vals is not None:
    for i, damping in enumerate(damping_vals):
      logging.info('Starting %d', i)
      whole_grid_index = [i, grid_index]
      res = adaptive_malt.run_grid_element(  # pytype: disable=missing-parameter
          damping=damping,
          seed=np.random.RandomState(list(whole_grid_index)).randint(1 << 32))
      utils.save_h5py(
          os.path.join(output_dir,
                       f'{whole_grid_index[0]}.{whole_grid_index[1]}.h5'), res)
  else:
    res = adaptive_malt.run_grid_element(  # pytype: disable=missing-parameter
        seed=np.random.RandomState(list(grid_index)).randint(1 << 32))
    utils.save_h5py(
        os.path.join(output_dir, f'{grid_index[0]}.{grid_index[1]}.h5'), res)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  utils.bind_hparams(_HPARAMS.value)
  # pylint: disable=no-value-for-parameter
  # pytype: disable=missing-parameter
  experiment()
  # pytype: enable=missing-parameter


if __name__ == '__main__':
  app.run(main)
