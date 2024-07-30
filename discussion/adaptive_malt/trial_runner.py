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
r"""Trial runner.

Usage:

```
trial_search_runner \
  --hparams='experiment.output_dir: /tmp'
```

See trial_runner_experiments.py for the hyperparameters.

"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from etils import epath
import gin
from discussion.adaptive_malt import adaptive_malt
from discussion.adaptive_malt import utils

_HPARAMS = flags.DEFINE(utils.YAMLDictParser(), 'hparams', '',
                        'Hyperparameters to override.')


@gin.configurable
def experiment(output_dir: str):
  """Runs an experiment."""
  epath.Path(output_dir).mkdir(parents=True, exist_ok=True)
  res = adaptive_malt.run_trial(  # pytype: disable=missing-parameter
  )
  utils.save_h5py(os.path.join(output_dir, 'trial.h5'), res)


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
