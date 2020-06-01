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
r"""Run a Stan model to get the ground truth.

This will run your target distribution using PyStan and generate a Python source
file with global variables containing the ground truth values.

Usage (run from TensorFlow Probability source directory):
```
venv=$(mktemp -d)
virtualenv -p python3.6 $venv
source $venv/bin/activate
pip install cmdstanpy==0.8 pandas numpy tf-nightly tfds-nightly
install_cmdstan

bazel run //tools/inference_gym_ground_truth:get_ground_truth -- \
  --target=<function name from targets.py>
```

NOTE: By default this will run for a *really* long time and use *a lot* of RAM,
be cautious! Reduce the value of the `stan_samples` flag to make things more
reasonable for quick tests.

NOTE: This must be run locally, and requires at least the following packages:

- cmdstanpy (also cmdstan: `pip install cmdstanpy; install_cmdstan`)
- numpy
- pandas
- tf-nightly
- tfds-nightly
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tools.inference_gym_ground_truth import targets
from tensorflow_probability.python.experimental.inference_gym.internal import array_to_source

flags.DEFINE_enum('target', None, targets.__all__, 'Which Stan model to '
                  'sample from.')
flags.DEFINE_integer('stan_samples', 150000,
                     'Number of samples to ask from Stan.')
flags.DEFINE_integer('stan_chains', 10, 'Number of chains to ask from Stan.')
flags.DEFINE_boolean('print_summary', True, 'Whether to print the Stan fit'
                     'summary')
flags.DEFINE_string('output_directory', None,
                    'Where to save the ground truth values. By default, this '
                    'places it in the appropriate directory in the '
                    'TensorFlow Probability source directory.')

FLAGS = flags.FLAGS


@tf.function(autograph=False)
def get_ess(samples):
  return tfp.mcmc.effective_sample_size(
      samples,
      filter_beyond_positive_pairs=True,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.enable_v2_behavior()

  stan_model = getattr(targets, FLAGS.target)()

  with stan_model.sample_fn(
      sampling_iters=FLAGS.stan_samples,
      chains=FLAGS.stan_chains,
      show_progress=True) as mcmc_output:
    summary = mcmc_output.summary()
    if FLAGS.print_summary:
      pd.set_option('display.max_rows', sys.maxsize)
      pd.set_option('display.max_columns', sys.maxsize)
      print(mcmc_output.diagnose())
      print(summary)

    array_strs = []
    for name, fn in sorted(stan_model.extract_fns.items()):
      transformed_samples = []

      mean = 0
      std = 0
      ess = 0
      for chain_id in range(FLAGS.stan_chains):
        # TODO(https://github.com/stan-dev/cmdstanpy/issues/218): This step is
        # very slow and wastes memory. Consider reading the CSV files ourselves.
        chain = mcmc_output.sample[:, chain_id, :]
        dataframe = pd.DataFrame(chain, columns=mcmc_output.column_names)

        transformed_samples = fn(dataframe)

        mean += transformed_samples.mean(0)
        std += transformed_samples.std(0)
        ess += get_ess(tf.convert_to_tensor(transformed_samples)).numpy()
      mean /= FLAGS.stan_chains
      std /= FLAGS.stan_chains
      sem = std / np.sqrt(ess)

      upper_name = name.upper()
      mean_name = '{}_MEAN'.format(upper_name)
      sem_name = '{}_MEAN_STANDARD_ERROR'.format(upper_name)
      std_name = '{}_STANDARD_DEVIATION'.format(upper_name)

      array_strs.append(array_to_source.array_to_source(mean_name, mean))
      array_strs.append(array_to_source.array_to_source(sem_name, sem))
      array_strs.append(array_to_source.array_to_source(std_name, std))

  array_str = '\n'.join(array_strs)
  argv_str = '\n'.join(['  {} \\'.format(arg) for arg in sys.argv[1:]])

  file_str = r'''# Lint as: python2, python3
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
r"""Ground truth values for `{target}`.

Automatically generated using the command:

```
bazel run //tools/inference_gym_ground_truth:get_ground_truth -- \
{argv_str}
```
"""

import numpy as onp

{array_str}'''.format(
    target=FLAGS.target, array_str=array_str, argv_str=argv_str)

  if FLAGS.output_directory is None:
    file_basedir = os.path.dirname(os.path.realpath(__file__))
    output_directory = os.path.join(
        file_basedir, '../../tensorflow_probability/python/experimental/'
        'inference_gym/targets/ground_truth')
  else:
    output_directory = FLAGS.output_directory
  file_path = os.path.join(output_directory, '{}.py'.format(FLAGS.target))
  print('Writing ground truth values to: {}'.format(file_path))
  with open(file_path, 'w') as f:
    f.write(file_str)


if __name__ == '__main__':
  flags.mark_flag_as_required('target')
  app.run(main)
