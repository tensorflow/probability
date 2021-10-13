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
"""Tests that linop static files are in sync with TF."""

import os

from absl import flags
from absl import logging
from absl.testing import absltest


flags.DEFINE_list('modules_to_check', [],
                  'list of TF -> numpy modules to verify')
flags.DEFINE_bool('update', False, 'update files that are out of date')

FLAGS = flags.FLAGS


TFP_PYTHON_DIR = 'tensorflow_probability/python'


class RewriteEquivalenceTest(absltest.TestCase):

  def test_file_equivalence_after_rewrite(self):
    wrong_modules = []
    for module in FLAGS.modules_to_check:
      static_file = os.path.join(
          absltest.get_default_test_srcdir(),
          '{}/internal/backend/numpy/gen/{}.py'.format(TFP_PYTHON_DIR, module))
      gen_file = os.path.join(
          absltest.get_default_test_srcdir(),
          '{}/internal/backend/numpy/{}_gen.py'.format(TFP_PYTHON_DIR, module))
      try:
        with open(static_file, 'r') as f:
          static_content = f.read()
      except IOError:
        static_content = None
      try:
        with open(gen_file, 'r') as f:
          gen_content = f.read()
      except IOError:
        gen_content = None
      if gen_content is None and static_content is None:
        raise ValueError('Could not load content for {}'.format(static_file))
      if gen_content != static_content:
        if FLAGS.update:
          to_update = static_file.split('runfiles/')[-1]
          to_update = '/'.join(to_update.split('/')[1:])
          with open(to_update, 'w') as f:
            f.write(gen_content)
          logging.info('Updating file %s', to_update)
        else:
          wrong_modules.append(module)
    if wrong_modules:
      msg = '\n'.join([
          'Modules `{}` require updates.  To update them, run'.format(
              repr(wrong_modules)),
          'bazel build -c opt :rewrite_equivalence_test',
          'bazel-py3/bin/.../rewrite_equivalence_test --update '
          '--modules_to_check={}'.format(','.join(wrong_modules)),
          'It may be necessary to adapt the generator programs.'
      ])
      raise AssertionError(msg)


if __name__ == '__main__':
  absltest.main()
