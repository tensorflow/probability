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
"""Tool to generate external api_docs for oryx."""
import os

from absl import app
from absl import flags
from tensorflow_docs.api_generator import generate_lib
import oryx


flags.DEFINE_string("output_dir", "/tmp/oryx_api",
                    "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    ("https://github.com/tensorflow/probability/blob/master/spinoffs/oryx/"
     "oryx"), "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "probability/oryx/api_docs",
                    "Path prefix in the _toc.yaml")

FLAGS = flags.FLAGS


def main(unused_argv):
  doc_generator = generate_lib.DocGenerator(
      root_title="Oryx",
      py_modules=[("oryx", oryx)],
      base_dir=os.path.dirname(oryx.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={"oryx.core": ["kwargs_util"]})

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
