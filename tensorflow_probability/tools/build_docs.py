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
"""Tool to generate external api_docs for tensorflow_probability.

Note:
  If duplicate or spurious docs are generated (e.g. internal names), consider
  blacklisting them via the `private_map` argument below.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
import tensorflow_probability as tfp


flags.DEFINE_string("output_dir", "/tmp/probability_api",
                    "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    ("https://github.com/tensorflow/probability/blob/master/"
     "tensorflow_probability"),
    "The url prefix for links to code.")

flags.DEFINE_bool("search_hints", True,
                  "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "probability/api_docs/python",
                    "Path prefix in the _toc.yaml")

FLAGS = flags.FLAGS

DO_NOT_GENERATE_DOCS_FOR = [
    tfp.experimental.substrates.jax.tf2jax,
    tfp.experimental.substrates.numpy.tf2numpy,
]


def internal_filter(path, parent, children):
  """Skip any object with "internal" in the name."""
  del path
  del parent
  children = [
      (name, value) for (name, value) in children if "internal" not in name
  ]
  return children


def main(unused_argv):
  for obj in DO_NOT_GENERATE_DOCS_FOR:
    doc_controls.do_not_generate_docs(obj)

  doc_generator = generate_lib.DocGenerator(
      root_title="TensorFlow Probability",
      py_modules=[("tfp", tfp)],
      base_dir=os.path.dirname(tfp.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map={"tfp": ["google", "staging", "python"]},
      callbacks=[internal_filter])

  doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
  app.run(main)
