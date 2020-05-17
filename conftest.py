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
"""Blacklist for pytest."""

from absl import app


collect_ignore = [
    "discussion/",
    "setup.py",
    "tensorflow_probability/python/experimental/substrates/"
]


def pytest_addoption(parser):
  parser.addoption(
      "--absl-flag",
      action="append",
      help="flag to be passed to absl, e.g. `--absl-flag='--vary_seed'`",
      default=[]
  )


def pytest_collection_finish(session):
  # Unlike bazel, `pytest` doesn't invoke `tf.test.run()` (which parses flags),
  # so for external developers using pytest we just parse the flags directly.
  absl_flags = session.config.getoption("absl_flag", default=[])
  app._register_and_parse_flags_with_usage(["test.py"] + absl_flags)  # pylint: disable=protected-access
