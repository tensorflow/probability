#!/usr/bin/env bash
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

#
# Using bazel to run test TFP tests in a virtualenv (which we strongly
# recommend) requires some special flags. Additionally, some tests are defined
# but known not to be working and are therefore tagged as such; using an
# additional flag ensures such tests are skipped. This script wraps the `bazel`
# command and ensures these flags are set. Arguments to this test are passed
# along to `bazel test` after the aforementioned flags
#
# ## Example 1: Run all tests (from TFP repo root dir)
#
# ```bash
#  $ ./testing/run_tfp_test.sh //tensorflow_probability/...
# ```
#
# ## Example 2: Repeat test with varying seeds
#
# ```bash
#   $ ./testing/run_tfp_test.sh --test_arg="--vary_seed --runs_per_test=100" \
#       //tensorflow_probability/path/to:stochastic_test
# ```

set -x  # print commands as they are executed
set -e  # fail and exit on any command erroring
set -u  # fail and exit on any undefined variable reference

if ! which bazel; then
  echo '`bazel` must be installed to run TFP tests with this script.'
  exit 1
fi

# Run tests. Notes on less obvious options:
#   --test_timeout -- comma separated values correspond to various test sizes
#     (short, moderate, long or eternal). We increase the timeouts from their
#     defaults to allow for slower machines.
#   --action_env -- specify environment vars to pass through to action
#     environment. (We need these in order to run inside a virtualenv.)
#     See https://github.com/bazelbuild/bazel/issues/6648 and b/121259040.
#   --test_env=TFP_HYPOTHESIS_MAX_EXAMPLES=2 -- several tests in TFP use the
#     hypothesis testing framework to automatically probe for bugs by generating
#     test examples. This flag limits the number of examples to 2, which helps
#     prevent extreme test slowness, but reduces coverage.
#   --test_tag_filters -- some tests are known to be broken in certain
#     configurations and are labelled as such with so-called "tags" (in the
#     relevant BUILD files). This flag tells bazel to avoid running those test
#     configurations.
bazel test \
  --compilation_mode=opt \
  --copt=-O3 \
  --copt=-march=native \
  --test_timeout 300,450,1200,3600 \
  --test_tag_filters="-gpu,-requires-gpu-nvidia,-notap,-no-oss-ci,-tf2-broken,-tf2-kokoro-broken" \
  --test_env=TFP_HYPOTHESIS_MAX_EXAMPLES=2 \
  --test_env=WRAPT_DISABLE_EXTENSIONS=true \
  --action_env=PATH \
  --action_env=LD_LIBRARY_PATH \
  --test_output=errors \
  $@
