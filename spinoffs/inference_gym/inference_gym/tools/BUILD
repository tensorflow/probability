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
# Ground truth computation.

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
        "//inference_gym:__subpackages__",
    ],
)

licenses(["notice"])

# We can't use strict/pytype because `cmdstanpy` is not available internally.
py_binary(
    name = "get_ground_truth",
    srcs = ["get_ground_truth.py"],
    python_version = "PY3",
    deps = [
        # tensorflow_probability dep,
        "//inference_gym/internal:ground_truth_encoding",
        "//inference_gym/tools/stan:targets",
    ],
)
