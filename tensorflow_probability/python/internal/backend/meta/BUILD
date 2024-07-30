# Copyright 2019 The TensorFlow Probability Authors.
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
# Description:
#   Numpy backend rewriter.

# Placeholder: py_binary

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_binary(
    name = "gen_linear_operators",
    srcs = ["gen_linear_operators.py"],
    deps = [
        # absl:app dep,
        # tensorflow dep,
    ],
)

py_binary(
    name = "gen_tensor_shape",
    srcs = ["gen_tensor_shape.py"],
    deps = [
        # absl:app dep,
        # tensorflow dep,
    ],
)
