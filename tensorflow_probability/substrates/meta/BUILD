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
# Tests for the backend integration.

# Placeholder: py_binary
# Placeholder: py_test

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_test(
    name = "numpy_integration_test",
    size = "small",
    srcs = ["numpy_integration_test.py"],
    tags = ["tfp_numpy"],
    deps = [
        "//tensorflow_probability/substrates:numpy",
    ],
)

py_test(
    name = "jax_integration_test",
    srcs = ["jax_integration_test.py"],
    tags = ["tfp_jax"],
    deps = [
        # jax dep,
        "//tensorflow_probability/substrates:jax",
    ],
)

py_binary(
    name = "rewrite",
    srcs = ["rewrite.py"],
    deps = [
        # absl:app dep,
    ],
)
