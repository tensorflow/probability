# Copyright 2021 The TensorFlow Probability Authors.
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
licenses(["notice"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "backend_jax",
    srcs = ["__init__.py"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # jax dep,
        # jax:stax dep,
    ],
)

py_library(
    name = "tf_on_jax",
    srcs = ["tf_on_jax.py"],
    deps = [
        # jax dep,
        # jax:stax dep,
    ],
)

py_library(
    name = "backend",
    srcs = ["backend.py"],
    deps = [
        ":tf_on_jax",
        ":util",
        # tensorflow_probability/substrates:jax dep,
    ],
)
