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
# TFP with a numpy backend.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "numpy",
    srcs = ["__init__.py"],
    deps = [
        "//tensorflow_probability/python/experimental/substrates/numpy/bijectors",
        "//tensorflow_probability/python/experimental/substrates/numpy/distributions",
        "//tensorflow_probability/python/experimental/substrates/numpy/experimental",
        "//tensorflow_probability/python/experimental/substrates/numpy/internal",
        "//tensorflow_probability/python/experimental/substrates/numpy/stats",
        "//tensorflow_probability/python/experimental/substrates/numpy/util",
        "//tensorflow_probability/python/internal/backend",
        "//tensorflow_probability/python/mcmc:mcmc.numpy",
    ],
)
