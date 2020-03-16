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
# TFP with a JAX backend.

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

exports_files(["LICENSE"])

py_library(
    name = "jax",
    srcs = ["__init__.py"],
    deps = [
        "//tensorflow_probability/python/experimental/substrates/jax/bijectors",
        "//tensorflow_probability/python/experimental/substrates/jax/distributions",
        "//tensorflow_probability/python/experimental/substrates/jax/experimental",
        "//tensorflow_probability/python/experimental/substrates/jax/internal",
        "//tensorflow_probability/python/experimental/substrates/jax/stats",
        "//tensorflow_probability/python/experimental/substrates/jax/util",
        "//tensorflow_probability/python/internal/backend",
        "//tensorflow_probability/python/mcmc:mcmc.jax",
        "//tensorflow_probability/python/optimizer:optimizer.jax",
    ],
)
