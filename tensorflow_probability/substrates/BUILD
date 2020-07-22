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
#   API-unstable code that is part of the TFP package.

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "substrates",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":jax",
        ":numpy",
        "//tensorflow_probability/python/internal:all_util",
        "//tensorflow_probability/python/internal:lazy_loader",
        "//tensorflow_probability/python/internal/backend",
    ],
)

py_library(
    name = "jax",
    srcs = ["jax/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/bijectors:bijectors.jax",
        "//tensorflow_probability/python/distributions:distributions.jax",
        "//tensorflow_probability/python/experimental:experimental.jax",
        "//tensorflow_probability/python/internal:internal.jax",
        "//tensorflow_probability/python/math:math.jax",
        "//tensorflow_probability/python/mcmc:mcmc.jax",
        "//tensorflow_probability/python/optimizer:optimizer.jax",
        "//tensorflow_probability/python/random:random.jax",
        "//tensorflow_probability/python/stats:stats.jax",
        "//tensorflow_probability/python/util:util.jax",
    ],
)

py_library(
    name = "numpy",
    srcs = ["numpy/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_probability/python/bijectors:bijectors.numpy",
        "//tensorflow_probability/python/distributions:distributions.numpy",
        "//tensorflow_probability/python/experimental:experimental.numpy",
        "//tensorflow_probability/python/internal:internal.numpy",
        "//tensorflow_probability/python/math:math.numpy",
        "//tensorflow_probability/python/mcmc:mcmc.numpy",
        "//tensorflow_probability/python/optimizer:optimizer.numpy",
        "//tensorflow_probability/python/random:random.numpy",
        "//tensorflow_probability/python/stats:stats.numpy",
        "//tensorflow_probability/python/util:util.numpy",
    ],
)
