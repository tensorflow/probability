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
#   tf.Module tools for building neural architectures.

licenses(["notice"])

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "nn",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":affine_layers",
        ":convolutional_layers",
        ":convolutional_layers_v2",
        ":convolutional_transpose_layers",
        ":layers",
        ":variational_base",
        "//tensorflow_probability/python/experimental/nn/initializers",
        "//tensorflow_probability/python/experimental/nn/losses",
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:all_util",
    ],
)

py_library(
    name = "affine_layers",
    srcs = ["affine_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":variational_base",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/experimental/nn/util:kernel_bias",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "affine_layers_test",
    size = "medium",
    srcs = ["affine_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "convolutional_layers",
    srcs = ["convolutional_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":variational_base",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "convolutional_layers_test",
    size = "medium",
    srcs = ["convolutional_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "convolutional_layers_v2",
    srcs = ["convolutional_layers_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":variational_base",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/experimental/nn/util",
        "//tensorflow_probability/python/internal:dtype_util",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "convolutional_layers_v2_test",
    size = "medium",
    srcs = ["convolutional_layers_v2_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "convolutional_transpose_layers",
    srcs = ["convolutional_transpose_layers.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        ":variational_base",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/experimental/nn/util:utils",
        "//tensorflow_probability/python/internal:prefer_static",
    ],
)

py_test(
    name = "convolutional_transpose_layers_test",
    size = "medium",
    srcs = ["convolutional_transpose_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "layers",
    srcs = ["layers.py"],
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability/python/experimental/nn/util:utils",
        "//tensorflow_probability/python/internal:name_util",
    ],
)

py_test(
    name = "layers_test",
    size = "small",
    srcs = ["layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "variational_base",
    srcs = ["variational_base.py"],
    srcs_version = "PY3",
    deps = [
        ":layers",
        # tensorflow dep,
        "//tensorflow_probability/python/distributions:distribution",
        "//tensorflow_probability/python/distributions:independent",
        "//tensorflow_probability/python/distributions:mvn_diag",
        "//tensorflow_probability/python/distributions:normal",
        "//tensorflow_probability/python/internal:prefer_static",
        "//tensorflow_probability/python/random",
        "//tensorflow_probability/python/util:seed_stream",
    ],
)
