# Copyright 2018 The TensorFlow Probability Authors.
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
#   Numpy backend.

# [internal] load python3.bzl

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
        ":compat",
        ":control_flow",
        ":debugging",
        ":dtype",
        ":errors",
        ":functional_ops",
        ":keras",
        ":linalg",
        ":misc",
        ":nn",
        ":numpy_array",
        ":numpy_logging",
        ":numpy_math",
        ":numpy_signal",
        ":ops",
        ":random_generators",
        ":sets_lib",
        ":sparse_lib",
        ":test_lib",
        # tensorflow dep,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    deps = [
        ":_utils",
        ":v1",
        ":v2",
        # tensorflow dep,
    ],
)

py_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    deps = [
        ":_utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "debugging",
    srcs = ["debugging.py"],
    deps = [
        ":_utils",
        ":v1",
        ":v2",
        # tensorflow dep,
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["functional_ops.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        ":_utils",
        ":keras_layers",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "keras_layers",
    srcs = ["keras_layers.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "linalg",
    srcs = [
        "linalg.py",
        "linalg_impl.py",
    ],
    deps = [
        ":_utils",
        ":linear_operator",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "misc",
    srcs = ["misc.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "nn",
    srcs = ["nn.py"],
    deps = [
        ":_utils",
        ":numpy_math",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "numpy_array",
    srcs = ["numpy_array.py"],
    deps = [
        ":_utils",
        ":ops",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "numpy_logging",
    srcs = ["numpy_logging.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "numpy_math",
    srcs = ["numpy_math.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "numpy_signal",
    srcs = ["numpy_signal.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "random_generators",
    srcs = ["random_generators.py"],
    deps = [
        ":_utils",
        ":numpy_math",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "sets_lib",
    srcs = ["sets_lib.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "sparse_lib",
    srcs = ["sparse_lib.py"],
    deps = [
        ":_utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "test_lib",
    srcs = ["test_lib.py"],
    deps = [
        ":_utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "numpy_test",
    srcs = ["numpy_test.py"],
    shard_count = 5,
    tags = ["tfp_numpy"],
    deps = [
        ":numpy",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:test_case",
    ],
)

py_library(
    name = "v1",
    srcs = ["v1.py"],
    deps = [
        ":_utils",
        ":initializers",
        ":ops",
    ],
)

py_library(
    name = "v2",
    srcs = ["v2.py"],
    deps = [
        ":_utils",
        ":ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "_utils",
    srcs = ["_utils.py"],
    deps = [
        # decorator dep,
    ],
)

py_binary(
    name = "gen_linear_operators",
    srcs = ["gen_linear_operators.py"],
    python_version = "PY3",
    deps = [
        # absl:app dep,
        # tensorflow dep,
    ],
)

LINOP_FILES = [
    "adjoint_registrations",
    "cholesky_registrations",
    "inverse_registrations",
    "linear_operator_addition",
    "linear_operator_adjoint",
    "linear_operator_algebra",
    "linear_operator_block_diag",
    "linear_operator_circulant",
    "linear_operator_composition",
    "linear_operator_diag",
    "linear_operator_full_matrix",
    "linear_operator_householder",
    "linear_operator_identity",
    "linear_operator_inversion",
    "linear_operator_kronecker",
    "linear_operator_lower_triangular",
    "linear_operator_low_rank_update",
    "linear_operator",
    "linear_operator_toeplitz",
    "linear_operator_util",
    "linear_operator_zeros",
    "matmul_registrations",
    "registrations_util",
    "solve_registrations",
]

[genrule(
    name = "rewrite_{}".format(filename),
    srcs = [],
    outs = ["{}.py".format(filename)],
    cmd = "$(location :gen_linear_operators) --module_name={} > $@".format(filename),
    tools = [":gen_linear_operators"],
) for filename in LINOP_FILES]

py_library(
    name = "linear_operator",
    srcs = [":{}.py".format(filename) for filename in LINOP_FILES],
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
    ],
)
