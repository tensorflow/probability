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
        ":bitwise",
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
        ":private",
        ":random_generators",
        ":raw_ops",
        ":sets_lib",
        ":sparse_lib",
        ":tensor_array_ops",
        ":test_lib",
        # tensorflow dep,
    ],
)

py_library(
    name = "bitwise",
    srcs = ["bitwise.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
    deps = [
        ":v1",
        ":v2",
    ],
)

py_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    deps = [
        ":_utils",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "debugging",
    srcs = ["debugging.py"],
    deps = [
        ":_utils",
        ":v1",
        ":v2",
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
    deps = [
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["functional_ops.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        ":_utils",
        ":keras_layers",
        # numpy dep,
    ],
)

py_library(
    name = "keras_layers",
    srcs = ["keras_layers.py"],
    deps = [
        ":_utils",
        # numpy dep,
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
    ],
)

py_library(
    name = "misc",
    srcs = ["misc.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "nn",
    srcs = ["nn.py"],
    deps = [
        ":_utils",
        ":numpy_array",
        ":numpy_math",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "numpy_array",
    srcs = ["numpy_array.py"],
    deps = [
        ":_utils",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "numpy_logging",
    srcs = ["numpy_logging.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "numpy_math",
    srcs = ["numpy_math.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "numpy_signal",
    srcs = ["numpy_signal.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        ":_utils",
        # numpy dep,
        # tensorflow dep,
        # wrapt dep,
    ],
)

py_library(
    name = "private",
    srcs = ["private.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "tensor_array_ops",
    srcs = ["tensor_array_ops.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["tensor_array_ops_test.py"],
    tags = ["tfp_numpy"],
    deps = [
        ":tensor_array_ops",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "random_generators",
    srcs = ["random_generators.py"],
    deps = [
        ":_utils",
        ":numpy_array",
        ":numpy_math",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "raw_ops",
    srcs = ["raw_ops.py"],
    deps = [
        ":_utils",
        ":numpy_array",
        ":numpy_math",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "sets_lib",
    srcs = ["sets_lib.py"],
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "sparse_lib",
    srcs = ["sparse_lib.py"],
    deps = [
        ":_utils",
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
    size = "small",
    srcs = ["numpy_test.py"],
    shard_count = 8,
    tags = ["tfp_numpy"],
    deps = [
        ":numpy",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # tensorflow dep,
        "//tensorflow_probability",
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "v1",
    srcs = ["v1.py"],
    deps = [
        ":_utils",
        ":initializers",
        ":ops",
        ":random_generators",
        ":tensor_array_ops",
    ],
)

py_library(
    name = "v2",
    srcs = ["v2.py"],
    deps = [
        ":_utils",
        ":ops",
        ":tensor_array_ops",
        # tensorflow dep,
    ],
)

py_library(
    name = "_utils",
    srcs = ["_utils.py"],
    deps = [
        # wrapt dep,
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
    "linear_operator_block_lower_triangular",
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
    cmd = ("$(location //tensorflow_probability/python/internal/backend/meta:gen_linear_operators) " +
           "--module_name={} --whitelist={} > $@").format(
        filename,
        ",".join(LINOP_FILES),
    ),
    tools = ["//tensorflow_probability/python/internal/backend/meta:gen_linear_operators"],
) for filename in LINOP_FILES]

py_library(
    name = "linear_operator",
    srcs = [":{}.py".format(filename) for filename in LINOP_FILES],
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
    ],
)

exports_files(
    [
        "__init__.py",
        "_utils.py",
        "bitwise.py",
        "compat.py",
        "control_flow.py",
        "debugging.py",
        "dtype.py",
        "errors.py",
        "functional_ops.py",
        "initializers.py",
        "keras.py",
        "keras_layers.py",
        "linalg.py",
        "linalg_impl.py",
        "misc.py",
        "nn.py",
        "numpy_array.py",
        "numpy_logging.py",
        "numpy_math.py",
        "numpy_signal.py",
        "numpy_test.py",
        "ops.py",
        "private.py",
        "random_generators.py",
        "raw_ops.py",
        "sets_lib.py",
        "sparse_lib.py",
        "tensor_array_ops.py",
        "tensor_array_ops_test.py",
        "test_lib.py",
        "v1.py",
        "v2.py",
    ],
    visibility = ["//tensorflow_probability:__subpackages__"],
)
