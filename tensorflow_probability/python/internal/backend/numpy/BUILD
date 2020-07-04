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
        ":deprecation",
        ":dtype",
        ":errors",
        ":functional_ops",
        ":keras",
        ":linalg",
        ":misc",
        ":nest",
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
        ":static_rewrites",
        ":tensor_array_ops",
        ":test_lib",
        ":tf_inspect",
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
        ":dtype",
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
        # six dep,
    ],
)

py_library(
    name = "deprecation",
    srcs = ["deprecation.py"],
    deps = [],
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
        ":static_rewrites",
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
    name = "nest",
    srcs = ["nest.py"],
    deps = [
        # tree dep,
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
        # six dep,
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

# py2and3
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

genrule(
    name = "rewrite_tensor_shape",
    srcs = [],
    outs = ["tensor_shape_gen.py"],
    cmd = ("$(location //tensorflow_probability/python/internal/backend/meta:gen_tensor_shape) > $@"),
    tools = ["//tensorflow_probability/python/internal/backend/meta:gen_tensor_shape"],
)

py_library(
    name = "tensor_shape_gen",
    testonly = 1,
    srcs = ["tensor_shape_gen.py"],
    deps = [
        # six dep,
    ],
)

py_library(
    name = "test_lib",
    srcs = ["test_lib.py"],
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        # numpy dep,
    ],
)

py_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    deps = [],
)

# py2and3
py_test(
    name = "numpy_test",
    size = "small",
    srcs = ["numpy_test.py"],
    shard_count = 11,
    tags = ["tfp_numpy"],
    deps = [
        ":numpy",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # six dep,
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
        ":nest",
        ":ops",
        ":tensor_array_ops",
    ],
)

py_library(
    name = "_utils",
    srcs = ["_utils.py"],
    deps = [
        ":nest",
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
    testonly = 1,
    srcs = [],
    outs = ["{}_gen.py".format(filename)],
    cmd = ("$(location //tensorflow_probability/python/internal/backend/meta:gen_linear_operators) " +
           "--module_name={} --allowlist={} > $@").format(
        filename,
        ",".join(LINOP_FILES),
    ),
    tools = ["//tensorflow_probability/python/internal/backend/meta:gen_linear_operators"],
) for filename in LINOP_FILES]

# Rules helpful for generating new rewritten files.
[genrule(
    name = "generate_{}".format(filename),
    testonly = 1,
    srcs = [],
    outs = ["gen_new/{}.py".format(filename)],
    cmd = ("$(location //tensorflow_probability/python/internal/backend/meta:gen_linear_operators) " +
           "--module_name={} --allowlist={} > $@").format(
        filename,
        ",".join(LINOP_FILES),
    ),
    tools = ["//tensorflow_probability/python/internal/backend/meta:gen_linear_operators"],
) for filename in LINOP_FILES]

py_library(
    name = "generated_files",
    testonly = 1,
    srcs = ["gen_new/{}.py".format(filename) for filename in LINOP_FILES],
    srcs_version = "PY2AND3",
)

py_library(
    name = "linear_operator_gen",
    testonly = 1,
    srcs = [":{}_gen.py".format(filename) for filename in LINOP_FILES],
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
    ],
)

# py2and3
py_test(
    name = "rewrite_equivalence_test",
    srcs = ["rewrite_equivalence_test.py"],
    args = [
        "--modules_to_check",
        ",".join(LINOP_FILES + ["tensor_shape"]),
    ],
    srcs_version = "PY2AND3",
    tags = ["no-oss-ci"],
    deps = [
        ":linear_operator_gen",
        ":static_rewrites",
        ":tensor_shape_gen",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_library(
    name = "static_rewrites",
    srcs = glob(["gen/*.py"]),
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
        # six dep,
    ],
)

exports_files(
    glob(["**/*.py"]),
    visibility = ["//tensorflow_probability:__subpackages__"],
)
