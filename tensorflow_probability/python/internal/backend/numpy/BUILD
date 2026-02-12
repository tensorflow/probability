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

# Placeholder: py_library
# Placeholder: py_test

licenses(["notice"])

package(
    # default_applicable_licenses
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

py_library(
    name = "numpy",
    srcs = ["__init__.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":__internal__",
        ":bitwise",
        ":compat",
        ":composite_tensor",
        ":composite_tensor_gradient",
        ":config",
        ":control_flow",
        ":data_structures",
        ":debugging",
        ":deprecation",
        ":dtype",
        ":errors",
        ":functional_ops",
        ":linalg",
        ":misc",
        ":nest",
        ":nn",
        ":numpy_array",
        ":numpy_keras",
        ":numpy_logging",
        ":numpy_math",
        ":numpy_signal",
        ":ops",
        ":private",
        ":random_generators",
        ":raw_ops",
        ":resource_variable_ops",
        ":sets_lib",
        ":sparse_lib",
        ":static_rewrites",
        ":tensor_array_ops",
        ":tensor_spec",
        ":test_lib",
        ":tf_inspect",
        ":type_spec",
        ":type_spec_registry",
        ":variable_utils",
        ":variables",
    ],
)

py_library(
    name = "__internal__",
    srcs = ["__internal__.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":composite_tensor",
    ],
)

py_library(
    name = "bitwise",
    srcs = ["bitwise.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "compat",
    srcs = ["compat.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":v1",
        ":v2",
    ],
)

py_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "composite_tensor_gradient",
    srcs = ["composite_tensor_gradient.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "config",
    srcs = ["config.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        ":dtype",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [],
)

py_library(
    name = "debugging",
    srcs = ["debugging.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        ":dtype",
        ":v1",
        ":v2",
        # six dep,
    ],
)

py_library(
    name = "deprecation",
    srcs = ["deprecation.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "errors",
    srcs = ["errors.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
    ],
)

py_library(
    name = "functional_ops",
    srcs = ["functional_ops.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "initializers",
    srcs = ["initializers.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "numpy_keras",
    srcs = ["numpy_keras.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        ":keras_layers",
        # numpy dep,
    ],
)

py_library(
    name = "keras_layers",
    srcs = ["keras_layers.py"],
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "nest",
    srcs = ["nest.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        # tree dep,
        # wrapt dep,
    ],
)

py_library(
    name = "nn",
    srcs = ["nn.py"],
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "numpy_logging",
    srcs = ["numpy_logging.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "numpy_math",
    srcs = ["numpy_math.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
    ],
)

py_library(
    name = "numpy_signal",
    srcs = ["numpy_signal.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "tensor_array_ops",
    srcs = ["tensor_array_ops.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_test(
    name = "tensor_array_ops_test",
    size = "small",
    srcs = ["tensor_array_ops_test.py"],
#     strict_deps = False,  # DisableOnExport
    tags = ["tfp_numpy"],
    deps = [
        ":tensor_array_ops",
        "//tensorflow_probability/python/internal:test_util.numpy",
    ],
)

py_library(
    name = "random_generators",
    srcs = ["random_generators.py"],
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        ":numpy_array",
        ":numpy_math",
        ":ops",
        # numpy dep,
    ],
)

py_library(
    name = "resource_variable_ops",
    srcs = ["resource_variable_ops.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "sets_lib",
    srcs = ["sets_lib.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
    ],
)

py_library(
    name = "sparse_lib",
    srcs = ["sparse_lib.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
    ],
)

genrule(
    name = "rewrite_tensor_shape",
    srcs = [],
    outs = ["tensor_shape_gen.py"],
    cmd = "$(location //tensorflow_probability/python/internal/backend/meta:gen_tensor_shape) > $@",
    tools = ["//tensorflow_probability/python/internal/backend/meta:gen_tensor_shape"],
)

py_library(
    name = "tensor_shape_gen",
    testonly = 1,
    srcs = ["tensor_shape_gen.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        # absl/logging dep,
        # six dep,
    ],
)

py_library(
    name = "test_lib",
    srcs = ["test_lib.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [],
)

py_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [],
)

py_library(
    name = "tensor_spec",
    srcs = ["tensor_spec.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "type_spec",
    srcs = ["type_spec.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "type_spec_registry",
    srcs = ["type_spec_registry.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "variable_utils",
    srcs = ["variable_utils.py"],
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "variables",
    srcs = ["variables.py"],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":ops",
    ],
)

py_library(
    name = "numpy_testlib",
    testonly = 1,
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":numpy",
        # absl/testing:parameterized dep,
        # hypothesis dep,
        # six dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "numpy_test",
    size = "small",
    srcs = ["numpy_test.py"],
    shard_count = 15,
#     strict_deps = False,  # DisableOnExport
    tags = ["tf1-broken"],
    deps = [
        ":numpy_testlib",
        # tensorflow dep,
    ],
)

py_test(
    name = "xla_test_cpu",
    size = "medium",
    srcs = ["numpy_test.py"],
    args = ["--test_mode=xla"],
    main = "numpy_test.py",
    shard_count = 15,
#     strict_deps = False,  # DisableOnExport
    tags = [
        "hypothesis",
        "no-oss-ci",
        "tfp_xla",
    ],
    deps = [
        ":numpy_testlib",
        # tensorflow/compiler/jit dep,
    ],
)

py_test(
    name = "xla_test_gpu",
    size = "small",
    srcs = ["numpy_test.py"],
    args = [
        "--test_mode=xla",
        # TODO(b/168718272): reduce_*([nan, nan], axis=0) (GPU)
        # histogram_fixed_width_bins fails with f32([0.]), [0.0, 0.0], 2
        ("--xla_disabled=math.cumulative_logsumexp,math.reduce_min,math.reduce_max,histogram_fixed_width_bins," +
         # TODO(b/298426124): TF floomod GPU bug
         "math.floormod"),
    ],
    main = "numpy_test.py",
    shard_count = 11,
#     strict_deps = False,  # DisableOnExport
    tags = [
        "no-oss-ci",
        "requires-gpu-nvidia",
        "tfp_xla",
    ],
    deps = [
        ":numpy_testlib",
        # tensorflow/compiler/jit dep,
    ],
)

test_suite(
    name = "xla_test",
    tests = [
        ":xla_test_cpu",
        ":xla_test_gpu",
    ],
)

py_library(
    name = "v1",
    srcs = ["v1.py"],
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
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
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":nest",
        # wrapt dep,
    ],
)

LINOP_FILES = [
    "linear_operator_addition",
    "linear_operator_adjoint",
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
    "linear_operator_permutation",
    "linear_operator",
    "linear_operator_toeplitz",
    "linear_operator_util",
    "linear_operator_zeros",
    "property_hint_util",
    "slicing",
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
#     strict_deps = False,  # DisableOnExport
)

py_library(
    name = "linear_operator_gen",
    testonly = 1,
    srcs = [":{}_gen.py".format(filename) for filename in LINOP_FILES],
#     strict_deps = False,  # DisableOnExport
    deps = [
        ":_utils",
        # numpy dep,
        # scipy dep,
    ],
)

py_test(
    name = "rewrite_equivalence_test",
    srcs = ["rewrite_equivalence_test.py"],
    args = [
        "--modules_to_check",
        ",".join(LINOP_FILES + ["tensor_shape"]),
    ],
#     strict_deps = False,  # DisableOnExport
    tags = [
        "no-oss-ci",
        "notap",
        "tfp-guitar",
    ],
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
#     strict_deps = False,  # DisableOnExport
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
