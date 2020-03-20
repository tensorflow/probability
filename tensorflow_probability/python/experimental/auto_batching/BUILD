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
#   An auto-batching system that keeps track of an explicit program counter.

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

# [internal] load python3.bzl

py_library(
    name = "auto_batching",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":dsl",
        ":frontend",
        ":instructions",
        ":lowering",
        ":numpy_backend",
        ":stack_optimization",
        ":tf_backend",
        ":type_inference",
        ":virtual_machine",
        ":xla",
    ],
)

py_library(
    name = "instructions",
    srcs = ["instructions.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,  # For pywrap_tensorflow.IsNamedtuple
    ],
)

py_library(
    name = "numpy_backend",
    srcs = [
        "numpy_backend.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
        # numpy dep,
    ],
)

py_library(
    name = "tf_backend",
    srcs = [
        "tf_backend.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
        ":xla",
        # tensorflow dep,
    ],
)

py_library(
    name = "type_inference",
    srcs = [
        "type_inference.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
        # absl/logging dep,
    ],
)

py_library(
    name = "virtual_machine",
    srcs = [
        "virtual_machine.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
    ],
)

py_library(
    name = "liveness",
    srcs = [
        "liveness.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
    ],
)

py_library(
    name = "lowering",
    srcs = [
        "lowering.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
        ":liveness",
    ],
)

py_library(
    name = "stack_optimization",
    srcs = [
        "stack_optimization.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
    ],
)

py_library(
    name = "allocation_strategy",
    srcs = [
        "allocation_strategy.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
        ":liveness",
    ],
)

py_library(
    name = "stackless",
    srcs = [
        "stackless.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
    ],
)

py_library(
    name = "dsl",
    srcs = [
        "dsl.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
    ],
)

py_library(
    name = "frontend",
    srcs = [
        "frontend.py",
        "gast_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":dsl",
        ":instructions",
        ":lowering",
        ":stackless",
        ":stack_optimization",
        ":tf_backend",
        ":type_inference",
        ":virtual_machine",
        # For AutoGraph
        # tensorflow dep,
    ],
)

py_library(
    name = "backend_test_lib",
    srcs = [
        "backend_test_lib.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_library(
    name = "test_programs",
    srcs = ["test_programs.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":instructions",
        # numpy dep,
    ],
)

py_library(
    name = "xla",
    srcs = ["xla.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "virtual_machine_test_cpu",
    size = "medium",
    srcs = ["virtual_machine_test.py"],
    args = ["--test_device=cpu"],
    main = "virtual_machine_test.py",
    shard_count = 10,
    tags = [
        "nozapfhahn",
    ],
    deps = [
        "//tensorflow_probability/python/experimental/auto_batching:numpy_backend",
        "//tensorflow_probability/python/experimental/auto_batching:test_programs",
        "//tensorflow_probability/python/experimental/auto_batching:tf_backend",
        "//tensorflow_probability/python/experimental/auto_batching:virtual_machine",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_test(
    name = "virtual_machine_test_gpu",
    size = "medium",
    srcs = ["virtual_machine_test.py"],
    args = ["--test_device=gpu"],
    main = "virtual_machine_test.py",
    shard_count = 10,
    tags = [
        "nozapfhahn",
        "requires-gpu-nvidia",
    ],
    deps = [
        "//tensorflow_probability/python/experimental/auto_batching:numpy_backend",
        "//tensorflow_probability/python/experimental/auto_batching:test_programs",
        "//tensorflow_probability/python/experimental/auto_batching:tf_backend",
        "//tensorflow_probability/python/experimental/auto_batching:virtual_machine",
        "//tensorflow_probability/python/internal:test_util",
#         "//third_party/tensorflow/compiler/jit:xla_cpu_jit",  # DisableOnExport
    ],
)

py_test(
    name = "instructions_test",
    size = "small",
    srcs = ["instructions_test.py"],
    deps = [
        ":instructions",
        ":test_programs",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "numpy_backend_test",
    size = "small",
    srcs = ["numpy_backend_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend_test_lib",
        ":instructions",
        ":numpy_backend",
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "tf_backend_test",
    size = "large",
    srcs = ["tf_backend_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend_test_lib",
        ":instructions",
        ":tf_backend",
        # hypothesis dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:hypothesis_testlib",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "lowering_test",
    size = "small",
    srcs = ["lowering_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":lowering",
        ":numpy_backend",
        ":test_programs",
        ":virtual_machine",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "allocation_strategy_test",
    size = "small",
    srcs = ["allocation_strategy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":instructions",
        ":test_programs",
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "stackless_test",
    size = "small",
    srcs = ["stackless_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":numpy_backend",
        ":stackless",
        ":test_programs",
        ":tf_backend",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "stack_optimization_test",
    size = "small",
    srcs = ["stack_optimization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":stack_optimization",
        ":test_programs",
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "type_inference_test",
    size = "small",
    srcs = ["type_inference_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":lowering",
        ":numpy_backend",
        ":test_programs",
        ":tf_backend",
        ":type_inference",
        ":virtual_machine",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "dsl_test",
    size = "small",
    srcs = ["dsl_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":allocation_strategy",
        ":dsl",
        ":lowering",
        ":numpy_backend",
        ":tf_backend",
        ":type_inference",
        ":virtual_machine",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)

py_test(
    name = "frontend_test",
    size = "small",
    srcs = ["frontend_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":frontend",
        ":instructions",
        ":numpy_backend",
        ":tf_backend",
        ":type_inference",
        ":virtual_machine",
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability/python/internal:test_util",
    ],
)
