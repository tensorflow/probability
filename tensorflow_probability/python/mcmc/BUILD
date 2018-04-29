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
#   MCMC methods, diagnostics, and related utilities.

package(
    default_visibility = [
        "//tensorflow_probability:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "mcmc",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":diagnostic",
        ":hmc",
        ":kernel",
        ":metropolis_hastings",
        ":random_walk_metropolis",
        ":replica_exchange_mc",
        ":sample",
        ":sample_annealed_importance",
        ":sample_halton_sequence",
    ],
)

py_library(
    name = "diagnostic",
    srcs = ["diagnostic.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "diagnostic_test",
    size = "small",
    srcs = ["diagnostic_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "hmc",
    srcs = ["hmc.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kernel",
        ":metropolis_hastings",
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "hmc_test",
    size = "large",
    srcs = ["hmc_test.py"],
    deps = [
        # numpy dep,
        # scipy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "kernel",
    srcs = ["kernel.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
    ],
)

py_library(
    name = "metropolis_hastings",
    srcs = ["metropolis_hastings.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kernel",
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "replica_exchange_mc",
    srcs = ["replica_exchange_mc.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kernel",
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "metropolis_hastings_test",
    size = "small",
    srcs = ["metropolis_hastings_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "random_walk_metropolis",
    srcs = ["random_walk_metropolis.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kernel",
        ":metropolis_hastings",
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "random_walk_metropolis_test",
    size = "small",
    srcs = ["random_walk_metropolis_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sample_test",
    size = "medium",
    srcs = ["sample_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "sample_annealed_importance",
    srcs = ["sample_annealed_importance.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":util",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sample_annealed_importance_test",
    size = "medium",
    srcs = ["sample_annealed_importance_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "sample_halton_sequence",
    srcs = ["sample_halton_sequence.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "sample_halton_sequence_test",
    size = "medium",
    srcs = ["sample_halton_sequence_test.py"],
    deps = [
        # numpy dep,
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
    deps = [
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    deps = [
        # tensorflow dep,
        "//tensorflow_probability",
    ],
)
