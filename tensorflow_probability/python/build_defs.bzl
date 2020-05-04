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
"""Build defs for TF/Numpy/JAX-variadic libraries & tests."""

# [internal] load python3.bzl

NO_REWRITE_NEEDED = [
    "internal:cache_util",
    "internal:docstring_util",
    "internal:name_util",
    "internal:reparameterization",
    "internal:tensorshape_util",
    "layers",
    "math:minimize",
    "math:sparse",
    "math/ode",
    "optimizer/convergence_criteria",
    "optimizer:sgld",
    "optimizer:variational_sgld",
    "platform_google",
]

REWRITER_TARGET = "//tensorflow_probability/python/experimental/substrates/meta:rewrite"

def _substrate_src(src, substrate):
    """Rewrite a single src filename for the given substrate."""
    return "_{}/{}".format(substrate, src)

def _substrate_srcs(srcs, substrate):
    """Rewrite src filenames for the given substrate."""
    return [_substrate_src(src, substrate) for src in srcs]

def _substrate_dep(dep, substrate):
    """Convert a single dep to one appropriate for the given substrate."""
    dep_to_check = dep
    if dep.startswith(":"):
        dep_to_check = "{}{}".format(native.package_name(), dep)
    for no_rewrite in NO_REWRITE_NEEDED:
        if no_rewrite in dep_to_check:
            return dep
    if dep_to_check.endswith("python/bijectors") or dep_to_check.endswith("python/bijectors:bijectors"):
        return "//tensorflow_probability/python/experimental/substrates/{}/bijectors".format(substrate)
    if dep_to_check.endswith("python/distributions") or dep_to_check.endswith("python/distributions:distributions"):
        return "//tensorflow_probability/python/experimental/substrates/{}/distributions".format(substrate)
    if "tensorflow_probability/" in dep or dep.startswith(":"):
        if "internal/backend" in dep:
            return dep
        if ":" in dep:
            return "{}.{}".format(dep, substrate)
        return "{}:{}.{}".format(dep, dep.split("/")[-1], substrate)
    return dep

def _substrate_deps(deps, substrate):
    """Convert deps to those appropriate for the given substrate."""
    new_deps = [_substrate_dep(dep, substrate) for dep in deps]
    backend_dep = "//tensorflow_probability/python/internal/backend/{}".format(substrate)
    if backend_dep not in new_deps:
        new_deps.append(backend_dep)
    return new_deps

# This is needed for the transitional period during which we have the internal
# py2and3_test and py_test comingling in BUILD files. Otherwise the OSS export
# rewrite process becomes irreversible.
def py3_test(*args, **kwargs):
    native.py_test(*args, **kwargs)

def multi_substrate_py_library(
        name,
        srcs = [],
        deps = [],
        substrates_omit_deps = [],
        testonly = 0,
        srcs_version = "PY2AND3"):
    """A TFP `py_library` for each of TF, Numpy, and JAX.

    Args:
        name: The TF `py_library` name. Numpy and JAX libraries have '.numpy' and
            '.jax' appended.
        srcs: As with `py_library`. A `genrule` is used to rewrite srcs for Numpy
            and JAX substrates.
        deps: As with `py_library`. The list is rewritten to depend on
            substrate-specific libraries for substrate variants.
        substrates_omit_deps: List of deps to omit if those libraries are not
            rewritten for the substrates.
        testonly: As with `py_library`.
        srcs_version: As with `py_library`.
    """

    native.py_library(
        name = name,
        srcs = srcs,
        deps = deps,
        srcs_version = srcs_version,
        testonly = testonly,
    )

    trimmed_deps = [dep for dep in deps if dep not in substrates_omit_deps]
    for src in srcs:
        native.genrule(
            name = "rewrite_{}_numpy".format(src.replace(".", "_")),
            srcs = [src],
            outs = [_substrate_src(src, "numpy")],
            cmd = "$(location {}) $(SRCS) > $@".format(REWRITER_TARGET),
            tools = [REWRITER_TARGET],
        )
    native.py_library(
        name = "{}.numpy".format(name),
        srcs = _substrate_srcs(srcs, "numpy"),
        deps = _substrate_deps(trimmed_deps, "numpy"),
        srcs_version = srcs_version,
        testonly = testonly,
    )

    jax_srcs = _substrate_srcs(srcs, "jax")
    for src in srcs:
        native.genrule(
            name = "rewrite_{}_jax".format(src.replace(".", "_")),
            srcs = [src],
            outs = [_substrate_src(src, "jax")],
            cmd = "$(location {}) $(SRCS) --numpy_to_jax > $@".format(REWRITER_TARGET),
            tools = [REWRITER_TARGET],
        )
    native.py_library(
        name = "{}.jax".format(name),
        srcs = jax_srcs,
        deps = _substrate_deps(trimmed_deps, "jax"),
        srcs_version = srcs_version,
        testonly = testonly,
    )

def multi_substrate_py_test(
        name,
        size = "small",
        jax_size = None,
        srcs = [],
        deps = [],
        tags = [],
        numpy_tags = [],
        jax_tags = [],
        srcs_version = "PY2AND3",
        timeout = None,
        shard_count = None):
    """A TFP `py2and3_test` for each of TF, Numpy, and JAX.

    Args:
        name: Name of the `test_suite` which covers TF, Numpy and JAX variants
            of the test. Each substrate will have a dedicated `py2and3_test`
            suffixed with '.tf', '.numpy', or '.jax' as appropriate.
        size: As with `py_test`.
        jax_size: A size override for the JAX target.
        srcs: As with `py_test`. These will have a `genrule` emitted to rewrite
            Numpy and JAX variants, writing the test file into a subdirectory.
        deps: As with `py_test`. The list is rewritten to depend on
            substrate-specific libraries for substrate variants.
        tags: Tags global to this test target. Numpy also gets a `'tfp_numpy'`
            tag, and JAX gets a `'tfp_jax'` tag. A `f'_{name}'` tag is used
            to produce the `test_suite`.
        numpy_tags: Tags specific to the Numpy test. (e.g. `'notap'`).
        jax_tags: Tags specific to the JAX test. (e.g. `'notap'`).
        srcs_version: As with `py_test`.
        timeout: As with `py_test`.
        shard_count: As with `py_test`.
    """

    name_tag = "_{}".format(name)
    tags = [t for t in tags]
    tags.append(name_tag)
    tags.append("multi_substrate")
    native.py_test(
        name = "{}.tf".format(name),
        size = size,
        srcs = srcs,
        main = "{}.py".format(name),
        deps = deps,
        tags = tags,
        srcs_version = srcs_version,
        timeout = timeout,
        shard_count = shard_count,
    )

    numpy_srcs = _substrate_srcs(srcs, "numpy")
    native.genrule(
        name = "rewrite_{}_numpy".format(name),
        srcs = srcs,
        outs = numpy_srcs,
        cmd = "$(location {}) $(SRCS) > $@".format(REWRITER_TARGET),
        tools = [REWRITER_TARGET],
    )
    py3_test(
        name = "{}.numpy".format(name),
        size = size,
        srcs = numpy_srcs,
        main = _substrate_src("{}.py".format(name), "numpy"),
        deps = _substrate_deps(deps, "numpy"),
        tags = tags + ["tfp_numpy"] + numpy_tags,
        srcs_version = srcs_version,
        python_version = "PY3",
        timeout = timeout,
        shard_count = shard_count,
    )

    jax_srcs = _substrate_srcs(srcs, "jax")
    native.genrule(
        name = "rewrite_{}_jax".format(name),
        srcs = srcs,
        outs = jax_srcs,
        cmd = "$(location {}) $(SRCS) --numpy_to_jax > $@".format(REWRITER_TARGET),
        tools = [REWRITER_TARGET],
    )
    jax_deps = _substrate_deps(deps, "jax")
    # [internal] Add JAX build dep
    py3_test(
        name = "{}.jax".format(name),
        size = jax_size if jax_size else size,
        srcs = jax_srcs,
        main = _substrate_src("{}.py".format(name), "jax"),
        deps = jax_deps,
        tags = tags + ["tfp_jax"] + jax_tags,
        srcs_version = srcs_version,
        python_version = "PY3",
        timeout = timeout,
        shard_count = shard_count,
    )

    native.test_suite(
        name = name,
        tags = [name_tag],
    )
