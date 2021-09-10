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
"""Build defs for TF/NumPy/JAX-variadic libraries & tests."""

NO_REWRITE_NEEDED = [
    "internal:all_util",
    "internal:docstring_util",
    "internal:reparameterization",
    "python/layers",
    "platform_google",
]

REWRITER_TARGET = "//tensorflow_probability/substrates/meta:rewrite"

RUNFILES_ROOT = "tensorflow_probability/"

def _substrate_src(src, substrate):
    """Rewrite a single src filename for the given substrate."""

    # When src's are sourced from a different package we cut away the package
    # name.
    parts = src.split(":")
    return "_{}/_generated_{}".format(substrate, parts[-1])

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

def _resolve_omit_dep(dep):
    """Resolves a `substrates_omit_deps` item to full target."""
    if ":" not in dep:
        dep = "{}:{}".format(dep, dep.split("/")[-1])
    if dep.startswith(":"):
        dep = "{}{}".format(native.package_name(), dep)
    return dep

def _substrate_runfiles_symlinks_impl(ctx):
    """A custom BUILD rule to generate python runfiles symlinks.

    A custom build rule which adds runfiles symlinks for files matching a
    substrate genrule file pattern, i.e. `'_jax/_generated_normal.py'`.

    This rule will aggregate and pass along deps while adding the given
    symlinks to the runfiles structure.

    Build rule attributes:
    - substrate: One of 'jax' or 'numpy'; which substrate this applies to.
    - deps: A list of py_library labels. These are passed along.

    Args:
        ctx: Rule analysis context.

    Returns:
        Info objects to propagate deps and add runfiles symlinks.
    """

    # Aggregate the depset inputs to resolve transitive dependencies.
    transitive_sources = []
    uses_shared_libraries = []
    imports = []
    has_py2_only_sources = []
    has_py3_only_sources = []
    cc_infos = []
    for dep in ctx.attr.deps:
        if PyInfo in dep:
            transitive_sources.append(dep[PyInfo].transitive_sources)
            uses_shared_libraries.append(dep[PyInfo].uses_shared_libraries)
            imports.append(dep[PyInfo].imports)
            has_py2_only_sources.append(dep[PyInfo].has_py2_only_sources)
            has_py3_only_sources.append(dep[PyInfo].has_py3_only_sources)

#         if PyCcLinkParamsProvider in dep:  # DisableOnExport
#             cc_infos.append(dep[PyCcLinkParamsProvider].cc_info)  # DisableOnExport

        if CcInfo in dep:
            cc_infos.append(dep[CcInfo])

    # Determine the set of symlinks to generate.
    transitive_sources = depset(transitive = transitive_sources)
    runfiles_dict = {}
    substrate = ctx.attr.substrate
    file_substr = "_{}/_generated_".format(substrate)
    for f in transitive_sources.to_list():
        if "tensorflow_probability/python/" in f.dirname and file_substr in f.short_path:
            pre, post = f.short_path.split("/python/")
            out_path = "{}/substrates/{}/{}".format(
                pre,
                substrate,
                post.replace(file_substr, ""),
            )
            runfiles_dict[RUNFILES_ROOT + out_path] = f
        elif "tensorflow_probability/google" in f.dirname and file_substr in f.short_path:
            pre, post = f.short_path.split("/google/")
            out_path = "{}/substrates/{}/google/{}".format(
                pre,
                substrate,
                post.replace(file_substr, ""),
            )
            runfiles_dict[RUNFILES_ROOT + out_path] = f

    # Construct the output structures to pass along Python srcs/deps/etc.
    py_info = PyInfo(
        transitive_sources = transitive_sources,
        uses_shared_libraries = any(uses_shared_libraries),
        imports = depset(transitive = imports),
        has_py2_only_sources = any(has_py2_only_sources),
        has_py3_only_sources = any(has_py3_only_sources),
    )

    py_cc_link_info = cc_common.merge_cc_infos(cc_infos = cc_infos)

    py_runfiles = depset(
        transitive = [depset(transitive = [
            dep[DefaultInfo].data_runfiles.files,
            dep[DefaultInfo].default_runfiles.files,
        ]) for dep in ctx.attr.deps],
    )

    runfiles = DefaultInfo(runfiles = ctx.runfiles(
        transitive_files = py_runfiles,
        root_symlinks = runfiles_dict,
    ))

    return py_info, py_cc_link_info, runfiles

# See documentation at:
# https://docs.bazel.build/versions/3.4.0/skylark/rules.html
substrate_runfiles_symlinks = rule(
    implementation = _substrate_runfiles_symlinks_impl,
    attrs = {
        "substrate": attr.string(),
        "deps": attr.label_list(),
    },
)

def multi_substrate_py_library(
        name,
        srcs = [],
        deps = [],
        substrates_omit_deps = [],
        jax_omit_deps = [],
        numpy_omit_deps = [],
        srcs_version = "PY3",
        **kwargs):
    """A TFP `py_library` for each of TF, NumPy, and JAX.

    Args:
        name: The TF `py_library` name. NumPy and JAX libraries have '.numpy' and
            '.jax' appended.
        srcs: As with `py_library`. A `genrule` is used to rewrite srcs for NumPy
            and JAX substrates.
        deps: As with `py_library`. The list is rewritten to depend on
            substrate-specific libraries for substrate variants.
        substrates_omit_deps: List of deps to omit if those libraries are not
            rewritten for the substrates.
        jax_omit_deps: List of deps to omit for the JAX substrate.
        numpy_omit_deps: List of deps to omit for the NumPy substrate.
        srcs_version: As with `py_library`.
        **kwargs: Additional keyword arguments for targets.
    """

    if srcs_version != "PY3":
        fail("Must use PY3 for srcs_version", srcs_version)

    native.py_library(
        name = name,
        srcs = srcs,
        deps = deps,
        srcs_version = srcs_version,
        **kwargs
    )
    remove_deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow",
    ]

    trimmed_deps = [dep for dep in deps if (dep not in substrates_omit_deps and
                                            dep not in remove_deps)]
    resolved_omit_deps_numpy = [
        _resolve_omit_dep(dep)
        for dep in substrates_omit_deps + numpy_omit_deps
    ]
    for src in srcs:
        native.genrule(
            name = "rewrite_{}_numpy".format(src.replace(".", "_")),
            srcs = [src],
            outs = [_substrate_src(src, "numpy")],
            cmd = "$(location {}) $(SRCS) --omit_deps={} > $@".format(
                REWRITER_TARGET,
                ",".join(resolved_omit_deps_numpy),
            ),
            exec_tools = [REWRITER_TARGET],
        )
    native.py_library(
        name = "{}.numpy.raw".format(name),
        srcs = _substrate_srcs(srcs, "numpy"),
        deps = _substrate_deps(trimmed_deps, "numpy"),
        srcs_version = srcs_version,
        **kwargs
    )

    # Add symlinks under tfp/substrates/numpy.
    substrate_runfiles_symlinks(
        name = "{}.numpy".format(name),
        substrate = "numpy",
        deps = [":{}.numpy.raw".format(name)],
        **kwargs
    )

    resolved_omit_deps_jax = [
        _resolve_omit_dep(dep)
        for dep in substrates_omit_deps + jax_omit_deps
    ]
    jax_srcs = _substrate_srcs(srcs, "jax")
    for src in srcs:
        native.genrule(
            name = "rewrite_{}_jax".format(src.replace(".", "_")),
            srcs = [src],
            outs = [_substrate_src(src, "jax")],
            cmd = "$(location {}) $(SRCS) --omit_deps={} --numpy_to_jax > $@".format(
                REWRITER_TARGET,
                ",".join(resolved_omit_deps_jax),
            ),
            exec_tools = [REWRITER_TARGET],
        )
    native.py_library(
        name = "{}.jax.raw".format(name),
        srcs = jax_srcs,
        deps = _substrate_deps(trimmed_deps, "jax"),
        srcs_version = srcs_version,
        **kwargs
    )

    # Add symlinks under tfp/substrates/jax.
    substrate_runfiles_symlinks(
        name = "{}.jax".format(name),
        substrate = "jax",
        deps = [":{}.jax.raw".format(name)],
        **kwargs
    )

def multi_substrate_py_test(
        name,
        size = "small",
        jax_size = None,
        numpy_size = None,
        srcs = [],
        main = None,
        deps = [],
        jax_extra_deps = [],
        tags = [],
        numpy_tags = [],
        jax_tags = [],
        tf_tags = [],
        disabled_substrates = [],
        srcs_version = "PY3",
        python_version = "PY3",
        timeout = None,
        shard_count = None,
        args = []):
    """A TFP `py_test` for each of TF, NumPy, and JAX.

    Args:
        name: Name of the `test_suite` which covers TF, NumPy and JAX variants
            of the test. Each substrate will have a dedicated `py_test`
            suffixed with '.tf', '.numpy', or '.jax' as appropriate.
        size: As with `py_test`.
        jax_size: A size override for the JAX target.
        numpy_size: A size override for the numpy target.
        srcs: As with `py_test`. These will have a `genrule` emitted to rewrite
            NumPy and JAX variants, writing the test file into a subdirectory.
        main: As with `py_test`. If this does not match "{name}.py", then we
            suppress the genrule that rewrites "{name}.py", since the typical
            use-case of the `main` argument is a secondary, i.e. GPU, test.
        deps: As with `py_test`. The list is rewritten to depend on
            substrate-specific libraries for substrate variants.
        jax_extra_deps: Extra dependencies for the JAX substrate.
        tags: Tags global to this test target. NumPy also gets a `'tfp_numpy'`
            tag, and JAX gets a `'tfp_jax'` tag. A `f'_{name}'` tag is used
            to produce the `test_suite`.
        numpy_tags: Tags specific to the NumPy test. (e.g. `"notap"`).
        jax_tags: Tags specific to the JAX test. (e.g. `"notap"`).
        tf_tags: Tags specific to the TF test. (e.g. `"notap"`).
        disabled_substrates: Iterable of substrates to disable, items from
            ["numpy", "jax"].
        srcs_version: As with `py_test`.
        python_version: As with `py_test`.
        timeout: As with `py_test`.
        shard_count: As with `py_test`.
        args: As with `py_test`.
    """

    tags = [t for t in tags]
    tags.append("multi_substrate")

    test_targets = []
    native.py_test(
        name = "{}.tf".format(name),
        size = size,
        srcs = srcs,
        main = main or "{}.py".format(name),
        deps = deps,
        tags = tags + tf_tags,
        srcs_version = srcs_version,
        python_version = python_version,
        timeout = timeout,
        shard_count = shard_count,
        args = args,
    )
    test_targets.append(":{}.tf".format(name))

    if "numpy" not in disabled_substrates:
        numpy_srcs = _substrate_srcs(srcs, "numpy")
        if main == None or main == "{}.py".format(name):
            native.genrule(
                name = "rewrite_{}_numpy".format(name),
                srcs = srcs,
                outs = numpy_srcs,
                cmd = "$(location {}) $(SRCS) > $@".format(REWRITER_TARGET),
                exec_tools = [REWRITER_TARGET],
            )
        native.py_test(
            name = "{}.numpy".format(name),
            size = numpy_size or size,
            srcs = numpy_srcs,
            main = _substrate_src(main or "{}.py".format(name), "numpy"),
            deps = _substrate_deps(deps, "numpy"),
            tags = tags + ["tfp_numpy"] + numpy_tags,
            srcs_version = srcs_version,
            python_version = "PY3",
            timeout = timeout,
            shard_count = shard_count,
            args = args,
        )
        test_targets.append(":{}.numpy".format(name))

    if "jax" not in disabled_substrates:
        jax_srcs = _substrate_srcs(srcs, "jax")
        if main == None or main == "{}.py".format(name):
            native.genrule(
                name = "rewrite_{}_jax".format(name),
                srcs = srcs,
                outs = jax_srcs,
                cmd = "$(location {}) $(SRCS) --numpy_to_jax > $@".format(REWRITER_TARGET),
                exec_tools = [REWRITER_TARGET],
            )
        jax_deps = _substrate_deps(deps, "jax") + jax_extra_deps
        # [internal] Add JAX build dep
        native.py_test(
            name = "{}.jax".format(name),
            size = jax_size or size,
            srcs = jax_srcs,
            main = _substrate_src(main or "{}.py".format(name), "jax"),
            deps = jax_deps,
            tags = tags + ["tfp_jax"] + jax_tags,
            srcs_version = srcs_version,
            python_version = "PY3",
            timeout = timeout,
            shard_count = shard_count,
            args = args,
        )
        test_targets.append(":{}.jax".format(name))

    native.test_suite(
        name = name,
        tests = test_targets,
    )
