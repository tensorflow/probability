# TensorFlow Probability Style Guide

This page contains style decisions that both developers and users of TensorFlow
Probability should follow to increase the readability of their code, reduce the
number of errors, and promote consistency.

## TensorFlow Style

Follow the [TensorFlow style
guide](https://www.tensorflow.org/community/style_guide) and [documentation
guide](https://www.tensorflow.org/community/documentation), and in particular
the [TensorFlow code
lint](https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md#python-coding-style).
Below are additional TensorFlow conventions not noted in those guides. (In the
future, these noted conventions may be moved upstream.)

1.  The name is TensorFlow, not Tensorflow.
1.  Use `name_scope` at the beginning of every Python function.

    Justification: it’s easier to debug TF graphs when they align with Python
    code.

1.  Run all Tensor args through `tf.convert_to_tensor` immediately after
    name_scope.

    Justification: not doing so can lead to surprising results when computing
    gradients. It can also lead to unnecessary graph ops as subsequent TF calls
    will keep creating a tensor from the same op.

1.  Every module should define the constant `__all__` in order to list all
    public members of the module.

    Justification: `__all__` is an explicit enumeration of what's intended to be
    public. It also governs what's imported when using `from foo import *`
    (although we do not use star-import w/in Google, users may.)

1.  Use ticks for any Python objects, types, or code. E.g., write \`Tensor\`
    instead of Tensor.

## TensorFlow Probability Style

Below are TensorFlow Probability-specific conventions. In the event of conflict,
they supersede all previous conventions.

1.  __Importing submodule aliases.__ Use the Pythonic style `from
    tensorflow_probability import sts`. For now, do not use this style
    for `tfd`, `tfb`, and `tfe`; use variable assignment via `tfd =
    tfp.distributions`. We will change the latter to use the Pythonic style in
    the future.

1.  __Examples in Docstrings.__ Write a `#### Examples` subsection below `Args`,
    `Returns`, `Raises`, etc. to illustrate examples. If the docstring's last
    line is a fence bracket (\`\`\`) closing a code snippet, add an empty line
    before closing the docstring with \"\"\". This properly displays the code
    snippet.

    Justification: Users regularly need to remind themselves of args and
    semantics. But rarely look at examples more than the first time. But since
    examples are usually long (which is great!) it means they have to do a lot
    of annoying scrolling ...unless Examples follow Args/Returns/Raises.

1.  __Citations in Docstrings.__ Write a `#### References` subsection at the
    bottom of any docstring with citations. Use ICLR’s bibliography style to
    write references; for example, order entries by the first author's last
    name. Add a link to the paper if the publication is open source (ideally,
    arXiv).

    Write in-paragraph citations in general, e.g., [(Tran and Blei, 2018)][1].
    Write in-text citations when the citation is a noun, e.g., [Tran and Blei
    (2018)][1]. Write citations with more than two authors using et al., e.g.,
    [(Tran et al., 2018)][1]. Separate multiple citations with semicolon, e.g.,
    ([Tran and Blei, 2018][1]; [Gelman and Rubin, 1992][2]).

    Examples:

    ```none
    #### References

    # technical report
    [1]: Tony Finch. Incremental calculation of weighted mean and variance.
         _Technical Report_, 2009.
         http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

    # journal
    [2]: Andrew Gelman and Donald B. Rubin. Inference from Iterative Simulation
         Using Multiple Sequences. _Statistical Science_, 7(4):457-472, 1992.

    # arXiv preprint
    # use "et al." for papers with too many authors to maintain
    [3]: Aaron van den Oord et al. Parallel WaveNet: Fast High-Fidelity Speech
         Synthesis. _arXiv preprint arXiv:1711.10433_, 2017.
         https://arxiv.org/abs/1711.10433

    # conference
    [4]: Yeming Wen, Paul Vicol, Jimmy Ba, Dustin Tran, and Roger Grosse.
         Flipout: Efficient Pseudo-Independent Weight Perturbations on
         Mini-Batches. In _International Conference on Learning
         Representations_, 2018.
         https://arxiv.org/abs/1803.04386
    ```

1.  When doing float math over literals eg use `1.` instead of `1` or `1.0`.

    *   Using `1.` is another line of defense against an automatic casting
        mistake. (Using `1.0` is also such a defense but is not minimal.)

1.  Prefer using named args for functions' 2nd args onward.

    *   Definitely use named args for 2nd args onward in docstrings.

1.  Use names which describe semantics, not computation or mathematics, e.g.,
    avoid `xp1 = x+1` or `tfd.Normal(loc=mu, scale=sigma)`.

1.  Prefer inlining intermediates which are used once.

    *   For intermediates, usually the actual code is better documentation than
        a variable. However if the intermediate math is not self-documenting,
        using an intermediate variable is ok--just ensure it has a great name!

    Justification: intermediates clutter scope and make it hard to see
    dependencies.

1.  Use literals, not `tf.constants`. Never use tf.constant in the API
    (user-side code is ok!). E.g., don't do: `two_pi = tf.constant(2. * np.pi)`.

    *   While using `tf.constant` may reduce graph size, it makes for
        substantially harder to read code.
    *   It also means you lose the benefit of automatic dtype casting (which is
        done only for literals).

1.  Avoid LaTeX in docstrings.

    *   It is not rendered in many (if not most) editors and can be hard to read
        for both LaTeX experts and non-experts.

1.  Write docstring and comment math using ASCII friendly notation; python using
    operators. E.g., `x**2` better than `x^2`, `x[i, j]` better than `x_{i,j}`,
    `sum{ f(x[i]) : i=1...n }` better than `\sum_{i=1}^n f(x_i)` `int{sin(x) dx:
    x in [0, 2 pi]}` better than `\int_0^{2\pi} sin(x) dx`.

    *   The more we stick to python style, the more someone can
        copy/paste/execute.
    *   Python style is usually easier to read as ASCII.

1.  All public functions require docstrings with: one line description, Args,
    Returns, Raises (if raises exceptions).

    *   Returns docstrings should be in the same format as Args, eg, of the form
        "name: Description." Part of the rationale is that we are suggesting a
        reasonable variable name for the returned object(s).

1.  Regard `*args` and/or `**kwargs` as features of last resort.

    *   Keyword arguments make the intention of a function call more clear.
    *   [Possible exceptions for
        `kwargs`](https://stackoverflow.com/questions/1415812/why-use-kwargs-in-python-what-are-some-real-world-advantages-over-using-named).

1.  Prefer using the most specific TF operator. E.g,

    *   Use `tf.squared_difference(x,y)` over `(x-y)**2`.
    *   Use `tf.rsqrt` over `1./tf.sqrt(x)`.

1.  Worry about gradients! (It's often not automatic for API builders!)

1.  When forced to choose between FLOPS and numerical accuracy, prefer numerical
    accuracy.

1.  Avoid tf.cast if possible. Eg, prefer `tf.where(cond, a, b)` over
    `tf.cast(cond,dtype=a.dtype)*a + (1-tf.cast(cond,dtype=b.dtype)*b`

1.  Preserve static shape hints.

1.  __Name management.__  Follow these conventions for indicating which
    names are internal vs external:

    *   Start names of private methods and private module functions with an
        underscore.
    *   List top-level classes and functions that are meant to be visible
        outside a module in its `__all__` constant.
    *   Import any names that are meant to be visible to clients of a package
        into that package's `__init__.py` file.  (A name that is public in a
        module but not imported into `__init__.py` is "package private".)
    *   Use TensorFlow's `remove_undocumented` feature in each `__init__.py`
        file to seal the package's methods.

1.  Submodule names should be singular, except where they overlap to TF.

    Justification: Having plural looks strange in user code, ie,
    tf.optimizer.Foo reads nicer than tf.optimizers.Foo since submodules are
    only used to access a single, specific thing (at a time).

1.  Use `tf.newaxis` rather than `None` to `tf.expand_dims`.

    Justification: Both work but only one is self-documenting.

1.  Use `'{}'.format()` rather than `'' %` for string formatting.

    Justification: [PEP 3101](https://www.python.org/dev/peps/pep-3101/) and
    [Python official
    tutorials](https://docs.python.org/3.2/tutorial/inputoutput.html#old-string-formatting):
    "...this old style of formatting will eventually be removed from the
    language, str.format() should generally be used."

1.  Prefer single quotes ('hello world') to double quotes ("hello world") for
    single-line string literals. However, when using triple quotes for multiline
    literals (docstrings, etc.), prefer """ (three double quotes) to '''.

    Justification: single quotes are slightly faster to type. They also avoid
    the need to escape double-quotes that occur organically in English text
    (at the cost of needing to escape contractions, which are debatably more
    common but can also often be avoided).

1.  When calling `tf.convert_to_tensor`, try to specify in advance a
    user-friendly dtype. If multiple `Tensor`s must be compatible, use
    `dtype_util.common_dtype([arg1, arg2], tf.float32)` to come up with a dtype
    that will be compatible across literals (even `int`s will work here),
    `Tensor`s, and Numpy objects. When converting a user-provided literal to a
    `Tensor` (see e.g. `Distribution._call_log_prob`), specify the dtype to
    `tf.convert_to_tensor` if it is available.
