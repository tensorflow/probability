#### Copyright 2018 The TensorFlow Authors.

```none
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

# Standards for `tfp.distributions`

This document represents the TFP standards for all members of the
`tfp.distributions` module. These standards only apply to the
`tfp.distributions` module and is intentionally a "high bar" meant to protect
users of the module. The ["TensorFlow Distributions"](
https://arxiv.org/abs/1711.10604) whitepaper is an exposition on the standards
described here (see Section 3). These standards have been in effect since
16-aug-2016.

You are encouraged to subclass `tfp.distributions.Distribution` and invited to
_disregard any/all of these standards_. We especially recommend you ignore point
#4.

All TFP code (including `tfp.distributions`) follows the [TFP Style Guide](
https://github.com/tensorflow/probability/blob/master/STYLE_GUIDE.md#tensorflow-probability-style-guide).

## Requirements (Comprehensive)

1. A `Distribution` subclass must implement a `sample` function. The `sample`
   function shall generate random outcomes which in expectation correspond to
   other properties of the `Distribution`. Other properties must assume the
   semantics implied by the output of `sample`.

2. A `Distribution` must implement a `log_prob` function. This function is the
   log probability mass function or probability density function, depending on
   the underlying measure.

3. All member functions must be _efficiently computable_. To us, efficiently
   computable means: computable in (at most) expected polynomial time.

4. All member functions (except sample) must be exact. This means, for example,
   Monte Carlo approximations are disallowed and even if they hard-code RNG
   state (for reproducibility). For example, it is acceptable to implement an
   efficiently computable analytical upper bound on entropy but it is not
   acceptable to implement a Monte Carlo estimate of entropy.

5. Implementing other distribution properties is highly encouraged iff they are
   mathematically well-defined and efficiently computable. For example
   `mean_log_det` is a meaningful property for a Wishart but not for a (scalar)
   Normal.

6. A `Distribution`'s inputs/outputs are `Tensor`s with ["`Distribution` shape
   semantics."](
   https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb)
   For example, functions like `prob` and `cdf` accept a `Tensor` with
   "`Distribution` shape semantics" whereas `sample` returns a `Tensor` with
   "Distribution shape semantics."

   - *Exception*: `tfd.JointDistribution` takes/returns a `list`-like of
     `Tensor`s.

7. A `Distribution`'s `event_ndims` must be known statically. For example,
   `Wishart` has `event_ndims=2`, `MultivariateNormalDiag` has `event_ndims=1`,
   `Normal` has `event_ndims=0`, `Categorical` has `event_ndims=0` and
   `OneHotCategorical` has `event_ndims=1`. The `event_shape` need not be known
   statically, i.e., this might only be known at runtime (in Eager mode) or at
   graph execution time (in graph mode). Often a `Distribution`'s `event_ndims`
   will be self-evident from the class name itself.

8. All `Distribution`s are implicitly or explicitly conditioned on global or
   local (per-sample) parameters. A `Distribution` infers dtype and batch/event
   shape from its global parameters. For example, a scalar `Distribution`'s
   `event_shape` is implicitly inferrable (`event_shape=[]`) thus always known
   statically; the same is not necessarily true of a `MultivariateNormalDiag`.

9. When possible, a `Distribution` must support [Numpy-like broadcasting](
   https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html)
   for all arguments. When broadcasting is not possible, arguments must be
     validated.

10. All possible effort is made to validate arguments prior to graph execution.
    Any validation requiring graph execution must be gated by a Boolean,
    global parameter.

11. `Distribution` parameters are descriptive English (i.e., not Greek letters)
    and draw upon a relatively small, shared lexicon. Examples include: `loc`,
    `scale`, `concentration`, `rate`, `probs`, `logits`, `df`. When forced to
    choose between mathematical purity and conveying intuitive meaning, prefer
    the latter but provide extensive documentation.

## Non-Requirements (Noncomprehensive)

In this section we list items which have historically been presumed true of
`tfp.distributions` but are not official requirements.

1. Mutable state is not explicitly disallowed. However, it is _highly
   discouraged_ as it makes reasoning about the object more challenging (for
   both API owner and user). As of 18-nov-2018, no `tfp.distributions` member
   has its own mutable state although all distributions do mutate the global
   random number generate state on access to `sample`.

2. Subclasses are free to override public base class members. I.e., you don't
   have to follow the "public calls private" pattern. (However, as of
   18-nov-2018, there has not yet been a reason to deviate from this pattern.)
