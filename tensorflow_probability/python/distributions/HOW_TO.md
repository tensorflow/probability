#### Copyright 2022 The TensorFlow Authors.

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

# How to Write a Tensorflow Probability Distribution

This document is primarily aimed at authors wishing to add one or more
Distribution classes to this directory, but it should also be useful to
people writing unofficial Distributions, especially if they wish those
classes to interoperate with TFP code (such as `Mixture` or `JointDistribution`).

Here are the steps:

1. Can your distribution be implemented as a change of variables of an
existing distribution?  If so, doing it that way can save you a lot of work.
You will just need `__init__` and `_parameter_properties` methods, and the
`__init__` should end with a `super.__init__` call that looks like

  ```python
  super(MyDistribution, self).__init__(
    distribution=OtherDistribution(param1=param1, param2=param2,
                                   validate_args=validate_args,
                                   allow_nan_stats=allow_nan_stats),
    bijector=bijector_or_invert_bijector_that_does_the_transformation,
    validate_args=validate_args,
    parameters=parameters,
    name=name
    )
  )
  ```

See [chi.py](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/distributions/chi.py)
for an example of this technique.

1. Write a class inheriting from `tfp.distributions.Distribution` or a
sub-class of such:

    ```python
    from tensorflow_probability.python.distributions import distribution
    from tensorflow_probability.python.internal import parameter_properties

    class MyDistribution(distribution.Distribution):
    ```

1. Give your class an `__init__` method.  Besides the arguments needed to
specify the distribution, the `__init__` method should also take
`validate_args`, `allow_nan_stats`, and `name` arguments:

    ```python
    def __init__(self,
                 loc, scale, kurtosis,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='MyDistribution):
      """Initialize a MyDistribution distribution.

      Longer description of MyDistribution.

      Args:
        Document each __init__ argument.

      Raises:
        Specify here if it does.
    ```

    You are strongly encouraged to use English names (`scale`, `total_count`,
    `rate`, ...) for your `__init__` parameters instead of Greek
    (`mu`, `lambda`, `alpha`, ...).

    Your `__init__` method doesn't actually need to do much work other than
    save the passed parameters and perform some static checks on their
    validity.  If `validate_args` is True, you can and should also perform
    dynamic validity checks.

1. Call your class's parent's `__init__` at the end of your `__init__`:

    ```python
    def __init__(self, ...):
      parameters=dict(locals())
      ...

      super().__init__(
        dtype=the_dtype_MyDistribution_returns_for_samples,
        reparameterization_type= # either tfd.FULLY_REPARAMETERIZED or
                                 #        tfd.NOT_REPARAMETERIZED,
        validate_args, allow_nan_stats,
        parameters=parameters
        name=name)
    ```
    Note that if you subclass from a Distribution that does not have a
    `parameters` argument in its `__init__`, you will need to set
    `self._parameters` in your own `__init__`, after calling `super().__init__`.

1. Give your class a class comment describing the distribution.  Be sure
to give some examples of creating and using the class.

1. Give your class a `_parameter_properties` classmethod:

    ```python
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
      return dict(
        loc=parameter_properties.ParameterProperties(),
        scale=parameter_properties.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype))))),
        log_scale=parameter_properties.ParameterProperties(
            is_preferred=False)
        )
    ```
    Each value in the returned dictionary should be an instance of
    `parameter_properties.ParameterProperties`.  For real-valued parameters that
    can accept any real value, use `parameter_properties.ParameterProperties()`.
    For real-valued parameters that can accept only a certain subset of real
    values, use
    `parameter_properties.ParameterProperties(
    default_constraining_bijector_fn=lambda: PUT_BIJECTOR_HERE)` with a bijector
    that maps arbitrary real values to that subset.  For example, for the common
    case of a parameter that only accepts positive values, you could use
    `softplus_bijector.Softplus` as the bijector.  For more information on
    ParameterProperties, including non-tensor valued parameters, parameters that
    specify the shape of samples, and mutually-exclusive parameters like
    `logits` and `probs`, please see [the class documentation of
    `ParameterProperties`](https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/internal/parameter_properties.py#L45).

1. If you inherited from a concrete Distribution subclass that implements
`_sample_n` and `_log_prob`, you can skip to the last step
(unless you want and are able to
implement more efficient versions for your particular case.)

1. Define a `_sample_n` method that returns `n` samples from your distribution:

    ```python
    def _sample_n(self, n, seed=None):
    ```
    Actually, this method shouldn't exactly return `n` samples.  Rather, it
    should return `n` samples for each *batch* of the Distribution's parameters.
    The idea is that the Distribution should support [Numpy-like broadcasting](
    https://docs.scipy.org/doc/numpy-1.15.0/user/basics.broadcasting.html)
    for its arguments.  So for example, a Normal distribution could be given
    a `loc` of shape `(100, 32, 5)` and a scale of shape `(100, 32, 5)` and
    expect that a call to `_sample_n` with `n = 4` to return a value of shape
    `(4, 100, 32, 5)`.  Your `_sample_n` should do the same.  See the
    class documentation for `Distribution` in the **Broadcasting, batching,
    and shapes** section for more information.

    You will need to read the [PRNGS and seeds](
    https://github.com/tensorflow/probability/blob/main/PRNGS.md)
    document for a full understanding of how TFP handles random number
    generation, but to the first approximation you can either pass the
    given `seed` to a single routine in
    `tensorflow_probability.python.internal.samplers`, or
    you can "split" the seed using `samplers.split_seed` and use each split
    in an unique location.

1. You might be aware the TensorFlow Probability supports three different
backends:  TensorFlow, Numpy and JAX.  If you add your distribution to the
`python/distributions/` subdirectory, add it to the "distributions" target in
the BUILD file there, and build it using bazel, then your distribution will
also be usable in all three substrates.  Otherwise, it is fine to target a
single backend and to write your tensor manipulating code with just that API.

1. Define `_event_shape` and `_event_shape_tensor` methods.  These specify
the shape returned by `_sample_n` for a single batch.  They are usually

    ```python
    def _event_shape_tensor(self):
      return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
      return tf.TensorShape([])
    ```

    but if your distribution is defined over `m`-dimensional space, they would
    instead be

    ```python
    def _event_shape_tensor(self):
      return tf.constant([self.m], dtype=tf.int32)

    def _event_shape(self):
      return tf.TensorShape([self.m])
    ```

1. Define either a `_log_prob` or `_prob` method:

    ```python
    def _log_prob(self, x):
    ```
    Remember that the input `x` will be batched the same way as your
    Distribution's parameters (and it is fine to raise an error if it isn't.)
    This normally is
    pretty transparent, as the tensorflow and numpy routines you were probably
    going to use to implement `_log_prob` support batching themselves.

1. Define additional statistical methods as desired, such as `_mean`,
   `_variance`, `mode`, `_cdf`, `_quantile`, `_entropy`, etc.  Remember to
   support batching; these should all return tensors of shape
   `batch_shape + event_shape`, manually broadcasting if necessary.

1. Congratulations!  You are almost done!  The last step is to write some
unittests for your distribution.  Each method should have one or more unittests
that check the shape and value of what the method returns.  All tests should
be deterministic; you can use `test_util.test_seed()` for methods like
`sample` that need a random number generator seed.  See [the TFP unit-test best
practices document](https://github.com/tensorflow/probability/blob/main/UNITTEST.md)
for more information.
