# Copyright 2021 The TensorFlow Probability Authors.
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
"""Trainable instances of distributions and bijectors."""

import functools

import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import docstring_util
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import trainable_state_util

from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'make_trainable',
    'make_trainable_stateless'
]


def _get_arg_value(arg_name, f, kwargs):
  """Attempts to infer arg's value within the scope created by `f(**kwargs)`."""
  if arg_name in kwargs:
    return kwargs[arg_name]
  argspec = tf_inspect.getfullargspec(f)
  if arg_name in argspec.args:
    return argspec.defaults[
        # Index from the right (with a negative index) since only a suffix of
        # args have defaults.
        argspec.args.index(arg_name) - len(argspec.args)]
  return (argspec.kwonlydefaults or {}).get(arg_name)


def _default_parameter_init_fn(initial_parameters):
  """Builds a function to specify user-provided initial values."""

  def _initialize_with_sampler_fallback(parameter_name,
                                        shape,
                                        dtype,
                                        seed,
                                        constraining_bijector):
    if parameter_name in initial_parameters:
      # Ensure that the variable we initialize has full shape. For example,
      # trainable(tfd.Normal,
      #           batch_and_event_shape=[100],
      #           initial_parameters={'scale': 1e-4})
      # should create a batch of 100 scale variables, not a single scalar
      # variable shared over all batch elements.
      return tf.broadcast_to(tf.cast(initial_parameters[parameter_name],
                                     dtype=dtype),
                             shape)
    # If no value was provided, sample an appropriately constrained value.
    return constraining_bijector.forward(
        samplers.normal(
            constraining_bijector.inverse_event_shape_tensor(shape),
            dtype=dtype,
            seed=seed))

  return _initialize_with_sampler_fallback


def _make_trainable(cls,
                    initial_parameters=None,
                    batch_and_event_shape=(),
                    parameter_dtype=tf.float32,
                    **init_kwargs):
  """Constructs a distribution or bijector instance with trainable parameters.

  This is a convenience method that instantiates a class with trainable
  parameters. Parameters are randomly initialized, and transformed to enforce
  any domain constraints. This method assumes that the class exposes a
  `parameter_properties` method annotating its trainable parameters, and that
  the caller provides any additional (non-trainable) arguments required by the
  class.

  Args:
    cls: Python class that implements `cls.parameter_properties()`, e.g., a TFP
      distribution (`tfd.Normal`) or bijector (`tfb.Scale`).
    initial_parameters: a dictionary containing initial values for some or
      all of the parameters to `cls`, OR a Python `callable` with signature
      `value = parameter_init_fn(parameter_name, shape, dtype, seed,
      constraining_bijector)`. If a dictionary is provided, any parameters not
      specified will be initialized to a random value in their domain.
      Default value: `None` (equivalent to `{}`; all parameters are
        initialized randomly).
    batch_and_event_shape: Optional int `Tensor` desired shape of samples
      (for distributions) or inputs (for bijectors), used to determine the shape
      of the trainable parameters.
      Default value: `()`.
    parameter_dtype: Optional float `dtype` for trainable variables.
    **init_kwargs: Additional keyword arguments passed to `cls.__init__()` to
      specify any non-trainable parameters. If a value is passed for
      an otherwise-trainable parameter---for example,
      `trainable(tfd.Normal, scale=1.)`---it will be taken as a fixed value and
      no variable will be constructed for that parameter.
  Yields:
    *parameters: sequence of `trainable_state_util.Parameter` namedtuples.
      These are intended to be consumed by
      `trainable_state_util.as_stateful_builder` and
      `trainable_state_util.as_stateless_builder` to define stateful and
      stateless variants respectively.

  #### Example

  Suppose we want to fit a normal distribution to observed data. We could
  of course just examine the empirical mean and standard deviation of the data:

  ```python
  samples = [4.57, 6.37, 5.93, 7.98, 2.03, 3.59, 8.55, 3.45, 5.06, 6.44]
  model = tfd.Normal(
    loc=tf.reduce_mean(samples),  # ==> 5.40
    scale=tf.math.reduce_std(sample))  # ==> 1.95
  ```

  and this would be a very sensible approach. But that's boring, so instead,
  let's do way more work to get the same result. We'll build a trainable normal
  distribution, and explicitly optimize to find the maximum-likelihood estimate
  for the parameters given our data:

  ${minimize_example_code}

  In this trivial case, doing the explicit optimization has few advantages over
  the first approach in which we simply matched the empirical moments of the
  data. However, trainable distributions are useful more generally. For example,
  they can enable maximum-likelihood estimation of distributions when a
  moment-matching estimator is not available, and they can also serve as
  surrogate posteriors in variational inference.

  """

  # Attempt to set a name scope using the name of the object we're about to
  # create, so that the variables we create are easy to identity.
  name_arg = _get_arg_value(arg_name='name', f=cls.__init__, kwargs=init_kwargs)
  with tf.name_scope(
      ((name_arg + '_') if name_arg else '') + 'trainable_variables'):

    # Canonicalize initial parameter specification as `parameter_init_fn`.
    if initial_parameters is None:
      initial_parameters = {}
    parameter_init_fn = initial_parameters
    if not callable(parameter_init_fn):
      parameter_init_fn = _default_parameter_init_fn(initial_parameters)

    # Create a trainable variable for each parameter.
    for parameter_name, properties in cls.parameter_properties(
        dtype=parameter_dtype).items():
      if parameter_name in init_kwargs:  # Prefer user-provided values.
        continue
      if not (properties.is_tensor and properties.is_preferred):
        continue
      if properties.specifies_shape or (properties.event_ndims is None):
        continue

      parameter_shape = properties.shape_fn(batch_and_event_shape)
      constraining_bijector = properties.default_constraining_bijector_fn()

      init_kwargs[parameter_name] = yield trainable_state_util.Parameter(
          init_fn=functools.partial(
              parameter_init_fn,
              parameter_name,
              shape=parameter_shape,
              dtype=parameter_dtype,
              constraining_bijector=constraining_bijector),
          constraining_bijector=constraining_bijector,
          name=parameter_name)

  return cls(**init_kwargs)

make_trainable = docstring_util.expand_docstring(
    minimize_example_code="""
```python
model = tfp.util.make_trainable(tfd.Normal)
losses = tfp.math.minimize(
  lambda: -model.log_prob(samples),
  optimizer=tf_keras.optimizers.Adam(0.1),
  num_steps=200)
print('Fit Normal distribution with mean {} and stddev {}'.format(
  model.mean(),
  model.stddev()))
```""")(trainable_state_util.as_stateful_builder(_make_trainable))

make_trainable_stateless = docstring_util.expand_docstring(
    minimize_example_code="""
```python
init_fn, apply_fn = tfe_util.make_trainable_stateless(tfd.Normal)

import optax  # JAX only.
mle_params, losses = tfp.math.minimize_stateless(
  lambda *params: -apply_fn(params).log_prob(samples),
  init=init_fn(),
  optimizer=optax.adam(0.1),
  num_steps=200)
model = apply_fn(mle_params)
print('Fit Normal distribution with mean {} and stddev {}'.format(
  model.mean(),
  model.stddev()))
```""")(trainable_state_util.as_stateless_builder(_make_trainable))
