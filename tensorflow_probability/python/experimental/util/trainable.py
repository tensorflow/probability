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

# Dependency imports
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.util import deferred_tensor
from tensorflow.python.util import tf_inspect  # pylint: disable=g-direct-tensorflow-import

__all__ = [
    'make_trainable',
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


def make_trainable(cls,
                   initial_parameters=None,
                   unconstrained_init_fn=samplers.normal,
                   batch_and_event_shape=(),
                   parameter_dtype=tf.float32,
                   seed=None,
                   **init_kwargs):
  """Constructs a distribution or bijector instance with trainable parameters.

  This is a convenience method that instantiates a class using
  `tf.Variables` for its underlying trainable parameters. Parameters are
  randomly initialized, and transformed to enforce any domain constraints. This
  method assumes that the class exposes a `parameter_properties` method
  annotating its trainable parameters, and that the caller provides any
  additional (non-trainable) arguments required by the class.

  Args:
    cls: Python class that implements `cls.parameter_properties()`, e.g., a TFP
      distribution (`tfd.Normal`) or bijector (`tfb.Scale`).
    initial_parameters: Optional `str : Tensor` dictionary specifying initial
      values for some or all of the trainable parameters. These values are
      used directly and must lie in the parameter domain, e.g., the initial
      value for a scale parameter must be positive. If no initial value is
      provided for a parameter, it will be initialized randomly as determined
      by the `unconstrained_unit_fn`.
      Default value: `None`.
    unconstrained_init_fn: Python `callable` that takes `shape`, `seed`, and
      `dtype` arguments, and returns a random real-valued `Tensor` of the
      specified shape and dtype. Any domain constraints, e.g. a requirement that
      a parameter must be positive, are applied by passing the sampled values
      through the default constraining bijectors specified in
      `cls.parameter_properties()`.
      Default value: `tf.random.stateless_normal`.
    batch_and_event_shape: Optional int `Tensor` desired shape of samples
      (for distributions) or inputs (for bijectors), used to determine the shape
      of the trainable parameters.
      Default value: `()`.
    parameter_dtype: Optional float `dtype` for trainable variables.
    seed: Optional random seed used to determine initial values.
      Default value: `None`.
    **init_kwargs: Additional keyword arguments passed to `cls.__init__()` to
      specify any non-trainable parameters. If a value is passed for
      an otherwise-trainable parameter---for example,
      `trainable(tfd.Normal, scale=1.)`---it will be taken as a fixed value and
      no variable will be constructed for that parameter.
  Returns:
    trainable_instance: an instance of `cls` parameterized by trainable
      variables.

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

  ```python
  model = tfp.util.make_trainable(tfd.Normal)
  losses = tfp.math.minimize(
    lambda: -model.log_prob(samples),
    optimizer=tf.optimizers.Adam(0.1),
    num_steps=200)
  print('Fit Normal distribution with mean {} and stddev {}'.format(
    model.mean(),
    model.stddev()))
  ```

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

    if initial_parameters is None:
      initial_parameters = {}

    # Create a trainable variable for each parameter.
    for parameter_name, properties in cls.parameter_properties(
        dtype=parameter_dtype).items():
      if parameter_name in init_kwargs:  # Prefer user-provided values.
        continue
      if not properties.is_preferred:
        continue

      parameter_shape = properties.shape_fn(batch_and_event_shape)
      constraining_bijector = properties.default_constraining_bijector_fn()
      if parameter_name in initial_parameters:
        # Ensure that the variable we initialize has full shape. For example,
        # trainable(tfd.Normal,
        #           batch_and_event_shape=[100],
        #           initial_parameters={'scale': 1e-4})
        # should create a batch of 100 scale variables, not a single scalar
        # variable shared over all batch elements. (A user who wants the latter
        # can always construct it manually, but that's a niche case and enough
        # of a footgun that we want to avoid going there automatically).
        initial_value = tf.broadcast_to(
            tf.cast(initial_parameters[parameter_name],
                    dtype=parameter_dtype),
            shape=parameter_shape)
      else:
        seed, init_seed = samplers.split_seed(seed)
        initial_value = constraining_bijector.forward(
            unconstrained_init_fn(
                constraining_bijector.inverse_event_shape(parameter_shape),
                seed=init_seed,
                dtype=parameter_dtype))
      init_kwargs[parameter_name] = deferred_tensor.TransformedVariable(
          initial_value,
          constraining_bijector,
          name=parameter_name)

  return cls(**init_kwargs)
