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
"""Transformations of Edward2 programs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import six
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.experimental.edward2.interceptor import interceptable
from tensorflow_probability.python.experimental.edward2.interceptor import interception

__all__ = [
    "make_log_joint_fn",
    "make_value_setter"
]


def make_value_setter(**model_kwargs):
  """Creates a value-setting interceptor.

  This function creates an interceptor that sets values of Edward2 random
  variable objects. This is useful for a range of tasks, including conditioning
  on observed data, sampling from posterior predictive distributions, and as a
  building block of inference primitives such as computing log joint
  probabilities (see examples below).

  Args:
    **model_kwargs: dict of str to Tensor. Keys are the names of random
      variables in the model to which this interceptor is being applied. Values
      are Tensors to set their value to. Variables not included in this dict
      will not be set and will maintain their existing value semantics (by
      default, a sample from the parent-conditional distribution).

  Returns:
    set_values: function that sets the value of intercepted ops.

  #### Examples

  Consider for illustration a model with latent `z` and
  observed `x`, and a corresponding trainable posterior model:

  ```python
  num_observations = 10
  def model():
    z = ed.Normal(loc=0, scale=1., name='z')  # log rate
    x = ed.Poisson(rate=tf.exp(z) * tf.ones(num_observations), name='x')
    return x

  def variational_model():
    return ed.Normal(loc=tf.Variable(0.),
                     scale=tf.nn.softplus(tf.Variable(-4.)),
                     name='z')  # for simplicity, match name of the model RV.
  ```

  We can use a value-setting interceptor to condition the model on observed
  data. This approach is slightly more cumbersome than that of partially
  evaluating the complete log-joint function, but has the potential advantage
  that it returns a new model callable, which may be used to sample downstream
  variables, passed into additional transformations, etc.

  ```python
  x_observed = np.array([6, 3, 1, 8, 7, 0, 6, 4, 7, 5])
  def observed_model():
    with ed.interception(ed.make_value_setter(x=x_observed)):
      model()
  observed_log_joint_fn = ed.make_log_joint_fn(observed_model)

  # After fixing 'x', the observed log joint is now only a function of 'z'.
  # This enables us to define a variational lower bound,
  # `E_q[ log p(x, z) - log q(z)]`, simply by evaluating the observed and
  # variational log joints at variational samples.
  variational_log_joint_fn = ed.make_log_joint_fn(variational_model)
  with ed.tape() as variational_sample:  # Sample trace from variational model.
    variational_model()
  elbo_loss = -(observed_log_joint_fn(**variational_sample) -
                variational_log_joint_fn(**variational_sample))
  ```

  After performing inference by minimizing the variational loss, a value-setting
  interceptor enables simulation from the posterior predictive distribution:

  ```python
  with ed.tape() as posterior_samples:  # tape is a map {rv.name : rv}
    variational_model()
  with ed.interception(ed.make_value_setter(**posterior_samples)):
    x = model()
  # x is a sample from p(X | Z = z') where z' ~ q(z) (the variational model)
  ```

  As another example, using a value setter inside of `ed.tape` enables
  computing the log joint probability, by setting all variables to
  posterior values and then accumulating the log probs of those values under
  the induced parent-conditional distributions. This is one way that we could
  have implemented `ed.make_log_joint_fn`:

  ```python
  def make_log_joint_fn_demo(model):
    def log_joint_fn(**model_kwargs):
      with ed.tape() as model_tape:
        with ed.make_value_setter(**model_kwargs):
          model()

      # accumulate sum_i log p(X_i = x_i | X_{:i-1} = x_{:i-1})
      log_prob = 0.
      for rv in model_tape.values():
        log_prob += tf.reduce_sum(rv.log_prob(rv.value))

      return log_prob
    return log_joint_fn
  ```

  """
  def set_values(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    name = kwargs.get("name")
    if name in model_kwargs:
      kwargs["value"] = model_kwargs[name]
    return interceptable(f)(*args, **kwargs)
  return set_values


def make_log_joint_fn(model):
  """Takes Edward probabilistic program and returns its log joint function.

  Args:
    model: Python callable which executes the generative process of a
      computable probability distribution using `ed.RandomVariable`s.

  Returns:
    A log-joint probability function. Its inputs are `model`'s original inputs
    and random variables which appear during the program execution. Its output
    is a scalar tf.Tensor.

  #### Examples

  Below we define Bayesian logistic regression as an Edward program,
  representing the model's generative process. We apply `make_log_joint_fn` in
  order to represent the model in terms of its joint probability function.

  ```python
  from tensorflow_probability import edward2 as ed

  def logistic_regression(features):
    coeffs = ed.Normal(loc=0., scale=1.,
                       sample_shape=features.shape[1], name="coeffs")
    outcomes = ed.Bernoulli(logits=tf.tensordot(features, coeffs, [[1], [0]]),
                            name="outcomes")
    return outcomes

  log_joint = ed.make_log_joint_fn(logistic_regression)

  features = tf.random.normal([3, 2])
  coeffs_value = tf.random.normal([2])
  outcomes_value = tf.round(tf.random.uniform([3]))
  output = log_joint(features, coeffs=coeffs_value, outcomes=outcomes_value)
  ```

  """
  def log_joint_fn(*args, **kwargs):
    """Log-probability of inputs according to a joint probability distribution.

    Args:
      *args: Positional arguments. They are the model's original inputs and can
        alternatively be specified as part of `kwargs`.
      **kwargs: Keyword arguments, where for each key-value pair `k` and `v`,
        `v` is passed as a `value` to the random variable(s) whose keyword
        argument `name` during construction is equal to `k`.

    Returns:
      Scalar tf.Tensor, which represents the model's log-probability summed
      over all Edward random variables and their dimensions.

    Raises:
      TypeError: If a random variable in the model has no specified value in
        `**kwargs`.
    """
    log_probs = []

    def interceptor(rv_constructor, *rv_args, **rv_kwargs):
      """Overrides a random variable's `value` and accumulates its log-prob."""
      # Set value to keyword argument indexed by `name` (an input tensor).
      rv_name = rv_kwargs.get("name")
      if rv_name is None:
        raise KeyError("Random variable constructor {} has no name "
                       "in its arguments.".format(rv_constructor.__name__))

      # If no value is explicitly passed in for an RV, default to the value
      # from the RV constructor. This may have been set explicitly by the user
      # or forwarded from a lower-level interceptor.
      previously_specified_value = rv_kwargs.get("value")
      value = kwargs.get(rv_name, previously_specified_value)
      if value is None:
        raise LookupError("Keyword argument specifying value for {} is "
                          "missing.".format(rv_name))
      rv_kwargs["value"] = value

      rv = rv_constructor(*rv_args, **rv_kwargs)
      log_prob = tf.reduce_sum(input_tensor=rv.distribution.log_prob(rv.value))
      log_probs.append(log_prob)
      return rv

    model_kwargs = _get_function_inputs(model, kwargs)
    with interception(interceptor):
      model(*args, **model_kwargs)
    log_prob = sum(log_probs)
    return log_prob
  return log_joint_fn


def _get_function_inputs(f, src_kwargs):
  """Filters inputs to be compatible with function `f`'s signature.

  Args:
    f: Function according to whose input signature we filter arguments.
    src_kwargs: Keyword arguments to filter according to `f`.

  Returns:
    kwargs: Dict of key-value pairs in `src_kwargs` which exist in `f`'s
      signature.
  """
  if hasattr(f, "_func"):  # functions returned by tf.make_template
    f = f._func  # pylint: disable=protected-access

  try:  # getargspec was deprecated in Python 3.6
    argspec = inspect.getfullargspec(f)
  except AttributeError:
    argspec = inspect.getargspec(f)

  fkwargs = {k: v for k, v in six.iteritems(src_kwargs) if k in argspec.args}
  return fkwargs
