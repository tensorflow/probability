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
"""Generalized Linear Model Fisher Scoring."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.math.linalg import sparse_or_dense_matvecmul


__all__ = [
    'fit',
    'fit_one_step',
    'compute_predicted_linear_response',
    'convergence_criteria_small_relative_norm_weights_change',
]


def fit(
    model_matrix,
    response,
    model,
    model_coefficients_start=None,
    predicted_linear_response_start=None,
    l2_regularizer=None,
    dispersion=None,
    offset=None,
    convergence_criteria_fn=None,
    learning_rate=None,
    fast_unsafe_numerics=True,
    maximum_iterations=None,
    l2_regularization_penalty_factor=None,
    name=None):
  """Runs multiple Fisher scoring steps.

  Args:
    model_matrix: (Batch of) `float`-like, matrix-shaped `Tensor` where each row
      represents a sample's features.
    response: (Batch of) vector-shaped `Tensor` where each element represents a
      sample's observed response (to the corresponding row of features). Must
      have same `dtype` as `model_matrix`.
    model: `tfp.glm.ExponentialFamily`-like instance which implicitly
      characterizes a negative log-likelihood loss by specifying the
      distribuion's `mean`, `gradient_mean`, and `variance`.
    model_coefficients_start: Optional (batch of) vector-shaped `Tensor`
      representing the initial model coefficients, one for each column in
      `model_matrix`. Must have same `dtype` as `model_matrix`.
      Default value: Zeros.
    predicted_linear_response_start: Optional `Tensor` with `shape`, `dtype`
      matching `response`; represents `offset` shifted initial linear
      predictions based on `model_coefficients_start`.
      Default value: `offset` if `model_coefficients is None`, and
      `tf.linalg.matvec(model_matrix, model_coefficients_start) + offset`
      otherwise.
    l2_regularizer: Optional scalar `Tensor` representing L2 regularization
      penalty, i.e.,
      `loss(w) = sum{-log p(y[i]|x[i],w) : i=1..n} + l2_regularizer ||w||_2^2`.
      Default value: `None` (i.e., no L2 regularization).
    dispersion: Optional (batch of) `Tensor` representing `response` dispersion,
      i.e., as in, `p(y|theta) := exp((y theta - A(theta)) / dispersion)`.
      Must broadcast with rows of `model_matrix`.
      Default value: `None` (i.e., "no dispersion").
    offset: Optional `Tensor` representing constant shift applied to
      `predicted_linear_response`.  Must broadcast to `response`.
      Default value: `None` (i.e., `tf.zeros_like(response)`).
    convergence_criteria_fn: Python `callable` taking:
      `is_converged_previous`, `iter_`, `model_coefficients_previous`,
      `predicted_linear_response_previous`, `model_coefficients_next`,
      `predicted_linear_response_next`, `response`, `model`, `dispersion` and
      returning a `bool` `Tensor` indicating that Fisher scoring has converged.
      See `convergence_criteria_small_relative_norm_weights_change` as an
      example function.
      Default value: `None` (i.e.,
      `convergence_criteria_small_relative_norm_weights_change`).
    learning_rate: Optional (batch of) scalar `Tensor` used to dampen iterative
      progress. Typically only needed if optimization diverges, should be no
      larger than `1` and typically very close to `1`.
      Default value: `None` (i.e., `1`).
    fast_unsafe_numerics: Optional Python `bool` indicating if faster, less
      numerically accurate methods can be employed for computing the weighted
      least-squares solution.
      Default value: `True` (i.e., "fast but possibly diminished accuracy").
    maximum_iterations: Optional maximum number of iterations of Fisher scoring
      to run; "and-ed" with result of `convergence_criteria_fn`.
      Default value: `None` (i.e., `infinity`).
    l2_regularization_penalty_factor: Optional (batch of) vector-shaped
      `Tensor`, representing a separate penalty factor to apply to each model
      coefficient, length equal to columns in `model_matrix`. Each penalty
      factor multiplies l2_regularizer to allow differential regularization. Can
      be 0 for some coefficients, which implies no regularization. Default is 1
      for all coefficients.
      `loss(w) = sum{-log p(y[i]|x[i],w) : i=1..n} + l2_regularizer ||w *
        l2_regularization_penalty_factor||_2^2`
      Default value: `None` (i.e., no per coefficient regularization).
    name: Python `str` used as name prefix to ops created by this function.
      Default value: `"fit"`.

  Returns:
    model_coefficients: (Batch of) vector-shaped `Tensor`; represents the
      fitted model coefficients, one for each column in `model_matrix`.
    predicted_linear_response: `response`-shaped `Tensor` representing linear
      predictions based on new `model_coefficients`, i.e.,
      `tf.linalg.matvec(model_matrix, model_coefficients) + offset`.
    is_converged: `bool` `Tensor` indicating that the returned
      `model_coefficients` met the `convergence_criteria_fn` criteria within the
      `maximum_iterations` limit.
    iter_: `int32` `Tensor` indicating the number of iterations taken.

  #### Example

  ```python
  from __future__ import print_function
  import numpy as np
  import tensorflow as tf
  import tensorflow_probability as tfp
  tfd = tfp.distributions

  def make_dataset(n, d, link, scale=1., dtype=np.float32):
    model_coefficients = tfd.Uniform(
        low=np.array(-1, dtype),
        high=np.array(1, dtype)).sample(d, seed=42)
    radius = np.sqrt(2.)
    model_coefficients *= radius / tf.linalg.norm(model_coefficients)
    model_matrix = tfd.Normal(
        loc=np.array(0, dtype),
        scale=np.array(1, dtype)).sample([n, d], seed=43)
    scale = tf.convert_to_tensor(scale, dtype)
    linear_response = tf.tensordot(
        model_matrix, model_coefficients, axes=[[1], [0]])
    if link == 'linear':
      response = tfd.Normal(loc=linear_response, scale=scale).sample(seed=44)
    elif link == 'probit':
      response = tf.cast(
          tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0,
          dtype)
    elif link == 'logit':
      response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
    else:
      raise ValueError('unrecognized true link: {}'.format(link))
    return model_matrix, response, model_coefficients

  X, Y, w_true = make_dataset(n=int(1e6), d=100, link='probit')

  w, linear_response, is_converged, num_iter = tfp.glm.fit(
      model_matrix=X,
      response=Y,
      model=tfp.glm.BernoulliNormalCDF())
  log_likelihood = tfp.glm.BernoulliNormalCDF().log_prob(Y, linear_response)

  print('is_converged: ', is_converged.numpy())
  print('    num_iter: ', num_iter.numpy())
  print('    accuracy: ', np.mean((linear_response > 0.) == tf.cast(Y, bool)))
  print('    deviance: ', 2. * np.mean(log_likelihood))
  print('||w0-w1||_2 / (1+||w0||_2): ', (np.linalg.norm(w_true - w, ord=2) /
                                         (1. + np.linalg.norm(w_true, ord=2))))

  # ==>
  # is_converged:  True
  #     num_iter:  6
  #     accuracy:  0.804382
  #     deviance:  -0.820746600628
  # ||w0-w1||_2 / (1+||w0||_2):  0.00619245105309
  ```

  """
  with tf.name_scope(name or 'fit'):
    [
        model_matrix,
        response,
        model_coefficients_start,
        predicted_linear_response_start,
        offset,
    ] = prepare_args(
        model_matrix,
        response,
        model_coefficients_start,
        predicted_linear_response_start,
        offset)
    if convergence_criteria_fn is None:
      convergence_criteria_fn = (
          convergence_criteria_small_relative_norm_weights_change())

    def _body(
        is_converged_previous,
        iter_,
        model_coefficients_previous,
        predicted_linear_response_previous):
      """`tf.while_loop` body."""
      model_coefficients_next, predicted_linear_response_next = fit_one_step(
          model_matrix,
          response,
          model,
          model_coefficients_previous,
          predicted_linear_response_previous,
          l2_regularizer,
          dispersion,
          offset,
          learning_rate,
          fast_unsafe_numerics,
          l2_regularization_penalty_factor,
          name)
      is_converged_next = convergence_criteria_fn(
          is_converged_previous=is_converged_previous,
          iter_=iter_,
          model_coefficients_previous=model_coefficients_previous,
          predicted_linear_response_previous=predicted_linear_response_previous,
          model_coefficients_next=model_coefficients_next,
          predicted_linear_response_next=predicted_linear_response_next,
          response=response,
          model=model,
          dispersion=dispersion)
      return [
          is_converged_next,
          iter_ + 1,
          model_coefficients_next,
          predicted_linear_response_next,
      ]

    # while not converged:
    #   fit_one_step
    [
        is_converged,
        iter_,
        model_coefficients,
        predicted_linear_response,
    ] = tf.while_loop(
        cond=lambda is_converged, *args: tf.logical_not(is_converged),
        body=_body,
        loop_vars=[
            tf.zeros([], np.bool_),   # is_converged
            tf.zeros([], np.int32),  # iter_
            model_coefficients_start,
            predicted_linear_response_start,
        ],
        maximum_iterations=maximum_iterations)

    return [
        model_coefficients,
        predicted_linear_response,
        is_converged,
        iter_
    ]


def fit_one_step(
    model_matrix,
    response,
    model,
    model_coefficients_start=None,
    predicted_linear_response_start=None,
    l2_regularizer=None,
    dispersion=None,
    offset=None,
    learning_rate=None,
    fast_unsafe_numerics=True,
    l2_regularization_penalty_factor=None,
    name=None):
  """Runs one step of Fisher scoring.

  Args:
    model_matrix: (Batch of) `float`-like, matrix-shaped `Tensor` where each row
      represents a sample's features.
    response: (Batch of) vector-shaped `Tensor` where each element represents a
      sample's observed response (to the corresponding row of features). Must
      have same `dtype` as `model_matrix`.
    model: `tfp.glm.ExponentialFamily`-like instance used to construct the
      negative log-likelihood loss, gradient, and expected Hessian (i.e., the
      Fisher information matrix).
    model_coefficients_start: Optional (batch of) vector-shaped `Tensor`
      representing the initial model coefficients, one for each column in
      `model_matrix`. Must have same `dtype` as `model_matrix`.
      Default value: Zeros.
    predicted_linear_response_start: Optional `Tensor` with `shape`, `dtype`
      matching `response`; represents `offset` shifted initial linear
      predictions based on `model_coefficients_start`.
      Default value: `offset` if `model_coefficients is None`, and
      `tf.linalg.matvec(model_matrix, model_coefficients_start) + offset`
      otherwise.
    l2_regularizer: Optional scalar `Tensor` representing L2 regularization
      penalty, i.e.,
      `loss(w) = sum{-log p(y[i]|x[i],w) : i=1..n} + l2_regularizer ||w||_2^2`.
      Default value: `None` (i.e., no L2 regularization).
    dispersion: Optional (batch of) `Tensor` representing `response` dispersion,
      i.e., as in, `p(y|theta) := exp((y theta - A(theta)) / dispersion)`.
      Must broadcast with rows of `model_matrix`.
      Default value: `None` (i.e., "no dispersion").
    offset: Optional `Tensor` representing constant shift applied to
      `predicted_linear_response`.  Must broadcast to `response`.
      Default value: `None` (i.e., `tf.zeros_like(response)`).
    learning_rate: Optional (batch of) scalar `Tensor` used to dampen iterative
      progress. Typically only needed if optimization diverges, should be no
      larger than `1` and typically very close to `1`.
      Default value: `None` (i.e., `1`).
    fast_unsafe_numerics: Optional Python `bool` indicating if solve should be
      based on Cholesky or QR decomposition.
      Default value: `True` (i.e., "prefer speed via Cholesky decomposition").
    l2_regularization_penalty_factor: Optional (batch of) vector-shaped
      `Tensor`, representing a separate penalty factor to apply to each model
      coefficient, length equal to columns in `model_matrix`. Each penalty
      factor multiplies l2_regularizer to allow differential regularization. Can
      be 0 for some coefficients, which implies no regularization. Default is 1
      for all coefficients.
      `loss(w) = sum{-log p(y[i]|x[i],w) : i=1..n} + l2_regularizer ||w *
        l2_regularization_penalty_factor||_2^2`
    name: Python `str` used as name prefix to ops created by this function.
      Default value: `"fit_one_step"`.

  Returns:
    model_coefficients: (Batch of) vector-shaped `Tensor`; represents the
      next estimate of the model coefficients, one for each column in
      `model_matrix`.
    predicted_linear_response: `response`-shaped `Tensor` representing linear
      predictions based on new `model_coefficients`, i.e.,
      `tf.linalg.matvec(model_matrix, model_coefficients_next) + offset`.
  """
  with tf.name_scope(name or 'fit_one_step'):

    [
        model_matrix,
        response,
        model_coefficients_start,
        predicted_linear_response_start,
        offset,
    ] = prepare_args(
        model_matrix,
        response,
        model_coefficients_start,
        predicted_linear_response_start,
        offset)

    # Compute: mean, grad(mean, predicted_linear_response_start), and variance.
    mean, variance, grad_mean = model(predicted_linear_response_start)

    # If either `grad_mean` or `variance is non-finite or zero, then we'll
    # replace it with a value such that the row is zeroed out. Although this
    # procedure may seem circuitous, it is necessary to ensure this algorithm is
    # itself differentiable.
    is_valid = (
        tf.math.is_finite(grad_mean) & tf.not_equal(grad_mean, 0.)
        & tf.math.is_finite(variance) & (variance > 0.))

    def mask_if_invalid(x, mask):
      return tf.where(
          is_valid, x, np.array(mask, dtype_util.as_numpy_dtype(x.dtype)))

    # Run one step of iteratively reweighted least-squares.
    # Compute "`z`", the adjusted predicted linear response.
    # z = predicted_linear_response_start
    #     + learning_rate * (response - mean) / grad_mean
    z = (response - mean) / mask_if_invalid(grad_mean, 1.)
    # TODO(jvdillon): Rather than use learning rate, we should consider using
    # backtracking line search.
    if learning_rate is not None:
      z *= learning_rate[..., tf.newaxis]
    z += predicted_linear_response_start
    if offset is not None:
      z -= offset

    # Compute "`w`", the per-sample weight.
    if dispersion is not None:
      # For convenience, we'll now scale the variance by the dispersion factor.
      variance *= dispersion
    w = (
        mask_if_invalid(grad_mean, 0.) *
        tf.math.rsqrt(mask_if_invalid(variance, np.inf)))

    a = model_matrix * w[..., tf.newaxis]
    b = z * w
    # Solve `min{ || A @ model_coefficients - b ||_2**2 : model_coefficients }`
    # where `@` denotes `matmul`.

    if l2_regularizer is None:
      l2_regularizer = np.array(0, dtype_util.as_numpy_dtype(a.dtype))
    else:
      l2_regularizer_ = distribution_util.maybe_get_static_value(
          l2_regularizer, dtype_util.as_numpy_dtype(a.dtype))
      if l2_regularizer_ is not None:
        l2_regularizer = l2_regularizer_

    def _embed_l2_regularization():
      """Adds synthetic observations to implement L2 regularization."""
      # `tf.matrix_solve_ls` does not respect the `l2_regularization` argument
      # when `fast_unsafe_numerics` is `False`. This function  adds synthetic
      # observations to the data to implement the regularization instead.
      # Adding observations `sqrt(l2_regularizer) * I` is mathematically
      # equivalent to adding the term
      # `-l2_regularizer ||coefficients||_2**2` to the log-likelihood.
      num_model_coefficients = num_cols(model_matrix)
      batch_shape = tf.shape(model_matrix)[:-2]
      if l2_regularization_penalty_factor is None:
        eye = tf.eye(
            num_model_coefficients, batch_shape=batch_shape, dtype=a.dtype)
      else:
        eye = tf.linalg.tensor_diag(
            tf.cast(l2_regularization_penalty_factor, dtype=a.dtype))
        broadcasted_shape = prefer_static.concat(
            [batch_shape, [num_model_coefficients, num_model_coefficients]],
            axis=0)
        eye = tf.broadcast_to(eye, broadcasted_shape)
      a_ = tf.concat([a, tf.sqrt(l2_regularizer) * eye], axis=-2)
      b_ = distribution_util.pad(
          b, count=num_model_coefficients, axis=-1, back=True)
      # Return l2_regularizer=0 since its now embedded.
      l2_regularizer_ = np.array(0, dtype_util.as_numpy_dtype(a.dtype))
      return a_, b_, l2_regularizer_

    a, b, l2_regularizer = prefer_static.cond(
        prefer_static.reduce_all([
            prefer_static.logical_or(
                not(fast_unsafe_numerics),
                l2_regularization_penalty_factor is not None),
            l2_regularizer > 0.
        ]),
        _embed_l2_regularization,
        lambda: (a, b, l2_regularizer))

    model_coefficients_next = tf.linalg.lstsq(
        a,
        b[..., tf.newaxis],
        fast=fast_unsafe_numerics,
        l2_regularizer=l2_regularizer,
        name='model_coefficients_next')
    model_coefficients_next = model_coefficients_next[..., 0]

    # TODO(b/79122261): The approach used in `matrix_solve_ls` could be made
    # faster by avoiding explicitly forming Q and instead keeping the
    # factorization in 'implicit' form with stacked (rescaled) Householder
    # vectors underneath the 'R' and then applying the (accumulated)
    # reflectors in the appropriate order to apply Q'. However, we don't
    # presently do this because we lack core TF functionality. For reference,
    # the vanilla QR approach is:
    #   q, r = tf.linalg.qr(a)
    #   c = tf.matmul(q, b, adjoint_a=True)
    #   model_coefficients_next = tf.matrix_triangular_solve(
    #       r, c, lower=False, name='model_coefficients_next')

    predicted_linear_response_next = compute_predicted_linear_response(
        model_matrix,
        model_coefficients_next,
        offset,
        name='predicted_linear_response_next')

    return model_coefficients_next, predicted_linear_response_next


def convergence_criteria_small_relative_norm_weights_change(
    tolerance=1e-5,
    norm_order=2):
  """Returns Python `callable` which indicates fitting procedure has converged.

  Writing old, new `model_coefficients` as `w0`, `w1`, this function
  defines convergence as,

  ```python
  relative_euclidean_norm = (tf.norm(w0 - w1, ord=2, axis=-1) /
                             (1. + tf.norm(w0, ord=2, axis=-1)))
  reduce_all(relative_euclidean_norm < tolerance)
  ```

  where `tf.norm(x, ord=2)` denotes the [Euclidean norm](
  https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm) of `x`.

  Args:
    tolerance: `float`-like `Tensor` indicating convergence, i.e., when
      max relative Euclidean norm weights difference < tolerance`.
      Default value: `1e-5`.
    norm_order: Order of the norm. Default value: `2` (i.e., "Euclidean norm".)

  Returns:
    convergence_criteria_fn: Python `callable` which returns `bool` `Tensor`
      indicated fitting procedure has converged. (See inner function
      specification for argument signature.)
      Default value: `1e-5`.
  """
  def convergence_criteria_fn(
      is_converged_previous,  # pylint: disable=unused-argument
      iter_,
      model_coefficients_previous,
      predicted_linear_response_previous,  # pylint: disable=unused-argument
      model_coefficients_next,
      predicted_linear_response_next,  # pylint: disable=unused-argument
      response,  # pylint: disable=unused-argument
      model,  # pylint: disable=unused-argument
      dispersion):  # pylint: disable=unused-argument
    """Returns `bool` `Tensor` indicating if fitting procedure has converged.

    Args:
      is_converged_previous: "old" convergence results.
      iter_: Iteration number.
      model_coefficients_previous: "old" `model_coefficients`.
      predicted_linear_response_previous: "old" `predicted_linear_response`.
      model_coefficients_next: "new" `model_coefficients`.
      predicted_linear_response_next: "new: `predicted_linear_response`.
      response: (Batch of) vector-shaped `Tensor` where each element represents
        a sample's observed response (to the corresponding row of features).
        Must have same `dtype` as `model_matrix`.
      model: `tfp.glm.ExponentialFamily`-like instance used to construct the
        negative log-likelihood loss, gradient, and expected Hessian (i.e., the
        Fisher information matrix).
      dispersion: `Tensor` representing `response` dispersion, i.e., as in:
        `p(y|theta) := exp((y theta - A(theta)) / dispersion)`. Must broadcast
        with rows of `model_matrix`.
        Default value: `None` (i.e., "no dispersion").

    Returns:
      is_converged: `bool` `Tensor`.
    """
    relative_euclidean_norm = (
        tf.norm(
            tensor=model_coefficients_previous - model_coefficients_next,
            ord=norm_order,
            axis=-1) /
        (1. +
         tf.norm(tensor=model_coefficients_previous, ord=norm_order, axis=-1)))
    return (iter_ > 0) & tf.reduce_all(relative_euclidean_norm < tolerance)

  return convergence_criteria_fn


def prepare_args(model_matrix,
                 response,
                 model_coefficients,
                 predicted_linear_response,
                 offset,
                 name=None):
  """Helper to `fit` which sanitizes input args.

  Args:
    model_matrix: (Batch of) `float`-like, matrix-shaped `Tensor` where each row
      represents a sample's features.
    response: (Batch of) vector-shaped `Tensor` where each element represents a
      sample's observed response (to the corresponding row of features). Must
      have same `dtype` as `model_matrix`.
    model_coefficients: Optional (batch of) vector-shaped `Tensor` representing
      the model coefficients, one for each column in `model_matrix`. Must have
      same `dtype` as `model_matrix`.
      Default value: `tf.zeros(tf.shape(model_matrix)[-1], model_matrix.dtype)`.
    predicted_linear_response: Optional `Tensor` with `shape`, `dtype` matching
      `response`; represents `offset` shifted initial linear predictions based
      on current `model_coefficients`.
      Default value: `offset` if `model_coefficients is None`, and
      `tf.linalg.matvec(model_matrix, model_coefficients_start) + offset`
      otherwise.
    offset: Optional `Tensor` with `shape`, `dtype` matching `response`;
      represents constant shift applied to `predicted_linear_response`.
      Default value: `None` (i.e., `tf.zeros_like(response)`).
    name: Python `str` used as name prefix to ops created by this function.
      Default value: `"prepare_args"`.

  Returns:
    model_matrix: A `Tensor` with `shape`, `dtype` and values of the
      `model_matrix` argument.
    response: A `Tensor` with `shape`, `dtype` and values of the
      `response` argument.
    model_coefficients_start: A `Tensor` with `shape`, `dtype` and
      values of the `model_coefficients_start` argument if specified.
      A (batch of) vector-shaped `Tensors` with `dtype` matching `model_matrix`
      containing the default starting point otherwise.
    predicted_linear_response:  A `Tensor` with `shape`, `dtype` and
      values of the `predicted_linear_response` argument if specified.
      A `Tensor` with `shape`, `dtype` matching `response` containing the
      default value otherwise.
    offset: A `Tensor` with `shape`, `dtype` and values of the `offset` argument
      if specified or `None` otherwise.
  """
  graph_deps = [model_matrix, response, model_coefficients,
                predicted_linear_response, offset]
  with tf.name_scope(name or 'prepare_args'):
    dtype = dtype_util.common_dtype(graph_deps, np.float32)

    model_matrix = tf.convert_to_tensor(
        model_matrix, dtype=dtype, name='model_matrix')

    if offset is not None:
      offset = tf.convert_to_tensor(offset, dtype=dtype, name='offset')

    response = tf.convert_to_tensor(
        response, dtype=dtype, name='response')

    use_default_model_coefficients = model_coefficients is None
    if use_default_model_coefficients:
      # User did not supply model coefficients; assume they're all zero.
      batch_shape = tf.shape(model_matrix)[:-2]
      num_columns = tf.shape(model_matrix)[-1]
      model_coefficients = tf.zeros(
          shape=tf.concat([batch_shape, [num_columns]], axis=0),
          dtype=dtype, name='model_coefficients')
    else:
      # User did supply model coefficients; convert to Tensor in case it's
      # numpy or literal.
      model_coefficients = tf.convert_to_tensor(
          model_coefficients, dtype=dtype, name='model_coefficients')

    if predicted_linear_response is None:
      if use_default_model_coefficients:
        # Since we're using zeros for model_coefficients, we know the predicted
        # linear response will also be all zeros.
        if offset is None:
          predicted_linear_response = tf.zeros_like(
              response, dtype, name='predicted_linear_response')
        else:
          predicted_linear_response = tf.broadcast_to(
              offset,
              tf.shape(response),
              name='predicted_linear_response')
      else:
        # We were given model_coefficients but not the predicted linear
        # response.
        predicted_linear_response = compute_predicted_linear_response(
            model_matrix, model_coefficients, offset)
    else:
      predicted_linear_response = tf.convert_to_tensor(
          predicted_linear_response,
          dtype=dtype,
          name='predicted_linear_response')

  return [
      model_matrix,
      response,
      model_coefficients,
      predicted_linear_response,
      offset,
  ]


def compute_predicted_linear_response(
    model_matrix, model_coefficients, offset=None, name=None):
  """Computes `model_matrix @ model_coefficients + offset`.

  Args:
    model_matrix: (Batch of) `float`-like, matrix-shaped `Tensor` where each row
      represents a sample's features.
    model_coefficients: (Batch of) vector-shaped `Tensor` representing the model
      coefficients, one for each column in `model_matrix`. Must have same
      `dtype` as `model_matrix`.
    offset: Optional `Tensor` representing constant shift applied to
      `predicted_linear_response`.  Must broadcast to `response`.
      Default value: `None` (i.e., `tf.zeros_like(predicted_linear_response)`).
    name: Python `str` used as name prefix to ops created by this function.
      Default value: `None` (i.e., `"compute_predicted_linear_response"`).

  Returns:
    predicted_linear_response: `response`-shaped `Tensor` representing linear
      predictions based on new `model_coefficients`, i.e.,
      `tf.linalg.matvec(model_matrix, model_coefficients) + offset`.
  """
  with tf.name_scope(name or 'compute_predicted_linear_response'):
    if isinstance(model_matrix, (tf.SparseTensor, tf1.SparseTensorValue)):
      matvecmul = sparse_or_dense_matvecmul
    else:
      matvecmul = tf.linalg.matvec
    predicted_linear_response = matvecmul(model_matrix, model_coefficients)
    if offset is not None:
      predicted_linear_response += offset
    return predicted_linear_response


def num_cols(x):
  """Returns number of cols in a given `Tensor`."""
  if tf.compat.dimension_value(x.shape[-1]) is not None:
    return tf.compat.dimension_value(x.shape[-1])
  return tf.shape(x)[-1]
