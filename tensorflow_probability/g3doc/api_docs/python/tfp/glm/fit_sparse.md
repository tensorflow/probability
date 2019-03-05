<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm.fit_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.glm.fit_sparse

``` python
tfp.glm.fit_sparse(
    model_matrix,
    response,
    model,
    model_coefficients_start,
    tolerance,
    l1_regularizer,
    l2_regularizer=None,
    maximum_iterations=None,
    maximum_full_sweeps_per_iteration=1,
    learning_rate=None,
    name=None
)
```

Fits a GLM using coordinate-wise FIM-informed proximal gradient descent.

This function uses a L1- and L2-regularized, second-order quasi-Newton method
to find maximum-likelihood parameters for the given model and observed data.
The second-order approximations use negative Fisher information in place of
the Hessian, that is,

```none
FisherInfo = E_Y[Hessian with respect to model_coefficients of -LogLikelihood(
    Y | model_matrix, current value of model_coefficients)]
```

For large, sparse data sets, `model_matrix` should be supplied as a
`SparseTensor`.

#### Args:

* <b>`model_matrix`</b>: (Batch of) matrix-shaped, `float` `Tensor` or `SparseTensor`
    where each row represents a sample's features.  Has shape `[N, n]` where
    `N` is the number of data samples and `n` is the number of features per
    sample.
* <b>`response`</b>: (Batch of) vector-shaped `Tensor` with the same dtype as
    `model_matrix` where each element represents a sample's observed response
    (to the corresponding row of features).
* <b>`model`</b>: <a href="../../tfp/glm/ExponentialFamily.md"><code>tfp.glm.ExponentialFamily</code></a>-like instance, which specifies the link
    function and distribution of the GLM, and thus characterizes the negative
    log-likelihood which will be minimized. Must have sufficient statistic
    equal to the response, that is, `T(y) = y`.
* <b>`model_coefficients_start`</b>: (Batch of) vector-shaped, `float` `Tensor` with
    the same dtype as `model_matrix`, representing the initial values of the
    coefficients for the GLM regression.  Has shape `[n]` where `model_matrix`
    has shape `[N, n]`.
* <b>`tolerance`</b>: scalar, `float` `Tensor` representing the tolerance for each
    optiization step; see the `tolerance` argument of `fit_sparse_one_step`.
* <b>`l1_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L1
    regularization term.
* <b>`l2_regularizer`</b>: scalar, `float` `Tensor` representing the weight of the L2
    regularization term.
    Default value: `None` (i.e., no L2 regularization).
* <b>`maximum_iterations`</b>: Python integer specifying maximum number of iterations
    of the outer loop of the optimizer (i.e., maximum number of calls to
    `fit_sparse_one_step`).  After this many iterations of the outer loop, the
    algorithm will terminate even if the return value `model_coefficients` has
    not converged.
    Default value: `1`.
* <b>`maximum_full_sweeps_per_iteration`</b>: Python integer specifying the maximum
    number of coordinate descent sweeps allowed in each iteration.
    Default value: `1`.
* <b>`learning_rate`</b>: scalar, `float` `Tensor` representing a multiplicative factor
    used to dampen the proximal gradient descent steps.
    Default value: `None` (i.e., factor is conceptually `1`).
* <b>`name`</b>: Python string representing the name of the TensorFlow operation.
    The default name is `"fit_sparse"`.


#### Returns:

* <b>`model_coefficients`</b>: (Batch of) `Tensor` of the same shape and dtype as
    `model_coefficients_start`, representing the computed model coefficients
    which minimize the regularized negative log-likelihood.
* <b>`is_converged`</b>: scalar, `bool` `Tensor` indicating whether the minimization
    procedure converged across all batches within the specified number of
    iterations.  Here convergence means that an iteration of the inner loop
    (`fit_sparse_one_step`) returns `True` for its `is_converged` output
    value.
* <b>`iter`</b>: scalar, `int` `Tensor` indicating the actual number of iterations of
    the outer loop of the optimizer completed (i.e., number of calls to
    `fit_sparse_one_step` before achieving convergence).

#### Example

```python
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def make_dataset(n, d, link, scale=1., dtype=np.float32):
  model_coefficients = tfd.Uniform(
      low=np.array(-1, dtype), high=np.array(1, dtype)).sample(
          d, seed=42)
  radius = np.sqrt(2.)
  model_coefficients *= radius / tf.linalg.norm(model_coefficients)
  mask = tf.random_shuffle(tf.range(d)) < tf.to_int32(0.5 * tf.to_float(d))
  model_coefficients = tf.where(mask, model_coefficients,
                                tf.zeros_like(model_coefficients))
  model_matrix = tfd.Normal(
      loc=np.array(0, dtype), scale=np.array(1, dtype)).sample(
          [n, d], seed=43)
  scale = tf.convert_to_tensor(scale, dtype)
  linear_response = tf.matmul(model_matrix,
                              model_coefficients[..., tf.newaxis])[..., 0]
  if link == 'linear':
    response = tfd.Normal(loc=linear_response, scale=scale).sample(seed=44)
  elif link == 'probit':
    response = tf.cast(
        tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0,
                   dtype)
  elif link == 'logit':
    response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
* <b>`else`</b>:     raise ValueError('unrecognized true link: {}'.format(link))
  return model_matrix, response, model_coefficients, mask

with tf.Session() as sess:
  x_, y_, model_coefficients_true_, _ = sess.run(make_dataset(
      n=int(1e5), d=100, link='probit'))

  model = tfp.glm.Bernoulli()
  model_coefficients_start = tf.zeros(x_.shape[-1], np.float32)

  model_coefficients, is_converged, num_iter = tfp.glm.fit_sparse(
      model_matrix=tf.convert_to_tensor(x_),
      response=tf.convert_to_tensor(y_),
      model=model,
      model_coefficients_start=model_coefficients_start,
      l1_regularizer=800.,
      l2_regularizer=None,
      maximum_iterations=10,
      maximum_full_sweeps_per_iteration=10,
      tolerance=1e-6,
      learning_rate=None)

  model_coefficients_, is_converged_, num_iter_ = sess.run([
      model_coefficients, is_converged, num_iter])

  print("is_converged:", is_converged_)
  print("    num_iter:", num_iter_)
  print("\nLearned / True")
  print(np.concatenate(
      [[model_coefficients_], [model_coefficients_true_]], axis=0).T)

# ==>
# is_converged: True
#     num_iter: 1
#
# Learned / True
# [[ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.11195257  0.12484948]
#  [ 0.          0.        ]
#  [ 0.05191106  0.06394956]
#  [-0.15090358 -0.15325639]
#  [-0.18187316 -0.18825999]
#  [-0.06140942 -0.07994166]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.14474444  0.15810856]
#  [ 0.          0.        ]
#  [-0.25249591 -0.24260855]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [-0.03888761 -0.06755984]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [-0.0192222  -0.04169233]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.01434913  0.03568212]
#  [-0.11336883 -0.12873614]
#  [ 0.          0.        ]
#  [-0.24496339 -0.24048163]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.04088281  0.06565224]
#  [-0.12784363 -0.13359821]
#  [ 0.05618424  0.07396613]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [ 0.         -0.01719233]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [-0.00076072 -0.03607186]
#  [ 0.21801499  0.21146794]
#  [-0.02161094 -0.04031265]
#  [ 0.0918689   0.10487888]
#  [ 0.0106154   0.03233612]
#  [-0.07817317 -0.09725142]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [-0.23725343 -0.24194022]
#  [ 0.          0.        ]
#  [-0.08725718 -0.1048776 ]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [-0.02114314 -0.04145789]
#  [ 0.          0.        ]
#  [ 0.          0.        ]
#  [-0.02710908 -0.04590397]
#  [ 0.15293184  0.15415154]
#  [ 0.2114463   0.2088728 ]
#  [-0.10969634 -0.12368613]
#  [ 0.         -0.01505797]
#  [-0.01140458 -0.03234904]
#  [ 0.16051085  0.1680062 ]
#  [ 0.09816848  0.11094204]
```

#### References

[1]: Jerome Friedman, Trevor Hastie and Rob Tibshirani. Regularization Paths
     for Generalized Linear Models via Coordinate Descent. _Journal of
     Statistical Software_, 33(1), 2010.
     https://www.jstatsoft.org/article/view/v033i01/v33i01.pdf

[2]: Guo-Xun Yuan, Chia-Hua Ho and Chih-Jen Lin. An Improved GLMNET for
     L1-regularized Logistic Regression. _Journal of Machine Learning
     Research_, 13, 2012.
     http://www.jmlr.org/papers/volume13/yuan12a/yuan12a.pdf