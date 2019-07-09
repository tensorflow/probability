<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.make_log_joint_fn" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.make_log_joint_fn

Takes Edward probabilistic program and returns its log joint function.

``` python
tfp.edward2.make_log_joint_fn(model)
```



Defined in [`python/edward2/program_transformations.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/program_transformations.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`model`</b>: Python callable which executes the generative process of a
  computable probability distribution using `ed.RandomVariable`s.


#### Returns:

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

features = tf.random_normal([3, 2])
coeffs_value = tf.random_normal([2])
outcomes_value = tf.round(tf.random_uniform([3]))
output = log_joint(features, coeffs=coeffs_value, outcomes=outcomes_value)
```