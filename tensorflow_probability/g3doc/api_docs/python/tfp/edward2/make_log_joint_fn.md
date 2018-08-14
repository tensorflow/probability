<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.make_log_joint_fn" />
</div>

# tfp.edward2.make_log_joint_fn

``` python
tfp.edward2.make_log_joint_fn(model)
```

Takes Edward probabilistic program and returns its log joint function.

#### Args:

* <b>`model`</b>: Python callable which executes the generative process of a
    computable probability distribution using Edward `RandomVariable`s.


#### Returns:

  A log-joint probability function. Its inputs are `model`'s original inputs
  and random variables which appear during the program execution. Its output
  is a scalar tf.Tensor.

#### Examples

Below we define Bayesian logistic regression as an Edward program, which
represents the model's generative process. We apply `make_log_joint_fn` in
order to alternatively represent the model in terms of its joint probability
function.

```python
from tensorflow_probability import edward2 as ed

def model(X):
  w = ed.Normal(loc=0., scale=1., sample_shape=X.shape[1], name="w")
  y = ed.Normal(loc=tf.tensordot(X, w, [[1], [0]]), scale=0.1, name="y")
  return y

log_joint = ed.make_log_joint_fn(model)

X = tf.random_normal([3, 2])
w_value = tf.random_normal([2])
y_value = tf.random_normal([3])
output = log_joint(X, w=w_value, y=y_value)
```