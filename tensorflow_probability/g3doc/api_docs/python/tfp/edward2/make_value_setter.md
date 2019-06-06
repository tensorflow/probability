<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.make_value_setter" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.make_value_setter

Creates a value-setting interceptor.

``` python
tfp.edward2.make_value_setter(**model_kwargs)
```



Defined in [`python/edward2/program_transformations.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/edward2/program_transformations.py).

<!-- Placeholder for "Used in" -->

This function creates an interceptor that sets values of Edward2 random
variable objects. This is useful for a range of tasks, including conditioning
on observed data, sampling from posterior predictive distributions, and as a
building block of inference primitives such as computing log joint
probabilities (see examples below).

#### Args:


* <b>`**model_kwargs`</b>: dict of str to Tensor. Keys are the names of random
  variables in the model to which this interceptor is being applied. Values
  are Tensors to set their value to. Variables not included in this dict
  will not be set and will maintain their existing value semantics (by
  default, a sample from the parent-conditional distribution).


#### Returns:


* <b>`set_values`</b>: function that sets the value of intercepted ops.

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
  with ed.interception(make_value_setter(x=x_observed)):
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