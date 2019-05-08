<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.StochasticGradientLangevinDynamics" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="iterations"/>
<meta itemprop="property" content="variable_scope"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__getattribute__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_slot"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_gradients"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="get_updates"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="variables"/>
</div>

# tfp.optimizer.StochasticGradientLangevinDynamics

## Class `StochasticGradientLangevinDynamics`

An optimizer module for stochastic gradient Langevin dynamics.





Defined in [`python/optimizer/sgld.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer/sgld.py).

<!-- Placeholder for "Used in" -->

This implements the preconditioned Stochastic Gradient Langevin Dynamics
optimizer [(Li et al., 2016)][1]. The optimization variable is regarded as a
sample from the posterior under Stochastic Gradient Langevin Dynamics with
noise rescaled in each dimension according to [RMSProp](
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).

Note: If a prior is included in the loss, it should be scaled by
`1/data_size`, where `data_size` is the number of points in the data set.
I.e., it should be divided by the `data_size` term described below.

#### Examples

##### Optimizing energy of a 3D-Gaussian distribution

This example demonstrates that for a fixed step size SGLD works as an
approximate version of MALA (tfp.mcmc.MetropolisAdjustedLangevinAlgorithm).

```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
dtype = np.float32

with tf.Session(graph=tf.Graph()) as sess:
  # Set up random seed for the optimizer
  tf.set_random_seed(42)
  true_mean = dtype([0, 0, 0])
  true_cov = dtype([[1, 0.25, 0.25], [0.25, 1, 0.25], [0.25, 0.25, 1]])
  # Loss is defined through the Cholesky decomposition
  chol = tf.linalg.cholesky(true_cov)

  var_1 = tf.compat.v2.Variable(name='var_1', initial_value=[1., 1.])
  var_2 = tf.compat.v2.Variable(name='var_2', initial_value=[1.])

  def loss_fn():
    var = tf.concat([var_1, var_2], axis=-1)
    loss_part = tf.linalg.cholesky_solve(chol, tf.expand_dims(var, -1))
    return tf.linalg.matvec(loss_part, var, transpose_a=True)

  # Set up the learning rate with a polynomial decay
  step = tf.compat.v2.Variable(0, dtype=tf.int64)
  starter_learning_rate = .3
  end_learning_rate = 1e-4
  decay_steps = 1e4
  learning_rate = tf.compat.v1.train.polynomial_decay(
      starter_learning_rate,
      step,
      decay_steps,
      end_learning_rate,
      power=1.)

  # Set up the optimizer
  optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
      learning_rate=learning_rate, preconditioner_decay_rate=0.99)
  optimizer_kernel.iterations = step
  optimizer = optimizer_kernel.minimize(loss_fn, var_list=[var_1, var_2])

  # Number of training steps
  training_steps = 5000
  # Record the steps as and treat them as samples
  samples = [np.zeros([training_steps, 2]), np.zeros([training_steps, 1])]
  sess.run(tf.compat.v1.global_variables_initializer())
  for step in range(training_steps):
    sess.run(optimizer)
    sample = [sess.run(var_1), sess.run(var_2)]
    samples[0][step, :] = sample[0]
    samples[1][step, :] = sample[1]

  samples_ = np.concatenate(samples, axis=-1)
  sample_mean = np.mean(samples_, 0)
  print('sample mean', sample_mean)
```
#### Args:

* <b>`learning_rate`</b>: Scalar `float`-like `Tensor`. The base learning rate for the
  optimizer. Must be tuned to the specific function being minimized.
* <b>`preconditioner_decay_rate`</b>: Scalar `float`-like `Tensor`. The exponential
  decay rate of the rescaling of the preconditioner (RMSprop). (This is
  "alpha" in Li et al. (2016)). Should be smaller than but nearly `1` to
  approximate sampling from the posterior. (Default: `0.95`)
* <b>`data_size`</b>: Scalar `int`-like `Tensor`. The effective number of
  points in the data set. Assumes that the loss is taken as the mean over a
  minibatch. Otherwise if the sum was taken, divide this number by the
  batch size. If a prior is included in the loss function, it should be
  normalized by `data_size`. Default value: `1`.
* <b>`burnin`</b>: Scalar `int`-like `Tensor`. The number of iterations to collect
  gradient statistics to update the preconditioner before starting to draw
  noisy samples. (Default: `25`)
* <b>`diagonal_bias`</b>: Scalar `float`-like `Tensor`. Term added to the diagonal of
  the preconditioner to prevent the preconditioner from degenerating.
  (Default: `1e-8`)
* <b>`name`</b>: Python `str` describing ops managed by this function.
  (Default: `"StochasticGradientLangevinDynamics"`)
* <b>`parallel_iterations`</b>: the number of coordinates for which the gradients of
    the preconditioning matrix can be computed in parallel. Must be a
    positive integer.


#### Raises:

  InvalidArgumentError: If preconditioner_decay_rate is a `Tensor` not in
    `(0,1]`.
  NotImplementedError: If eager execution is enabled.

#### References

[1]: Chunyuan Li, Changyou Chen, David Carlson, and Lawrence Carin.
     Preconditioned Stochastic Gradient Langevin Dynamics for Deep Neural
     Networks. In _Association for the Advancement of Artificial
     Intelligence_, 2016. https://arxiv.org/abs/1512.07666

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    learning_rate,
    preconditioner_decay_rate=0.95,
    data_size=1,
    burnin=25,
    diagonal_bias=1e-08,
    name=None,
    parallel_iterations=10
)
```

Create a new Optimizer.

This must be called by the constructors of subclasses.
Note that Optimizer instances should not bind to a single graph,
and so shouldn't keep Tensors as member variables. Generally
you should be able to use the _set_hyper()/state.get_hyper()
facility instead.

This class in stateful and thread-compatible.

#### Args:

* <b>`name`</b>: A non-empty string.  The name to use for accumulators created
  for the optimizer.
* <b>`**kwargs`</b>: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
  `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
  gradients by value, `decay` is included for backward compatibility to
  allow time inverse decay of learning rate. `lr` is included for backward
  compatibility, recommended to use `learning_rate` instead.


#### Raises:

* <b>`ValueError`</b>: If name is malformed.
* <b>`RuntimeError`</b>: If _create_slots has been overridden instead of
    _create_vars.



## Properties

<h3 id="iterations"><code>iterations</code></h3>

Variable. The number of training steps this Optimizer has run.

<h3 id="variable_scope"><code>variable_scope</code></h3>

Variable scope of all calls to `tf.get_variable`.

<h3 id="weights"><code>weights</code></h3>

Returns variables of this Optimizer based on the order created.



## Methods

<h3 id="__getattribute__"><code>__getattribute__</code></h3>

``` python
__getattribute__(name)
```

Overridden to support hyperparameter access.

<h3 id="add_slot"><code>add_slot</code></h3>

``` python
add_slot(
    var,
    slot_name,
    initializer='zeros'
)
```

Add a new slot variable for `var`.

<h3 id="add_weight"><code>add_weight</code></h3>

``` python
add_weight(
    name,
    shape,
    dtype=None,
    initializer='zeros',
    trainable=None,
    synchronization=tf_variables.VariableSynchronization.AUTO,
    aggregation=tf_variables.VariableAggregation.NONE
)
```



<h3 id="apply_gradients"><code>apply_gradients</code></h3>

``` python
apply_gradients(
    grads_and_vars,
    name=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:

* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs.
* <b>`name`</b>: Optional name for the returned operation.  Default to the name
  passed to the `Optimizer` constructor.


#### Returns:

An `Operation` that applies the specified gradients. If `global_step`
was not None, that operation also increments `global_step`.


#### Raises:

* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.

<h3 id="from_config"><code>from_config</code></h3>

``` python
from_config(
    cls,
    config,
    custom_objects=None
)
```

Creates an optimizer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same optimizer from the config
dictionary.

#### Arguments:

* <b>`config`</b>: A Python dictionary, typically the output of get_config.
* <b>`custom_objects`</b>: A Python dictionary mapping names to additional Python
  objects used to create this optimizer, such as a function used for a
  hyperparameter.


#### Returns:

An optimizer instance.

<h3 id="get_config"><code>get_config</code></h3>

``` python
get_config()
```

Returns the config of the optimimizer.

An optimizer config is a Python dictionary (serializable)
containing the configuration of an optimizer.
The same optimizer can be reinstantiated later
(without any saved state) from this configuration.

#### Returns:

Python dictionary.

<h3 id="get_gradients"><code>get_gradients</code></h3>

``` python
get_gradients(
    loss,
    params
)
```

Returns gradients of `loss` with respect to `params`.

#### Arguments:

* <b>`loss`</b>: Loss tensor.
* <b>`params`</b>: List of variables.


#### Returns:

List of gradient tensors.


#### Raises:

* <b>`ValueError`</b>: In case any gradient cannot be computed (e.g. if gradient
  function not implemented).

<h3 id="get_slot"><code>get_slot</code></h3>

``` python
get_slot(
    var,
    slot_name
)
```



<h3 id="get_slot_names"><code>get_slot_names</code></h3>

``` python
get_slot_names()
```

A list of names for this optimizer's slots.

<h3 id="get_updates"><code>get_updates</code></h3>

``` python
get_updates(
    loss,
    params
)
```



<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```



<h3 id="minimize"><code>minimize</code></h3>

``` python
minimize(
    loss,
    var_list,
    grad_loss=None,
    name=None
)
```

Minimize `loss` by updating `var_list`.

This method simply computes gradient using `tf.GradientTape` and calls
`apply_gradients()`. If you want to process the gradient before applying
then call `tf.GradientTape` and `apply_gradients()` explicitly instead
of using this function.

#### Args:

* <b>`loss`</b>: A callable taking no arguments which returns the value to minimize.
* <b>`var_list`</b>: list or tuple of `Variable` objects to update to minimize
  `loss`, or a callable returning the list or tuple of `Variable` objects.
  Use callable when the variable list would otherwise be incomplete before
  `minimize` since the variables are created at the first time `loss` is
  called.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.
* <b>`name`</b>: Optional name for the returned operation.


#### Returns:

An Operation that updates the variables in `var_list`.  If `global_step`
was not `None`, that operation also increments `global_step`.


#### Raises:

* <b>`ValueError`</b>: If some of the variables are not `Variable` objects.

<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```



<h3 id="variables"><code>variables</code></h3>

``` python
variables()
```

Returns variables of this Optimizer based on the order created.



