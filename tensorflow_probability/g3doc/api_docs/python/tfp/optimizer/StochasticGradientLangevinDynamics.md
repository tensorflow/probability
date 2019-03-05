<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.StochasticGradientLangevinDynamics" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="variable_scope"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="compute_gradients"/>
<meta itemprop="property" content="get_name"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="GATE_GRAPH"/>
<meta itemprop="property" content="GATE_NONE"/>
<meta itemprop="property" content="GATE_OP"/>
</div>

# tfp.optimizer.StochasticGradientLangevinDynamics

## Class `StochasticGradientLangevinDynamics`



An optimizer module for stochastic gradient Langevin dynamics.

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
  var_1 = tf.get_variable(
      'var_1', initializer=[1., 1.])
  var_2 = tf.get_variable(
      'var_2', initializer=[1.])

  var = tf.concat([var_1, var_2], axis=-1)
  # Partially defined loss function
  loss_part = tf.cholesky_solve(chol, tf.expand_dims(var, -1))
  # Loss function
  loss = 0.5 * tf.linalg.matvec(loss_part, var, transpose_a=True)

  # Set up the learning rate with a polynomial decay
  global_step = tf.Variable(0, trainable=False)
  starter_learning_rate = .3
  end_learning_rate = 1e-4
  decay_steps = 1e4
  learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                            global_step, decay_steps,
                                            end_learning_rate, power=1.)

  # Set up the optimizer
  optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(
      learning_rate=learning_rate, preconditioner_decay_rate=0.99)

  optimizer = optimizer_kernel.minimize(loss)

  init = tf.global_variables_initializer()
  # Number of training steps
  training_steps = 5000
  # Record the steps as and treat them as samples
  samples = [np.zeros([training_steps, 2]), np.zeros([training_steps, 1])]
  sess.run(init)
  for step in range(training_steps):
    sess.run([optimizer, loss])
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
* <b>`variable_scope`</b>: Variable scope used for calls to `tf.get_variable`.
    If `None`, a new variable scope is created using name
    `tf.get_default_graph().unique_name(name or default_name)`.


#### Raises:

* <b>`InvalidArgumentError`</b>: If preconditioner_decay_rate is a `Tensor` not in
    `(0,1]`.
* <b>`NotImplementedError`</b>: If eager execution is enabled.

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
    parallel_iterations=10,
    variable_scope=None
)
```





## Properties

<h3 id="variable_scope"><code>variable_scope</code></h3>

Variable scope of all calls to `tf.get_variable`.



## Methods

<h3 id="apply_gradients"><code>apply_gradients</code></h3>

``` python
apply_gradients(
    grads_and_vars,
    global_step=None,
    name=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:

* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs as returned by
    `compute_gradients()`.
* <b>`global_step`</b>: Optional `Variable` to increment by one after the
    variables have been updated.
* <b>`name`</b>: Optional name for the returned operation.  Default to the
    name passed to the `Optimizer` constructor.


#### Returns:

An `Operation` that applies the specified gradients. If `global_step`
was not None, that operation also increments `global_step`.


#### Raises:

* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.
* <b>`RuntimeError`</b>: If you should use `_distributed_apply()` instead.

<h3 id="compute_gradients"><code>compute_gradients</code></h3>

``` python
compute_gradients(
    loss,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    grad_loss=None
)
```

Compute gradients of `loss` for the variables in `var_list`.

This is the first part of `minimize()`.  It returns a list
of (gradient, variable) pairs where "gradient" is the gradient
for "variable".  Note that "gradient" can be a `Tensor`, an
`IndexedSlices`, or `None` if there is no gradient for the
given variable.

#### Args:

* <b>`loss`</b>: A Tensor containing the value to minimize or a callable taking
    no arguments which returns the value to minimize. When eager execution
    is enabled it must be a callable.
* <b>`var_list`</b>: Optional list or tuple of `tf.Variable` to update to minimize
    `loss`.  Defaults to the list of variables collected in the graph
    under the key `GraphKeys.TRAINABLE_VARIABLES`.
* <b>`gate_gradients`</b>: How to gate the computation of gradients.  Can be
    `GATE_NONE`, `GATE_OP`, or `GATE_GRAPH`.
* <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Valid values are defined in the class `AggregationMethod`.
* <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with
    the corresponding op.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.


#### Returns:

A list of (gradient, variable) pairs. Variable is always present, but
gradient can be `None`.


#### Raises:

* <b>`TypeError`</b>: If `var_list` contains anything else than `Variable` objects.
* <b>`ValueError`</b>: If some arguments are invalid.
* <b>`RuntimeError`</b>: If called with eager execution enabled and `loss` is
    not callable.



#### Eager Compatibility
When eager execution is enabled, `gate_gradients`, `aggregation_method`,
and `colocate_gradients_with_ops` are ignored.



<h3 id="get_name"><code>get_name</code></h3>

``` python
get_name()
```



<h3 id="get_slot"><code>get_slot</code></h3>

``` python
get_slot(
    var,
    name
)
```

Return a slot named `name` created for `var` by the Optimizer.

Some `Optimizer` subclasses use additional variables.  For example
`Momentum` and `Adagrad` use variables to accumulate updates.  This method
gives access to these `Variable` objects if for some reason you need them.

Use `get_slot_names()` to get the list of slot names created by the
`Optimizer`.

#### Args:

* <b>`var`</b>: A variable passed to `minimize()` or `apply_gradients()`.
* <b>`name`</b>: A string.


#### Returns:

The `Variable` for the slot if it was created, `None` otherwise.

<h3 id="get_slot_names"><code>get_slot_names</code></h3>

``` python
get_slot_names()
```

Return a list of the names of slots created by the `Optimizer`.

See `get_slot()`.

#### Returns:

A list of strings.

<h3 id="minimize"><code>minimize</code></h3>

``` python
minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)
```

Add operations to minimize `loss` by updating `var_list`.

This method simply combines calls `compute_gradients()` and
`apply_gradients()`. If you want to process the gradient before applying
them call `compute_gradients()` and `apply_gradients()` explicitly instead
of using this function.

#### Args:

* <b>`loss`</b>: A `Tensor` containing the value to minimize.
* <b>`global_step`</b>: Optional `Variable` to increment by one after the
    variables have been updated.
* <b>`var_list`</b>: Optional list or tuple of `Variable` objects to update to
    minimize `loss`.  Defaults to the list of variables collected in
    the graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
* <b>`gate_gradients`</b>: How to gate the computation of gradients.  Can be
    `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
* <b>`aggregation_method`</b>: Specifies the method used to combine gradient terms.
    Valid values are defined in the class `AggregationMethod`.
* <b>`colocate_gradients_with_ops`</b>: If True, try colocating gradients with
    the corresponding op.
* <b>`name`</b>: Optional name for the returned operation.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.


#### Returns:

An Operation that updates the variables in `var_list`.  If `global_step`
was not `None`, that operation also increments `global_step`.


#### Raises:

* <b>`ValueError`</b>: If some of the variables are not `Variable` objects.



#### Eager Compatibility
When eager execution is enabled, `loss` should be a Python function that
takes no arguments and computes the value to be minimized. Minimization (and
gradient computation) is done with respect to the elements of `var_list` if
not None, else with respect to any trainable variables created during the
execution of the `loss` function. `gate_gradients`, `aggregation_method`,
`colocate_gradients_with_ops` and `grad_loss` are ignored when eager
execution is enabled.



<h3 id="variables"><code>variables</code></h3>

``` python
variables()
```

A list of variables which encode the current state of `Optimizer`.

Includes slot variables and additional global variables created by the
optimizer in the current default graph.

#### Returns:

A list of variables.



## Class Members

<h3 id="GATE_GRAPH"><code>GATE_GRAPH</code></h3>

<h3 id="GATE_NONE"><code>GATE_NONE</code></h3>

<h3 id="GATE_OP"><code>GATE_OP</code></h3>

