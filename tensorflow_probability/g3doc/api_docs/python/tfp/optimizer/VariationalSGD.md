<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.optimizer.VariationalSGD" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="iterations"/>
<meta itemprop="property" content="weights"/>
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

# tfp.optimizer.VariationalSGD

## Class `VariationalSGD`

An optimizer module for constant stochastic gradient descent.





Defined in [`python/optimizer/variational_sgd.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/optimizer/variational_sgd.py).

<!-- Placeholder for "Used in" -->

This implements an optimizer module for the constant stochastic gradient
descent algorithm [(Mandt et al., 2017)][1]. The optimization variable is
regarded as an approximate sample from the posterior .

Note: If a prior is included in the loss, it should be scaled by
`1/num_pseudo_batches`, where num_pseudo_batches is the number of minibatches
in the data.  I.e., it should be divided by the `num_pseudo_batches` term
described below.

#### Args:


* <b>`batch_size`</b>: Scalar `int`-like `Tensor`. The number of examples in a
  minibatch in the data set. Note: Assumes the loss is taken as the mean
  over a minibatch. Otherwise if the sum was taken set this to 1.
* <b>`total_num_examples`</b>: Scalar `int`-like `Tensor`. The total number of examples
  in the data set.
* <b>`max_learning_rate`</b>: Scalar `float`-like `Tensor`. A maximum allowable
  effective coordinate-wise learning rate. The algorithm scales down any
  effective learning rate (i.e. after preconditioning) that is larger than
  this. (Default: `1`)
* <b>`preconditioner_decay_rate`</b>: Scalar `float`-like `Tensor`. The exponential
  decay rate of the rescaling of the preconditioner (RMSprop). (This is
  "alpha" in Mandt et al. (2017)). Should be smaller than but nearly `1` to
  approximate sampling from the posterior. (Default: `0.95`)
* <b>`burnin`</b>: Scalar `int`-like `Tensor`. The number of iterations to collect
  gradient statistics to update the preconditioner before starting to draw
  noisy samples. (Default: `25`)
* <b>`burnin_max_learning_rate`</b>: Scalar `float`-like `Tensor`. Maximum learning
  rate to use during the burnin period.
  (Default: `1e-8`)
* <b>`use_single_learning_rate`</b>: Boolean Indicates whether one single learning
  rate is used or coordinate_wise learning rates are used.
  (Default: `False`)
* <b>`name`</b>: Python `str` describing ops managed by this function.
  (Default: `"VariationalSGD"`)


#### Raises:


* <b>`InvalidArgumentError`</b>: If preconditioner_decay_rate is a `Tensor` not in
  `(0,1]`.

#### References

[1]: Stephan Mandt, Matthew D. Hoffman, and David M. Blei. Stochastic
     Gradient Descent as Approximate Bayesian Inference. _arXiv preprint
     arXiv:1704.04289_, 2017. https://arxiv.org/abs/1704.04289

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    batch_size,
    total_num_examples,
    max_learning_rate=1.0,
    preconditioner_decay_rate=0.95,
    burnin=25,
    burnin_max_learning_rate=1e-06,
    use_single_learning_rate=False,
    name=None
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


<h3 id="weights"><code>weights</code></h3>

Returns variables of this Optimizer based on the order created.




## Methods

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




