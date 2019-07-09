<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.glm.ExponentialFamily" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="is_canonical"/>
<meta itemprop="property" content="name"/>
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="log_prob"/>
</div>

# tfp.glm.ExponentialFamily

## Class `ExponentialFamily`

Specifies a mean-value parameterized exponential family.





Defined in [`python/glm/family.py`](https://github.com/tensorflow/probability/tree/master/tensorflow_probability/python/glm/family.py).

<!-- Placeholder for "Used in" -->

Subclasses implement [exponential-family distribution](
https://en.wikipedia.org/wiki/Exponential_family) properties (e.g.,
`log_prob`, `variance`) as a function of a real-value which is transformed via
some [link function](
https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)
to be interpreted as the distribution's mean. The distribution is
parameterized by this mean, i.e., "mean-value parametrized."

Subclasses are typically used to specify a Generalized Linear Model (GLM). A
[GLM]( https://en.wikipedia.org/wiki/Generalized_linear_model) is a
generalization of linear regression which enables efficient fitting of
log-likelihood losses beyond just assuming `Normal` noise. See <a href="../../tfp/glm/fit.md"><code>tfp.glm.fit</code></a>
for more details.

Subclasses must implement `_call`, `_log_prob`, and `_is_canonical`. In
context of <a href="../../tfp/glm/fit.md"><code>tfp.glm.fit</code></a>, these functions are used to find the best fitting
weights for given model matrix ("X") and responses ("Y").

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(name=None)
```

Creates the ExponentialFamily.


#### Args:


* <b>`name`</b>: Python `str` used as TF namescope for ops created by member
  functions. Default value: `None` (i.e., the subclass name).



## Properties

<h3 id="is_canonical"><code>is_canonical</code></h3>

Returns `True` when `variance(r) == grad_mean(r)` for all `r`.


<h3 id="name"><code>name</code></h3>

Returns TF namescope prefixed to ops created by member functions.




## Methods

<h3 id="__call__"><code>__call__</code></h3>

``` python
__call__(
    predicted_linear_response,
    name=None
)
```

Computes `mean(r), var(mean), d/dr mean(r)` for linear response, `r`.

Here `mean` and `var` are the mean and variance of the sufficient statistic,
which may not be the same as the mean and variance of the random variable
itself.  If the distribution's density has the form

```none
p_Y(y) = h(y) Exp[dot(theta, T(y)) - A]
```

where `theta` and `A` are constants and `h` and `T` are known functions,
then `mean` and `var` are the mean and variance of `T(Y)`.  In practice,
often `T(Y) := Y` and in that case the distinction doesn't matter.

#### Args:


* <b>`predicted_linear_response`</b>: `float`-like `Tensor` corresponding to
  `tf.matmul(model_matrix, weights)`.
* <b>`name`</b>: Python `str` used as TF namescope for ops created by member
  functions. Default value: `None` (i.e., 'call').


#### Returns:


* <b>`mean`</b>: `Tensor` with shape and dtype of `predicted_linear_response`
  representing the distribution prescribed mean, given the prescribed
  linear-response to mean mapping.
* <b>`variance`</b>: `Tensor` with shape and dtype of `predicted_linear_response`
  representing the distribution prescribed variance, given the prescribed
  linear-response to mean mapping.
* <b>`grad_mean`</b>: `Tensor` with shape and dtype of `predicted_linear_response`
  representing the gradient of the mean with respect to the
  linear-response and given the prescribed linear-response to mean
  mapping.

<h3 id="log_prob"><code>log_prob</code></h3>

``` python
log_prob(
    response,
    predicted_linear_response,
    name=None
)
```

Computes `D(param=mean(r)).log_prob(response)` for linear response, `r`.


#### Args:


* <b>`response`</b>: `float`-like `Tensor` representing observed ("actual")
  responses.
* <b>`predicted_linear_response`</b>: `float`-like `Tensor` corresponding to
  `tf.matmul(model_matrix, weights)`.
* <b>`name`</b>: Python `str` used as TF namescope for ops created by member
  functions. Default value: `None` (i.e., 'log_prob').


#### Returns:


* <b>`log_prob`</b>: `Tensor` with shape and dtype of `predicted_linear_response`
  representing the distribution prescribed log-probability of the observed
  `response`s.



