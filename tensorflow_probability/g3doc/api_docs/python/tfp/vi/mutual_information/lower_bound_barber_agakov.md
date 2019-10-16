<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.vi.mutual_information.lower_bound_barber_agakov" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.vi.mutual_information.lower_bound_barber_agakov


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/vi/mutual_information.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Lower bound on mutual information from [Barber and Agakov (2003)][1].

``` python
tfp.vi.mutual_information.lower_bound_barber_agakov(
    logu,
    entropy,
    name=None
)
```



<!-- Placeholder for "Used in" -->

This method gives a lower bound on the mutual information I(X; Y),
by replacing the unknown conditional p(x|y) with a variational
decoder q(x|y), but it requires knowing the entropy of X, h(X).
The lower bound was introduced in [Barber and Agakov (2003)][1].
```none
I(X; Y) = E_p(x, y)[log( p(x|y) / p(x) )]
        = E_p(x, y)[log( q(x|y) / p(x) )] + E_p(y)[KL[ p(x|y) || q(x|y) ]]
        >= E_p(x, y)[log( q(x|y) )] + h(X) = I_[lower_bound_barbar_agakov]
```

#### Example:



`x`, `y` are samples from a joint Gaussian distribution, with correlation
`0.8` and both of dimension `1`.

```python
batch_size, rho, dim = 10000, 0.8, 1
y, eps = tf.split(
    value=tf.random.normal(shape=(2 * batch_size, dim), seed=7),
    num_or_size_splits=2, axis=0)
mean, conditional_stddev = rho * y, tf.sqrt(1. - tf.square(rho))
x = mean + conditional_stddev * eps

# Conditional distribution of p(x|y)
conditional_dist = tfd.MultivariateNormalDiag(
    mean, scale_identity_multiplier=conditional_stddev)

# Scores/unnormalized likelihood of pairs of joint samples `x[i], y[i]`
joint_scores = conditional_dist.log_prob(x)

# Differential entropy of `X` that is `1-D` Normal distributed.
entropy_x = 0.5 * np.log(2 * np.pi * np.e)


# Barber and Agakov lower bound on mutual information
lower_bound_barber_agakov(logu=joint_scores, entropy=entropy_x)
```

#### Args:


* <b>`logu`</b>: `float`-like `Tensor` of size [batch_size] representing
  log(q(x_i | y_i)) for each (x_i, y_i) pair.
* <b>`entropy`</b>: `float`-like `scalar` representing the entropy of X.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this function.
  Default value: `None` (i.e., 'lower_bound_barber_agakov').


#### Returns:


* <b>`lower_bound`</b>: `float`-like `scalar` for lower bound on mutual information.

#### References

[1]: David Barber, Felix V. Agakov. The IM algorithm: a variational
     approach to Information Maximization. In _Conference on Neural
     Information Processing Systems_, 2003.