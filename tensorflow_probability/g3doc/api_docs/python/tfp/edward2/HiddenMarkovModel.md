<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.HiddenMarkovModel" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.HiddenMarkovModel

``` python
tfp.edward2.HiddenMarkovModel(
    *args,
    **kwargs
)
```

Create a random variable for HiddenMarkovModel.

See HiddenMarkovModel for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize hidden Markov model.


#### Args:

* <b>`initial_distribution`</b>: A `Categorical`-like instance.
    Determines probability of first hidden state in Markov chain.
    The number of categories must match the number of categories of
    `transition_distribution` as well as both the rightmost batch
    dimension of `transition_distribution` and the rightmost batch
    dimension of `observation_distribution`.
* <b>`transition_distribution`</b>: A `Categorical`-like instance.
    The rightmost batch dimension indexes the probability distribution
    of each hidden state conditioned on the previous hidden state.
* <b>`observation_distribution`</b>: A <a href="../../tfp/distributions/Distribution.md"><code>tfp.distributions.Distribution</code></a>-like
    instance.  The rightmost batch dimension indexes the distribution
    of each observation conditioned on the corresponding hidden state.
* <b>`num_steps`</b>: The number of steps taken in Markov chain. A python `int`.
* <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
    Default value: `False`.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`, statistics
    (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
    result is undefined. When `False`, an exception is raised if one or
    more of the statistic's batch members are undefined.
    Default value: `True`.
* <b>`name`</b>: Python `str` name prefixed to Ops created by this class.
    Default value: "HiddenMarkovModel".


#### Raises:

* <b>`ValueError`</b>: if `num_steps` is not at least 1.
* <b>`ValueError`</b>: if `initial_distribution` does not have scalar `event_shape`.
* <b>`ValueError`</b>: if `transition_distribution` does not have scalar
    `event_shape.`
* <b>`ValueError`</b>: if `transition_distribution` and `observation_distribution`
    are fully defined but don't have matching rightmost dimension.