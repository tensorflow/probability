<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.edward2.LinearGaussianStateSpaceModel" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.edward2.LinearGaussianStateSpaceModel

``` python
tfp.edward2.LinearGaussianStateSpaceModel(
    *args,
    **kwargs
)
```

Create a random variable for LinearGaussianStateSpaceModel.

See LinearGaussianStateSpaceModel for more details.

#### Returns:

  RandomVariable.

#### Original Docstring for Distribution

Initialize a `LinearGaussianStateSpaceModel.


#### Args:

* <b>`num_timesteps`</b>: Integer `Tensor` total number of timesteps.
* <b>`transition_matrix`</b>: A transition operator, represented by a Tensor or
    LinearOperator of shape `[latent_size, latent_size]`, or by a
    callable taking as argument a scalar integer Tensor `t` and
    returning a Tensor or LinearOperator representing the transition
    operator from latent state at time `t` to time `t + 1`.
* <b>`transition_noise`</b>: An instance of
    `tfd.MultivariateNormalLinearOperator` with event shape
    `[latent_size]`, representing the mean and covariance of the
    transition noise model, or a callable taking as argument a
    scalar integer Tensor `t` and returning such a distribution
    representing the noise in the transition from time `t` to time `t + 1`.
* <b>`observation_matrix`</b>: An observation operator, represented by a Tensor
    or LinearOperator of shape `[observation_size, latent_size]`,
    or by a callable taking as argument a scalar integer Tensor
    `t` and returning a timestep-specific Tensor or
    LinearOperator.
* <b>`observation_noise`</b>: An instance of
    `tfd.MultivariateNormalLinearOperator` with event shape
    `[observation_size]`, representing the mean and covariance of
    the observation noise model, or a callable taking as argument
    a scalar integer Tensor `t` and returning a timestep-specific
    noise model.
* <b>`initial_state_prior`</b>: An instance of `MultivariateNormalLinearOperator`
    representing the prior distribution on latent states; must
    have event shape `[latent_size]`.
* <b>`initial_step`</b>: optional `int` specifying the time of the first
    modeled timestep.  This is added as an offset when passing
    timesteps `t` to (optional) callables specifying
    timestep-specific transition and observation models.
* <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
    with asserts. If `validate_args` is `False`, and the inputs are
    invalid, correct behavior is not guaranteed.
* <b>`allow_nan_stats`</b>: Python `bool`, default `True`. If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
* <b>`name`</b>: The name to give Ops created by the initializer.