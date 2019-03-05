<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc.make_simple_step_size_update_policy" />
<meta itemprop="path" content="Stable" />
</div>

# tfp.mcmc.make_simple_step_size_update_policy

``` python
tfp.mcmc.make_simple_step_size_update_policy(
    num_adaptation_steps,
    target_rate=0.75,
    decrement_multiplier=0.01,
    increment_multiplier=0.01,
    step_counter=None
)
```

Create a function implementing a step-size update policy.

The simple policy increases or decreases the `step_size_var` based on the
average of `exp(minimum(0., log_accept_ratio))`. It is based on
[Section 4.2 of Andrieu and Thoms (2008)](
http://www4.ncsu.edu/~rsmith/MA797V_S12/Andrieu08_AdaptiveMCMC_Tutorial.pdf).

The `num_adaptation_steps` argument is set independently of any burnin
for the overall chain. In general, adaptation prevents the chain from
reaching a stationary distribution, so obtaining consistent samples requires
`num_adaptation_steps` be set to a value [somewhat smaller](
http://andrewgelman.com/2017/12/15/burn-vs-warm-iterative-simulation-algorithms/#comment-627745)
than the number of burnin steps. However, it may sometimes be helpful to set
`num_adaptation_steps` to a larger value during development in order to
inspect the behavior of the chain during adaptation.

#### Args:

* <b>`num_adaptation_steps`</b>: Scalar `int` `Tensor` number of initial steps to
    during which to adjust the step size. This may be greater, less than, or
    equal to the number of burnin steps. If `None`, the step size is adapted
    on every step (note this breaks stationarity of the chain!).
* <b>`target_rate`</b>: Scalar `Tensor` representing desired `accept_ratio`.
    Default value: `0.75` (i.e., [center of asymptotically optimal
    rate](https://arxiv.org/abs/1411.6669)).
* <b>`decrement_multiplier`</b>: `Tensor` representing amount to downscale current
    `step_size`.
    Default value: `0.01`.
* <b>`increment_multiplier`</b>: `Tensor` representing amount to upscale current
    `step_size`.
    Default value: `0.01`.
* <b>`step_counter`</b>: Scalar `int` `Variable` specifying the current step. The step
    size is adapted iff `step_counter < num_adaptation_steps`.
    Default value: if `None`, an internal variable
      `step_size_adaptation_step_counter` is created and initialized to `-1`.


#### Returns:

* <b>`step_size_simple_update_fn`</b>: Callable that takes args
    `step_size_var, kernel_results` and returns updated step size(s).