# FunMC

A functional API for creating new Markov Chains.

## Example

```python
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import fun_mc.using_jax as fun_mc

tfb = tfp.bijectors

step_size = 0.2
num_steps = 2000
num_warmup_steps = 1000
num_integrator_steps = 10
num_chains = 16
state = jnp.ones([num_chains, 2])

base_mean = [1., 0]
base_cov = [[1, 0.5], [0.5, 1]]

bijector = tfb.Softplus()
base_dist = tfd.MultivariateNormalFullCovariance(
    loc=base_mean, covariance_matrix=base_cov)
target_dist = bijector(base_dist)

def orig_target_log_prob_fn(x):
  return target_dist.log_prob(x), ()

target_log_prob_fn, state = fun_mc.transform_log_prob_fn(
  orig_target_log_prob_fn, bijector, state)

def kernel(hmc_state, seed):
  hmc_seed, seed = jax.random.split(seed)
  hmc_state, hmc_extra = fun_mc.hamiltonian_monte_carlo_step(
      hmc_state,
      step_size=step_size,
      num_integrator_steps=num_integrator_steps,
      target_log_prob_fn=target_log_prob_fn,
      seed=hmc_seed,
  )
  transformed_state = state.state_extra[0]
  extra = {
    'chain': chain,
    'is_accepted': hmc_extra.is_accepted
  }
  return (hmc_state, seed), extra

_, traced = fun_mc.trace(
    state=fun_mc.hamiltonian_monte_carlo_init(state, target_log_prob_fn),
    fn=kernel,
    num_steps=num_steps,
)

ess = tfp.mcmc.effective_sample_size(
    traced['chain'][num_warmup_steps:],
    cross_chain_dims=1
)
rhat = tfp.mcmc.potential_scale_reduction(
    traced['chain'][num_warmup_steps:],
    split_chains=True
)
p_accept = traced['is_accepted'][num_warmup_steps:].mean()
```

## Installation

```none
pip install -e 'git+https://github.com/tensorflow/probability.git#egg=fun_mc&subdirectory=spinoffs/fun_mc'
```

## Citation

```none
@article{sountsov2021funmc,
  title={FunMC: A functional API for building Markov Chains},
  author={Pavel Sountsov and Alexey Radul and Srinivas Vasudevan},
  year={2020},
  journal={PROBPROG},
}
```
