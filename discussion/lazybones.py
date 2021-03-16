# Copyright 2021 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# -*- coding: utf-8 -*-
# pylint: skip-file
"""[LazyBones] End to End modeling example
"""

import tensorflow_probability.python.experimental.lazybones as lb

DeferredInput = lb.DeferredInput
Deferred = lb.Deferred
DeferredScope = lb.DeferredScope

UNKNOWN = lb.UNKNOWN

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

# %matplotlib inline

"""# A simple regression with pure numpy and scipy"""

import scipy as sp

sp = DeferredInput(sp)

#@title Data simulation
n_feature = 15
n_obs = 1000

hyper_mu_ = np.array(10.)
hyper_sigma_ = np.array(2.)
sigma_ = np.array(1.5)

beta_ = hyper_mu_ + hyper_sigma_ * np.random.randn(n_feature)
design_matrix = np.random.rand(n_obs, n_feature)
y_ = design_matrix @ beta_ + np.random.randn(n_obs) * sigma_

#@title LazyBones model
hyper_mu = sp.stats.norm(0., 100.).rvs()
hyper_sigma = sp.stats.halfnorm(0., 5.).rvs()
beta = sp.stats.norm(hyper_mu, hyper_sigma).rvs(n_feature)
y_hat = sp.matmul(design_matrix, beta)
sigma = sp.stats.halfnorm(0., 5.).rvs()
y = sp.stats.norm(y_hat, sigma).rvs()

"""### Inference with MAP"""

def target_log_prob_fn(*values):
  return lb.utils.distribution_measure(
      vertexes=[hyper_mu, hyper_sigma, beta, sigma, y],
      values=[*values, y_],
      get_attr_fn=lambda dist: dist.logpdf,
      combine=sum,
      reduce_op=np.sum)

loss_fn = lambda x: -target_log_prob_fn(x[0], np.exp(x[1]), x[2:-1], np.exp(x[-1]))

x = np.concatenate([hyper_mu_[None], np.log(hyper_sigma_[None]), beta_, np.log(sigma_[None])])
loss_fn(x)

output = sp.optimize.minimize(
    loss_fn, 
    np.random.randn(n_feature+3),
    method='L-BFGS-B')

est_x = output.x.eval() # <== actually evoke the computation
loss_fn(est_x)

_, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, est_x, 'o', label='MAP');
ax.plot(beta_, np.linalg.lstsq(design_matrix, y_, rcond=None)[0], 'o', label='LSTSQ')
ax.legend();

"""# Mixture regression model (using Jax)"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

jaxw = DeferredInput(jax)
jnpw = DeferredInput(jnp)
tfpw = DeferredInput(tfp)
tfdw = DeferredInput(tfp.distributions)

#@title Set up data.
predictors = np.asarray([
    201., 244., 47., 287., 203., 58., 210., 202., 198., 158., 165., 201.,
    157., 131., 166., 160., 186., 125., 218., 146.
])
obs = np.asarray([
    592., 401., 583., 402., 495., 173., 479., 504., 510., 416., 393., 442.,
    317., 311., 400., 337., 423., 334., 533., 344.
])
y_sigma = np.asarray([
    61., 25., 38., 15., 21., 15., 27., 14., 30., 16., 14., 25., 52., 16.,
    34., 31., 42., 26., 16., 22.
])
y_sigma = y_sigma / (2 * obs.std(axis=0))
obs = (obs - obs.mean(axis=0)) / (2 * obs.std(axis=0))
predictors = (predictors - predictors.mean(axis=0)) / (2 * predictors.std(axis=0))

#@title LazyBones model
nobs = len(y_sigma)
seed = jax.random.PRNGKey(10)
seed, *rv_seed = jax.random.split(seed, 7)
# Priors
b0 = tfdw.Normal(loc=0., scale=10.).sample(seed=rv_seed[0])
b1 = tfdw.Normal(loc=0., scale=10.).sample(seed=rv_seed[1])
mu_out = tfdw.Normal(loc=0., scale=10.).sample(seed=rv_seed[2])
sigma_out = tfdw.HalfNormal(scale=1.).sample(seed=rv_seed[3])
weight = tfdw.Uniform(low=0., high=.5).sample(seed=rv_seed[4])
# Likelihood
# note we are constructing components as distributions but not RV
mixture_dist = tfdw.Categorical(
    probs=jnpw.repeat(
        jnpw.array([1-weight, weight])[None, ...], nobs, axis=0))
component_dist = tfdw.Normal(
    loc=jnpw.stack([b0 + b1*predictors,
                    jnpw.repeat(mu_out, nobs)]).T,
    scale=jnpw.stack([y_sigma,
                      sigma_out + y_sigma]).T)
observed = tfdw.Independent(
    tfdw.MixtureSameFamily(mixture_dist, component_dist), 1).sample(seed=rv_seed[5])
# Posterior
target_log_prob_fn = lambda *values: lb.utils.log_prob(
    vertexes=[b0, b1, mu_out, sigma_out, weight, observed],
    values=[*values, obs])

#@title JointDistributionSequential model
def gen_mixturemodel(X, sigma, hyper_mean=0., hyper_scale=10.):
    nobs = len(sigma)
    return tfd.JointDistributionSequential([
      tfd.Normal(loc=hyper_mean, scale=hyper_scale),
      tfd.Normal(loc=hyper_mean, scale=hyper_scale),
      tfd.Normal(loc=hyper_mean, scale=10.),
      tfd.HalfNormal(scale=1.),
      tfd.Uniform(low=0., high=.5),
      lambda weight, sigma_out, mu_out, b1, b0: tfd.Independent(
          tfd.MixtureSameFamily(
              tfd.Categorical(
                  probs=jnp.repeat(
                      jnp.array([1-weight, weight])[None, ...], nobs, axis=0)),
              tfd.Normal(
                  loc=jnp.stack([b0 + b1*X, jnp.repeat(mu_out, nobs)]).T,
                  scale=jnp.stack([sigma, sigma+sigma_out]).T)
        ), 1)
    ], validate_args=True)

mdl_mixture = gen_mixturemodel(predictors, y_sigma)

values = mdl_mixture.sample(seed=seed)[:-1]
assert mdl_mixture.log_prob(*values, obs) == target_log_prob_fn(*values)

# Out of order RV list also works
target_log_prob_fn_ = lambda *values: lb.utils.log_prob(
    vertexes=[b1, mu_out, sigma_out, weight, b0, observed],
    values=[*values, obs])
values = mdl_mixture.sample(seed=seed)[:-1]
assert mdl_mixture.log_prob(*values, obs) == target_log_prob_fn_(*values[1:], values[0])

"""## Inference with MCMC"""

sample_fn = jax.vmap(lambda seed: mdl_mixture.sample(seed=seed))
log_prob_fn = jax.vmap(target_log_prob_fn)

init_state = sample_fn(jax.random.split(seed, 5))[:-1]
_ = tfp.math.value_and_gradient(log_prob_fn, init_state)

#@title Sample with NUTS
from tensorflow_probability.python.internal import unnest

def gen_nuts_sample_fn(target_log_prob_fn, bijector, draws, tune):

  @jax.jit
  def run_inference_nuts(init_state, seed):
    seed, tuning_seed, sample_seed = jax.random.split(seed, 3)
    def gen_kernel(step_size):
      hmc = tfp.mcmc.NoUTurnSampler(
          target_log_prob_fn=target_log_prob_fn, step_size=step_size)
      hmc = tfp.mcmc.TransformedTransitionKernel(
          hmc, bijector=bijector)
      tuning_hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
          hmc, tune // 2, target_accept_prob=0.85)
      return tuning_hmc

    def tuning_trace_fn(_, pkr): 
      return (pkr.inner_results.transformed_state,
              pkr.new_step_size)

    def get_tuned_stepsize(samples, step_size):
      return jnp.std(samples, axis=0) * step_size[-1]

    step_size = jax.tree_map(lambda x: jnp.ones_like(x), init_state)
    tuning_hmc = gen_kernel(step_size)
    init_samples, tuning_result = tfp.mcmc.sample_chain(
        num_results=200,
        num_burnin_steps=tune // 2 - 200,
        current_state=init_state,
        kernel=tuning_hmc,
        trace_fn=tuning_trace_fn,
        seed=tuning_seed)
    
    step_size_new = jax.tree_multimap(get_tuned_stepsize, *tuning_result)
    sample_hmc = gen_kernel(step_size_new)
    def sample_trace_fn(_, pkr):
      return (
          unnest.get_innermost(pkr, 'target_log_prob'),
          unnest.get_innermost(pkr, 'leapfrogs_taken'),
          unnest.get_innermost(pkr, 'has_divergence'),
          unnest.get_innermost(pkr, 'energy'),
          unnest.get_innermost(pkr, 'log_accept_ratio'),
          unnest.get_innermost(pkr, 'reach_max_depth'),
      )
    return tfp.mcmc.sample_chain(
        num_results=draws,
        num_burnin_steps=tune // 2,
        current_state=[x[-1] for x in init_samples],
        kernel=sample_hmc,
        trace_fn=sample_trace_fn,
        seed=sample_seed)
  
  return run_inference_nuts

#@title Sample with CheeS
from tensorflow_probability.python.internal import unnest

def gen_chess_sample_fn(target_log_prob_fn, bijector, draws, tune):

  @jax.jit
  def run_inference_chess(init_state, seed, step_size=.1, max_energy_diff=1000):
    num_adaptation_steps = int(tune * 0.8)

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=10,
    )
    kernel = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(
        kernel,
        num_adaptation_steps=num_adaptation_steps)
    kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        kernel, num_adaptation_steps=num_adaptation_steps)
    kernel = tfp.mcmc.TransformedTransitionKernel(
        kernel, bijector)

    def trace_fn(_, pkr):
      energy_diff = pkr.inner_results.inner_results.inner_results.log_accept_ratio
      has_divergence = jnp.abs(energy_diff) > max_energy_diff
      return (
          unnest.get_innermost(pkr, 'target_log_prob'),
          unnest.get_innermost(pkr, 'num_leapfrog_steps'),
          has_divergence,
          energy_diff,
          pkr.inner_results.inner_results.inner_results.log_accept_ratio,
          pkr.inner_results.inner_results.max_trajectory_length,
          unnest.get_innermost(pkr, 'step_size'),
      )
    # The chain will be stepped for num_results + num_burnin_steps, adapting for
    # the first num_adaptation_steps.
    return tfp.mcmc.sample_chain(
            num_results=draws,
            num_burnin_steps=tune,
            current_state=init_state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=seed)
    
  return run_inference_chess

# Commented out IPython magic to ensure Python compatibility.
bijector = [
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Exp(),
    tfb.Sigmoid(0., .5),
]

# run_inference = gen_nuts_sample_fn(log_prob_fn, bijector, 1000, 1000)
run_inference = gen_chess_sample_fn(log_prob_fn, bijector, 1000, 1000)

seed, *init_seed = jax.random.split(seed, len(bijector)+1)
init_state_ = jax.tree_multimap(lambda bij, x, rng: bij.forward(
    tfd.Uniform(-1., 1.).sample(
        bij.inverse(x).shape, seed=rng)),
    bijector, list(init_state), init_seed)

seed, inference_seed = jax.random.split(seed, 2)
# %time mcmc_samples, sampler_stats = run_inference(init_state_, inference_seed)

posterior = {
    k:np.swapaxes(v, 1, 0)
    for k, v in zip([t[0] for t in mdl_mixture.resolve_graph()[:-1]], mcmc_samples)}

# sample_stats_name = ['lp', 'tree_size', 'diverging', 'energy', 'mean_tree_accept', 'reach_max_depth']
sample_stats_name = ['lp', 'tree_size', 'diverging', 'energy', 'mean_tree_accept', 'max_tree_size', 'step_size']
sample_stats = {k: v.T for k, v in zip(sample_stats_name, sampler_stats)}

nuts_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)
axes = az.plot_trace(nuts_trace, compact=True);

"""# Multi-level Regression model (using Jax)"""

#@title Load raw data and clean up
import pandas as pd
srrs2 = pd.read_csv('https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/srrs2.dat')

srrs2.columns = srrs2.columns.map(str.strip)
srrs_mn = srrs2[srrs2.state=='MN'].copy()
srrs_mn['fips'] = srrs_mn.stfips*1000 + srrs_mn.cntyfips

cty = pd.read_csv('https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/cty.dat')
cty_mn = cty[cty.st=='MN'].copy()
cty_mn[ 'fips'] = 1000*cty_mn.stfips + cty_mn.ctfips

srrs_mn = srrs_mn.merge(cty_mn[['fips', 'Uppm']], on='fips')
srrs_mn = srrs_mn.drop_duplicates(subset='idnum')
uranium = np.log(srrs_mn.Uppm).unique()

n = len(srrs_mn)

srrs_mn.county = srrs_mn.county.map(str.strip)
mn_counties = srrs_mn.county.unique()
counties = len(mn_counties)
county_lookup = dict(zip(mn_counties, range(len(mn_counties))))

county_idx = srrs_mn['county_code'] = srrs_mn.county.replace(county_lookup).values
radon = srrs_mn.activity
srrs_mn['log_radon'] = log_radon = np.log(radon + 0.1).values
floor_measure = srrs_mn.floor.values.astype('float')

# Create new variable for mean of floor across counties
avg_floor = srrs_mn.groupby('county')['floor'].mean().rename(county_lookup).values

#@title LazyBones model
seed = jax.random.PRNGKey(642346)
seed, *rv_seed = jax.random.split(seed, 7)

# Hyperpriors:
g = tfdw.Sample(tfdw.Normal(0., 10.), 3).sample(seed=rv_seed[0])
sigma_a = tfdw.Exponential(1.0).sample(seed=rv_seed[1])

# Varying intercepts uranium model:
a = g[0] + g[1] * uranium + g[2] * avg_floor
za_county = tfdw.Sample(
    tfdw.Normal(0., 1.), counties).sample(seed=rv_seed[2])
a_county = a + za_county * sigma_a
# Common slope:
b = tfdw.Normal(0., 1.).sample(seed=rv_seed[3])

# Expected value per county:
theta = a_county[county_idx] + b * floor_measure
# Model error:
sigma = tfdw.Exponential(1.0).sample(seed=rv_seed[4])

y = tfdw.Independent(
    tfdw.Normal(theta, sigma), 1).sample(seed=rv_seed[5])

print(y[0])  # <== value concretization
rvs = [g, sigma_a, za_county, b, sigma, y]
nchain = 5
init_state = [jnp.repeat(rv.value[None, ...], nchain, axis=0) for rv in rvs[:-1]]

bijector = [rv.parents[0].parents[0].experimental_default_event_space_bijector().eval()  # <== concretization
            for rv in rvs[:-1]]

"""## Inference with MCMC"""

# Commented out IPython magic to ensure Python compatibility.
target_log_prob_fn = lambda *values: lb.utils.log_prob(
    vertexes=rvs,
    values=[*values, log_radon])
log_prob_fn = jax.vmap(target_log_prob_fn)

run_inference = gen_nuts_sample_fn(log_prob_fn, bijector, 1000, 1000)

seed, *init_seed = jax.random.split(seed, len(bijector)+1)
init_state_ = jax.tree_multimap(lambda bij, x, rng: bij.forward(
    tfd.Uniform(-1., 1.).sample(bij.inverse(x).shape, seed=rng)),
    bijector, list(init_state), init_seed)

seed, inference_seed = jax.random.split(seed, 2)
# %time mcmc_samples, sampler_stats = run_inference(init_state_, inference_seed)

posterior = {
    k:np.swapaxes(v, 1, 0)
    for k, v in zip(['sigma_a', 'eps', 'gamma', 'b', 'sigma_y'], mcmc_samples)}

sample_stats_name = ['lp', 'tree_size', 'diverging', 'energy', 'mean_tree_accept', 'reach_max_depth']
# sample_stats_name = ['lp', 'tree_size', 'diverging', 'energy', 'mean_tree_accept', 'max_tree_size', 'step_size']
sample_stats = {k: v.T for k, v in zip(sample_stats_name, sampler_stats)}

nuts_trace = az.from_dict(posterior=posterior, sample_stats=sample_stats)
axes = az.plot_trace(nuts_trace, compact=True);

"""# Autoregressive model (using TF)"""

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tf.enable_v2_behavior()

tfd = tfp.distributions

tfw = DeferredInput(tf)
tfpw = DeferredInput(tfp)
tfdw = DeferredInput(tfp.distributions)
tfbw = DeferredInput(tfp.bijectors)

T = 50
driving_noise = 1.
measure_noise = 0.3
n_obs = 5

#@title JointDistributionCoroutine Model
# Autoregressive model as in https://github.com/AI-DI/Brancher#example-autoregressive-modeling
root = tfd.JointDistributionCoroutine.Root

@tfd.JointDistributionCoroutineAutoBatched
def ar_model():
  b = yield root(tfd.LogitNormal(.5, 1., name='b'))
  x0 = yield root(tfd.Normal(0., driving_noise, name='x0'))
  x = [x0]
  for t in range(1, T):
    x_ = yield tfd.Normal(b*x[t-1], driving_noise, name=f'x{t}')
    x.append(x_)
  y = yield tfd.Sample(
      tfd.Normal(
          tf.transpose(tf.stack(x)), 
          measure_noise),
      sample_shape=n_obs,
      name='y'
    )

seed = tfp.util.SeedStream(5, 'test')
*stimulate_params, yobs = ar_model.sample(seed=seed)
plt.plot(np.arange(T), np.squeeze(yobs));

#@title LazyBones model
b = tfdw.LogitNormal(.5, 1.).sample(seed=seed)
x0 = tfdw.Normal(0., driving_noise).sample(seed=seed)
x = [x0]
for t in range(1, T):
  x_ = tfdw.Normal(b * x[t-1], driving_noise).sample(seed=seed)
  x.append(x_)
yobs2 = tfdw.Independent(
    tfdw.Normal(
        tfw.repeat(tfw.stack(x)[..., None], n_obs, axis=-1), 
        measure_noise),
    reinterpreted_batch_ndims=2
  ).sample(seed=seed)

log_prob_parts = ar_model.log_prob_parts([*stimulate_params, yobs])
assert log_prob_parts[0] == lb.utils.log_prob(b, stimulate_params[0])

b.value = stimulate_params[0]
assert tf.reduce_sum(log_prob_parts[1:-1]) == lb.utils.log_prob(x, stimulate_params[1:])

np.testing.assert_allclose(tf.reduce_sum(log_prob_parts[:-1]), lb.utils.log_prob([b, x], stimulate_params), rtol=1e-5)

assert ar_model.log_prob(*stimulate_params, yobs) == lb.utils.log_prob([b, x, yobs2], [stimulate_params, yobs])

print("Without pinning the value of random variable, log_prob is also a random variable:")
b.reset()
print("log_prob of x with dependency b")
for _ in range(10):
  print(lb.utils.log_prob(x, stimulate_params[1:]))

log_prob_fn0 = lambda *values: ar_model.log_prob(*values, yobs)
log_prob_fn = lambda *values: lb.utils.log_prob(
    vertexes=[b, x, yobs2], values=[*values, yobs])

new_values = ar_model.sample()[:-1]

assert log_prob_fn0(*new_values) == log_prob_fn(*new_values)
