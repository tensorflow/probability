Project: /probability/_project.yaml
Book: /probability/_book.yaml
page_type: reference
<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfp.mcmc" />
</div>

# Module: tfp.mcmc

TensorFlow Probability MCMC python package.

## Classes

[`class HamiltonianMonteCarlo`](../tfp/mcmc/HamiltonianMonteCarlo.md): Runs one step of Hamiltonian Monte Carlo.

[`class MetropolisAdjustedLangevinAlgorithm`](../tfp/mcmc/MetropolisAdjustedLangevinAlgorithm.md): Runs one step of Metropolis-adjusted Langevin algorithm.

[`class MetropolisHastings`](../tfp/mcmc/MetropolisHastings.md): Runs one step of the Metropolis-Hastings algorithm.

[`class RandomWalkMetropolis`](../tfp/mcmc/RandomWalkMetropolis.md): Runs one step of the RWM algorithm with symmetric proposal.

[`class ReplicaExchangeMC`](../tfp/mcmc/ReplicaExchangeMC.md): Runs one step of the Replica Exchange Monte Carlo.

[`class SliceSampler`](../tfp/mcmc/SliceSampler.md): Runs one step of the slice sampler using a hit and run approach.

[`class TransformedTransitionKernel`](../tfp/mcmc/TransformedTransitionKernel.md): TransformedTransitionKernel applies a bijector to the MCMC's state space.

[`class TransitionKernel`](../tfp/mcmc/TransitionKernel.md): Base class for all MCMC `TransitionKernel`s.

[`class UncalibratedHamiltonianMonteCarlo`](../tfp/mcmc/UncalibratedHamiltonianMonteCarlo.md): Runs one step of Uncalibrated Hamiltonian Monte Carlo.

[`class UncalibratedLangevin`](../tfp/mcmc/UncalibratedLangevin.md): Runs one step of Uncalibrated Langevin discretized diffusion.

[`class UncalibratedRandomWalk`](../tfp/mcmc/UncalibratedRandomWalk.md): Generate proposal for the Random Walk Metropolis algorithm.

## Functions

[`default_exchange_proposed_fn(...)`](../tfp/mcmc/default_exchange_proposed_fn.md): Default function for `exchange_proposed_fn` of `kernel`.

[`effective_sample_size(...)`](../tfp/mcmc/effective_sample_size.md): Estimate a lower bound on effective sample size for each independent chain.

[`potential_scale_reduction(...)`](../tfp/mcmc/potential_scale_reduction.md): Gelman and Rubin (1992)'s potential scale reduction for chain convergence.

[`random_walk_normal_fn(...)`](../tfp/mcmc/random_walk_normal_fn.md): Returns a callable that adds a random normal perturbation to the input.

[`random_walk_uniform_fn(...)`](../tfp/mcmc/random_walk_uniform_fn.md): Returns a callable that adds a random uniform perturbation to the input.

[`sample_annealed_importance_chain(...)`](../tfp/mcmc/sample_annealed_importance_chain.md): Runs annealed importance sampling (AIS) to estimate normalizing constants.

[`sample_chain(...)`](../tfp/mcmc/sample_chain.md): Implements Markov chain Monte Carlo via repeated `TransitionKernel` steps.

[`sample_halton_sequence(...)`](../tfp/mcmc/sample_halton_sequence.md): Returns a sample from the `dim` dimensional Halton sequence.

