This directory contains proposals and design documents for turnkey inference.

## Turnkey MCMC sampling

Goal: user specifies how many MCMC samples (or effective samples) they want, and
the sampling method takes care of the rest. This includes the definition of
`target_log_prob_fn`, initial states, and choosing the optimal
(parameterization of) the `TransitionKernel`.

### An expanding window tuning for HMC/NUTS

To get well-mixed MCMC samples using Hamiltonian Monte Carlo (i.e., no
divergent, Rhat close to 1 and high effective sample size), we need to make sure
the step size and the covariance matrix of the auxiliary momentum variables is
chosen appropriately. One of the most popular strategies (as implemented in Stan
and PyMC3) is a window adaptation strategy, which basically samples using NUTS
multiple times: start with some default parameters and run a short chain to
get the sample to converge to the typical set (and tune the step size scaling);
then run multiple chains with increasing numbers of samples (to get more and
more accurate estimates of posterior covariance that are then used as the mass
matrix for HMC), and at the end using these tuned parameters to sample from the
posterior.

Currently, the TFP NUTS implementation has a speed bottleneck of waiting for the
slowest chain/batch (due to the SIMD nature), and it could seriously hinder
performance, especially when the (initial) step size is poorly chosen. Thus,
our strategy here is to run very few chains in the initial warm up (1 or 2).
Moreover, by analogy to Stan's expanding memoryless windows (stage II of Stan's
automatic parameter tuning), we implemented an expanding batch, fixed step count
method.

It is worth noting that, in TFP HMC step sizes are defined per dimension of the
target_log_prob_fn. To separate the tuning of the step size (a scalar) and the
mass matrix (a vector for diagonal mass matrix), we apply an inner transform
transition kernel (recall that the covariance matrix Σ acts as a Euclidean
metric to rotate and scale the target_log_prob_fn) using a shift and scale
bijector.
