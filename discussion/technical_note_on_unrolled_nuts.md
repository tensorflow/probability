# Unrolled Implementation of No-U-Turn Sampler

Author: junpenglao@, jvdillon@

This document describes our implementation of the
[No U-Turn Sampler (NUTS)](https://arxiv.org/abs/1111.4246)
(specifically, algorithm 3). Implementing NUTS in TF graph is challenging for
two reasons:
1. TF has limited constructs for control flow, essentially just `tf.cond`,
   `tf.switch_case`, and `tf.while_loop`. Due to this limitation, we cannot
   naively implement recursive algorithms (eg, NUTS).
2. We wish to make efficient use of SIMD architectures across multiple chains.

To accomodate these concerns our implementation makes the following
novel observations:
  - We *offline enumerate* the recursion possibilties and note all read/write
    operations.
  - We pre-allocate requisite memory (for what would otherwise be the recursion
    stack).
  - We compute the U turn check in one pass (unlike in the recursive) via
    broadcasting semantics.
  - We update candidate state after each single leapfrog thus saving additional
    memory.

## Unrolling the recursive tree doubling

The NUTS recursion is a pre-order tree traversal. In the original algorithm 3,
this traversal terminates when the trajectory makes a U turn or there is
divergent sample (during leapfrog integration). We observed that:
1. Typical implementations of NUTS cap the the recursion
   (by default we cap `max_tree_depth` to 10).
2. The NUTS computation is dominated by gradient evals in the leapfrog
   calculation.
3. The remain computation is in U-turn checking and slice sampling which notably
   needs access to (part of) the trajectory's history.

These observations in concert with the need to vectorize control flow motivated
our choice to first analyzing the recursion memory access pattern then "unroll"
NUTS as a simple `tf.while_loop` with memory access patterns governed by the
offline analysis. In effect, we as algorithm implementor are acting as a
compiler to rewrite the recursive into "machine" code. In theory, the same
approach could be apply to _any_ recursive algorithm if the developer is willing
to do program analysis of the recursive.

The memory access pattern is computed as:

```python
def build_tree_uturn_instruction(max_depth, init_memory=0):
  """Run build tree and output the u turn checking input instruction."""
  def _buildtree(address, depth):
    if depth == 0:
      address += 1
      return address, address
    else:
      address_left, address_right = _buildtree(address, depth - 1)
      _, address_right = _buildtree(address_right, depth - 1)
      instruction.append((address_left, address_right))
      return address_left, address_right
  instruction = []
  _, _ = _buildtree(init_memory, max_depth)
  return np.array(instruction, dtype=np.int32)
```

Comparing with the tree building side by side, you can see that we skip the
computation, and only allocate index for memory access. Running it on tree
depth 3 gives:

```
build_tree_uturn_instruction(3)
# ==> array([[1, 2],
             [3, 4],
             [1, 4],
             [5, 6],
             [7, 8],
             [5, 8],
             [1, 8]], dtype=int32)
```

Intuitively, it means that for tree depth 3, we have:

```
leapfrog index:           #1        #2        #3        #4        #5        ...
state/position:      [0]  ==>  [1]  ==>  [2]  ==>  [3]  ==>  [4]  ==>  [5]  ...
U turn check:                   +---------+
                                                    +---------+
                                +-----------------------------+
```

Note that 0 is the previous end of the trajectory, which is also the position
that the new tree doubling leapfrog from.

The most straightforward way to implement this is to have a memory slot of
`2**max_tree_depth` so that even at the largest tree depth we can save all the
state/position and momentum information for U-turn check. (In
`Performance Optimization` we discuss how this exponential space can be reduced
to a linear space requirement.) We now discuss how this unrolling makes sampling
on the trajectory easier.

## Slice sampling within one NUTS step.

The no-u-turn sampler builds a trajectory to integrate over the
Hamiltonian that terminates whenever there is a u-turn. To guarantee time
reversibility within one step, NUTS uses a recursive algorithm reminiscent of
the doubling procedure devised by [Neal (2003)][1] for slice sampling. There
are 3 prominent ways to do so currently:

- In the naive NUTS (algorithm 2 in [Hoffman, Gelman (2011)][2]), this is done
by uniformly sampled from candidate position-momentum states that are in the
slice defined by `u` to satisfy detail balance.

- In the efficient NUTS (algorithm 3 in [2]), a weighted sampling is applied
after every doubling, proposing a move to each new half-tree in turn. This is
akin to delayed-rejection MCMC methods that increases the probability we jump
to a state $\theta^{t+1}$ far from the initial state $\theta^{t}$

- In STAN, a multinomial sampling weighted by the energy change at each sub
tree is used. It is similar to the efficient NUTS above, with additional MH
selection and more options in weighting [3].

In a unrolled batched version (n_chains * n_dimension of states) of NUTS, we want
to generalized the operations after each leapfrog to be always constant. Below
is a breakdown of operations at runtime:
The aim is to do uniform sample from the valid candidates within a subtree. In
Algorithm 3, it is originally done by a sampling within each tree doubling to
save memory. Basically something like:

say we are at treedepth 3

```
leapfrog 1, n'_0  = 1
leapfrog 2, n''_0 = 1
  - select state between 0 and 1 with n'_0/(n'_0+n''_0) probability
    now n'_0 = 2 and we have state_0
leapfrog 3, n'_1  = 1
leapfrog 4, n''_1 = 1
  - select state between 2 and 3 with n'_1/(n'_1+n''_1) probability
    now n''_0 = 2 and we have state_1
  - we then select between state_0 and state_1 with n'_0/(n'_0+n''_0)
    probability and we have state_prime (from state_{0, 1, 2, 3}) and
    n_prime = 4
  (additional sampling after one tree doubling)
  - select state_prime and state0 (state from last tree doubling) with
    min(1, n_prime/n). This procedure basically preferred the later tree and
    maximized the jump.
```

So in a linear way, the programs runs like:

```
  x{k} : the candidate state with k indicating at what depth/stack it is at
  ==>  : selection between 2 states at the same depth within tree doubling

step k with k being the leapfrog counter within one tree doubling
step 1: x0
     2: x0x0 ==> x1  # here x1 is selected from the two different x0
     3: x1x0
     4: x1x0x0 ==> x1x1 ==> x2
     5: x2x0
     6: x2x0x0 ==> x2x1
     7: x2x1x0
     8: x2x1x0x0 ==> x2x1x1 ==> x2x2 ==> x3
     ...
```
similarly, at the last leapfrog of treedepth 10, we would do

```
  1024: x9x8x7x6x5x4x3x2x1x0x0 ==> x9x8x7x6x5x4x3x2x1x1 ==> ... ==> x10
```

With the additional sampling after each tree doubling, we have one fully unrolled
nuts step as the following:

```
  x'   : theta_prime, last state in the beginning of each NUTS step
  x{k} : the candidate state with k indicating at what depth/stack it is at
  ==>  : selection between 2 states at the same depth within tree doubling
  -->  : selection between 2 states after each tree doubling

step k(j) with j being the tree depth, k being the overall leapfrog counter

          states (in the memory)
step 1(0): x'x0 --> x'
     2(1): x'x0
     3(1): x'x0x0 ==> x'x1 --> x'
     4(2): x'x0
     5(2): x'x0x0 ==> x'x1
     6(2): x'x1x0
     7(2): x'x1x0x0 ==> x'x1x1 ==> x'x2 --> x'
     8(3): x'x0
     9(3): x'x0x0 ==> x'x1
    10(3): x'x1x0
    11(3): x'x1x0x0 ==> x'x1x1 ==> x'x2
    12(3): x'x2x0
    ...
```

An batch-friendly implementation is to do weighted sample from a subtree at
every even leapfrog step, or a weighted sample every time a tree doubling is
done. The advantage of that is we can do 1 single sample ops instead of k
sampling at 2^k step (and 1 more if it is at the end of a tree doubling).

Observed that we have:

```
  U([a b c d e f g h]) = U([U[a] U[b] U[c d] U[e f g h]], [1, 1, 2, 4])
```

where `U(iterable, weights=np.ones(iterable.shape[0]))` is uniform sampling.

Now we have for algorithm 2 (not all candidates are valid):

```
  x'   : theta_prime, last state in the beginning of each NUTS step
  x{k} : the candidate state with k indicating at what depth/stack it is at
  ==> U([x,...], [w]) ==> : weighted sampling of x with weight w

step k(j) with j being the tree depth
step 1(0): x0        ==> U([x',x0], [1,1]) ==>                   x''
     2(1): x0                                                                   # This sample is invalid
     3(1): x0x0      ==> U([x'',x0,x0], [2,0,1]) ==>             x'''           # Resulting weight for x''' is 3
     4(2): x0
     5(2): x0x0
     6(2): x0x0x0
     7(2): x0x0x0x0  ==> U([x''',x0,x0,x0,x0], [3,1,1,1,1]) ==>  x''''
     8(3): x0
     9(3): x0x0
    10(3): x0x0x0
    11(3): x0x0x0x0
    12(3): x0x0x0x0x0
    ...
```

and algorithm 3 (note that the difference with above is that whether we are
doing Uniform sample or biased MH sample):

```
  x'   : theta_prime, last state in the beginning of each NUTS step
  x{k} : the candidate state with k indicating at what depth/stack it is at
  ==> U([x,...], [w]) ==> : weighted sampling of x with weight w
  --> MH([x0,x1], p) -->  : mh sample with probability p

step k(j) with j being the tree depth
step 1(0): x0        ==> U([x0], [1]) ==> x1 --> MH([x',x1], 1/1) --> x''
     2(1): x0
     3(1): x0x0      ==> U([x0,x0], [1,1]) ==> x1 --> MH([x'',x1], 2/2) --> x'''
     4(2): x0
     5(2): x0x0
     6(2): x0x0x0                                                               # This sample is invalid
     7(2): x0x0x0x0  ==> U([x0,x0,x0,x0], [1,0,1,1]) ==> x1 --> MH([x''',x1], 4/3) --> x''''
     8(3): x0
     9(3): x0x0
    10(3): x0x0x0
    11(3): x0x0x0x0
    12(3): x0x0x0x0x0
    ...
```

## Performance Optimization

Using a memory slot of the size 2^max_tree_depth like above is quite
convenient for both sampling and u turn check, as we have the whole history avaiable
and can easily index to it. In practice, while it works well for small
problem, users could quickly ran into memory problem with large batch size (i.e.,
number of chains), large latent size (i.e., dimension of the free parameters),
and/or large max_tree_depth. However, we don't actually need to record all the
state.

Here, we used another function `generate_efficient_write_read_instruction`
to compress the memory address write/read instruction - it turns out that we
only need `max_tree_depth` length of memory for U turn checks. The function
takes the leapfrog U-turn instruction as input, and layout the memory write/read
in a sparse matrix, and then perform an address collection by 1, compressing the
location that never been write; 2, reindexing the read access according to the
program counter.
Note that we are currently using `max_tree_depth+1` length so that the operation
after each leapfrog is the same, but this could be further compress/optimized by
fusing the odd and even leapfrog steps together.

As for slice sampling, we do sample after _each_ leapfrog:

```
  x'   : theta_prime, last state in the beginning of each NUTS step
  x_   : placeholder, usually it is the beginning state of the current leapfrog
  x{k} : the candidate state with k indicating at what depth/stack it is at
  ==> U([x,...], [w]) ==> : weighted sampling of x with weight w
  --> MH([x0,x1], p) -->  : mh sample with probability p

step k(j) with j being the tree depth
step 1(0): x0        ==> U([x_,x0], [0,1]) ==> x1 --> MH([x',x1], 1/1) --> x'
     2(1): x0        ==> U([x_,x0], [0,1]) ==> x1
     3(1): x1x0      ==> U([x1,x0], [1,1]) ==> x2 --> MH([x',x2], 2/2) --> x'
     4(2): x0        ==> U([x_,x0], [0,1]) ==> x1
     5(2): x1x0      ==> U([x1,x0], [1,1]) ==> x2
     6(2): x2x0      ==> U([x2,x0], [2,0]) ==> x3   # x3 == x2 as this sample is invalid
     7(2): x3x0      ==> U([x3,x0], [2,1]) ==> x4 --> MH([x',x4], 4/3) --> x'
     8(3): x0        ==> U([x_,x0], [0,1]) ==> x1
     9(3): x1x0      ==> U([x1,x0], [1,1]) ==> x2
    10(3): x2x0      ==> U([x2,x0], [2,1]) ==> x3
    11(3): x3x0      ==> U([x3,x0], [3,1]) ==> x4
    12(3): x4x0      ==> U([x4,x0], [4,1]) ==> x5
    ...
```

which means that for the purpose of slice sampling, it could be memory-less.
This is also valid for multinominal sampling as we accumulating the weight the
same way.

## FAQ

Q: How does this relate to Pyro's [iterative NUTS][4]?
A: There are a few differences:
  1. numpyro only considering unrolling the recursive, but did not think about
  async step across batch of chains, as the implementation is inherently single
  batch - multi chain sampling is enabled by wrapping with `pmap` from JAX.
  (This is similar to the idea of using `pfor` to do autobatch). Instead, our
  version take batching as top priority so that the resulting algorithm is
  compatible with other `tfp.mcmc.*` kernels. This different approach to
  batching makes the two implementations looks/feels quite different when you
  look at the source code. More importantly, a batch-first implementation
  enables the unique opportunity of cross chain step size adaptation and
  diagnostics, which would be extremely difficult if not impossible using `pfor`.
  2. the check u-turn is done quite differently: numpyro compute the checkpoint
  for checking u-turn at [each leaf](https://github.com/pyro-ppl/numpyro/blob/0c9696b147098730ba1a487afdc5de51a9c675c9/numpyro/hmc_util.py#L571-L583)
  at runtime, whereas we pre-run the recursive to build the instruction table
  and indexing to get the current check point at run time.


### References:

[1] Neal R. Slice sampling. Annals of Statistics, 31(3):705–741, 2003.

[2] Matthew D. Hoffman, Andrew Gelman.  The No-U-Turn Sampler: Adaptively
  Setting Path Lengths in Hamiltonian Monte Carlo.  2011.
  https://arxiv.org/pdf/1111.4246.pdf.

[3] https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/hmc/nuts/base_nuts.hpp

[4] https://github.com/pyro-ppl/numpyro/wiki/Iterative-NUTS
