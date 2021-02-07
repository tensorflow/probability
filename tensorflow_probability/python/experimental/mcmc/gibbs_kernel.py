"""Gibbs sampling kernel"""
import collections
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.internal import unnest
from tensorflow_probability.python.internal import prefer_static

tfd = tfp.distributions  # pylint: disable=no-member
tfb = tfp.bijectors  # pylint: disable=no-member
mcmc = tfp.mcmc  # pylint: disable=no-member


class GibbsKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        "GibbsKernelResults",
        [
            "target_log_prob",
            "inner_results",
        ],
    ),
):
    __slots__ = ()


def _flatten_results(results):
    """Results structures from nested Gibbs samplers sometimes
    need flattening for writing out purposes.
    """

    def recurse(r):
        for i in iter(r):
            if isinstance(i, list):
                for j in _flatten_results(i):
                    yield j
            else:
                yield i

    return [r for r in recurse(results)]


def _has_gradients(results):
    return unnest.has_nested(results, "grads_target_log_prob")


def _get_target_log_prob(results):
    """Fetches a target log prob from a results structure"""
    return unnest.get_innermost(results, "target_log_prob")


def _update_target_log_prob(results, target_log_prob):
    """Puts a target log prob into a results structure"""
    if isinstance(results, GibbsKernelResults):
        replace_fn = unnest.replace_outermost
    else:
        replace_fn = unnest.replace_innermost
    return replace_fn(results, target_log_prob=target_log_prob)


def _maybe_transform_value(tlp, state, kernel, direction):
    if not isinstance(kernel, tfp.mcmc.TransformedTransitionKernel):
        return tlp

    tlp_rank = prefer_static.rank(tlp)
    event_ndims = prefer_static.rank(state) - tlp_rank

    if direction == "forward":
        return tlp + kernel.bijector.inverse_log_det_jacobian(
            state, event_ndims=event_ndims
        )
    if direction == "inverse":
        return tlp - kernel.bijector.inverse_log_det_jacobian(
            state, event_ndims=event_ndims
        )
    raise AttributeError("`direction` must be `forward` or `inverse`")


class GibbsKernel(mcmc.TransitionKernel):
    """Gibbs Sampling Algorithm.
    Gibbs sampling may be useful when the joint distribution is explicitly unknown
    or difficult to sample from directly, but the conditional distribution for each
    variable is known and can be sampled directly. The Gibbs sampling algorithm
    generates a realisation from each variable's conditional distribution in turn,
    conditional on the current realisations of the other variables.
    The resulting sequence of samples forms a Markov chain whose stationary
    distribution represents the joint distribution.

    In pseudo code the algorithm is:
    ```
      Inputs:
        D number of dimensions (i.e. number of parameters)
        D' = D - 1
        X1, X2, ...,XD', XD random variables
        x1[0], x2[0], ..., xD'[0], xD[0] initial chain state for random variables
        i iteration index
        N total number of steps
        pi(.) denotes probability distribution of argument

      for i = 1,..., N do

        x1[i] ~ pi(X1=x1|X2=x2[i-1], X3=x3[i-1], ..., XD=xD[i-1])
        x2[i] ~ pi(X2=x2|X1=x1[i-1], X3=x3[i-1], ..., XD=xD[i-1])
        ...
        xD[i] ~ pi(XD=xD|X1=x1[i-1], X2=x2[i-1], ..., XD'=xD'[i-1])

        i = i + 1

      end
    ```

    #### Example 1: 2-variate MVN
    ```python
        import numpy as np
        import tensorflow as tf
        import tensorflow_probability as tfp
        from gemlib.mcmc.gibbs_kernel import GibbsKernel

        tfd = tfp.distributions

        dtype = np.float32
        true_mean = dtype([1, 1])
        true_cov = dtype([[1, 0.5], [0.5, 1]])
        target = tfd.MultivariateNormalTriL(
            loc=true_mean,
            scale_tril=tf.linalg.cholesky(true_cov)
        )


        def log_prob(x1, x2):
            return target.log_prob([x1, x2])


        def kernel_make_fn(target_log_prob_fn, state):
            return tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_log_prob_fn)


        @tf.function
        def posterior(iterations, burnin, initial_state):
            kernel_list = [(0, kernel_make_fn),
                           (1, kernel_make_fn)]
            kernel = GibbsKernel(
                target_log_prob_fn=log_prob,
                kernel_list=kernel_list
            )
            return tfp.mcmc.sample_chain(
                num_results=iterations,
                current_state=initial_state,
                kernel=kernel,
                num_burnin_steps=burnin,
                trace_fn=None)


        samples = posterior(
            iterations=10000,
            burnin=1000,
            initial_state=[dtype(1), dtype(1)])

        tf.print('sample_mean', tf.math.reduce_mean(samples, axis=1))
        tf.print('sample_cov', tfp.stats.covariance(tf.transpose(samples)))

    ```

    #### Example 2: linear model
    ```python
        import numpy as np
        import tensorflow as tf
        import tensorflow_probability as tfp
        from gemlib.mcmc.gibbs_kernel import GibbsKernel

        tfd = tfp.distributions

        dtype = np.float32

        # data
        x = dtype([2.9, 4.2, 8.3, 1.9, 2.6, 1.0, 8.4, 8.6, 7.9, 4.3])
        y = dtype([6.2, 7.8, 8.1, 2.7, 4.8, 2.4, 10.7, 9.0, 9.6, 5.7])


        # define linear regression model
        def Model(x):
            def alpha():
                return tfd.Normal(loc=dtype(0.), scale=dtype(1000.))

            def beta():
                return tfd.Normal(loc=dtype(0.), scale=dtype(100.))

            def sigma():
                return tfd.Gamma(concentration=dtype(0.1), rate=dtype(0.1))

            def y(alpha, beta, sigma):
                mu = alpha + beta * x
                return tfd.Normal(mu, scale=sigma)

            return tfd.JointDistributionNamed(dict(
                alpha=alpha,
                beta=beta,
                sigma=sigma,
                y=y))


        # target log probability of linear model
        def log_prob(alpha, beta, sigma):
            lp = model.log_prob({'alpha': alpha,
                                 'beta': beta,
                                 'sigma': sigma,
                                 'y': y})
            return tf.reduce_sum(lp)


        # random walk Markov chain function
        def kernel_make_fn(target_log_prob_fn, state):
            return tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=target_log_prob_fn)


        # posterior distribution MCMC chain
        @tf.function
        def posterior(iterations, burnin, thinning, initial_state):
            kernel_list = [(0, kernel_make_fn), # conditional probability for zeroth parmeter alpha
                           (1, kernel_make_fn), # conditional probability for first parameter beta
                           (2, kernel_make_fn)] # conditional probability for second parameter sigma
            kernel = GibbsKernel(
                target_log_prob_fn=log_prob,
                kernel_list=kernel_list
            )
            return tfp.mcmc.sample_chain(
                num_results=iterations,
                current_state=initial_state,
                kernel=kernel,
                num_burnin_steps=burnin,
                num_steps_between_results=thinning,
                parallel_iterations=1,
                trace_fn=None)


        # initialize model
        model = Model(x)
        initial_state = [dtype(0.1), dtype(0.1), dtype(0.1)]  # start chain at alpha=0.1, beta=0.1, sigma=0.1

        # estimate posterior distribution
        samples = posterior(
            iterations=10000,
            burnin=1000,
            thinning=0,
            initial_state=initial_state)

        tf.print('alpha samples:', samples[0])
        tf.print('beta  samples:', samples[1])
        tf.print('sigma samples:', samples[2])
        tf.print('sample means: [alpha, beta, sigma] =', tf.math.reduce_mean(samples, axis=1))

    ```



    """

    def __init__(self, target_log_prob_fn, kernel_list, name=None):
        """Build a Gibbs sampling scheme from component kernels.

        :param target_log_prob_fn: a function that takes `state` arguments
                                   and returns the target log probability
                                   density.
        :param kernel_list: a list of tuples `(state_part_idx, kernel_make_fn)`.
                            `state_part_idx` denotes the index (relative to
                            positional args in `target_log_prob_fn`) of the
                            state the kernel updates.  `kernel_make_fn` takes
                            arguments `target_log_prob_fn` and `state`, returning
                            a `tfp.mcmc.TransitionKernel`.
        :returns: an instance of `GibbsKernel`
        """
        # Require to check if all kernel.is_calibrated is True
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            kernel_list=kernel_list,
            name=name,
        )

    @property
    def is_calibrated(self):
        return True

    @property
    def target_log_prob_fn(self):
        return self._parameters["target_log_prob_fn"]

    @property
    def kernel_list(self):
        return self._parameters["kernel_list"]

    @property
    def name(self):
        return self._parameters["name"]

    def one_step(self, current_state, previous_results, seed=None):
        """We iterate over the state elements, calling each kernel in turn.

        The `target_log_prob` is forwarded to the next `previous_results`
        such that each kernel has a current `target_log_prob` value.
        Transformations are automatically performed if the kernel is of
        type tfp.mcmc.TransformedTransitionKernel.

        In graph and XLA modes, the for loop should be unrolled.
        """
        if mcmc_util.is_list_like(current_state):
            state_parts = list(current_state)
        else:
            state_parts = [current_state]

        state_parts = [
            tf.convert_to_tensor(s, name="current_state") for s in state_parts
        ]

        next_results = []
        untransformed_target_log_prob = previous_results.target_log_prob

        for (state_part_idx, kernel_fn), previous_step_results in zip(
            self.kernel_list, previous_results.inner_results
        ):

            def target_log_prob_fn(state_part):
                state_parts[
                    state_part_idx  # pylint: disable=cell-var-from-loop
                ] = state_part
                return self.target_log_prob_fn(*state_parts)

            kernel = kernel_fn(target_log_prob_fn, state_parts)

            # Forward the current tlp to the kernel.  If the kernel is gradient-based,
            # we need to calculate fresh gradients, as these cannot easily be forwarded
            # from the previous Gibbs step.
            if _has_gradients(previous_step_results):
                # TODO would be better to avoid re-calculating the whole of
                # `bootstrap_results` when we just need to calculate gradients.
                fresh_previous_results = unnest.UnnestingWrapper(
                    kernel.bootstrap_results(state_parts[state_part_idx])
                )
                previous_step_results = unnest.replace_innermost(
                    previous_step_results,
                    target_log_prob=fresh_previous_results.target_log_prob,
                    grads_target_log_prob=fresh_previous_results.grads_target_log_prob,
                )

            else:
                previous_step_results = _update_target_log_prob(
                    previous_step_results,
                    _maybe_transform_value(
                        tlp=untransformed_target_log_prob,
                        state=state_parts[state_part_idx],
                        kernel=kernel,
                        direction="inverse",
                    ),
                )

            state_parts[state_part_idx], next_kernel_results = kernel.one_step(
                state_parts[state_part_idx], previous_step_results, seed
            )

            next_results.append(next_kernel_results)

            # Cache the new tlp for use in the next Gibbs step
            untransformed_target_log_prob = _maybe_transform_value(
                tlp=_get_target_log_prob(next_kernel_results),
                state=state_parts[state_part_idx],
                kernel=kernel,
                direction="forward",
            )

        return (
            state_parts
            if mcmc_util.is_list_like(current_state)
            else state_parts[0],
            GibbsKernelResults(
                target_log_prob=untransformed_target_log_prob,
                inner_results=next_results,
            ),
        )

    def bootstrap_results(self, current_state):

        if mcmc_util.is_list_like(current_state):
            current_state = list(current_state)
        else:
            current_state = [tf.convert_to_tensor(current_state)]
        current_state = [
            tf.convert_to_tensor(s, name="current_state") for s in current_state
        ]

        inner_results = []
        untransformed_target_log_prob = 0.0
        for state_part_idx, kernel_fn in self.kernel_list:

            def target_log_prob_fn(_):
                return self.target_log_prob_fn(*current_state)

            kernel = kernel_fn(target_log_prob_fn, current_state)
            kernel_results = kernel.bootstrap_results(
                current_state[state_part_idx]
            )
            inner_results.append(kernel_results)
            untransformed_target_log_prob = _maybe_transform_value(
                tlp=_get_target_log_prob(kernel_results),
                state=current_state[state_part_idx],
                kernel=kernel,
                direction="forward",
            )

        return GibbsKernelResults(
            target_log_prob=untransformed_target_log_prob,
            inner_results=inner_results,
        )
