import functools

import tensorflow.compat.v2 as tf
from inference_gym.targets import bayesian_model
from inference_gym.targets import model

import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions

Root = tfd.JointDistributionCoroutine.Root


# coupling link could be e.g. tf.nn.tanh
def gaussian_binary_tree_prior_fn(num_layers, initial_scale, nodes_scale,
                                  coupling_link=None):
  initial_loc = 0.
  # in the "root" layer (or inverse root, as it is a reversed tree) we have
  # 2**(num_layers-1) nodes (with depth 2 --> 2 nodes, depth 4 --> 8 nodes)

  nodes = yield Root(
    tfd.Sample(tfd.Normal(initial_loc, initial_scale), 2 ** (num_layers - 1),
               name=f'layer_{num_layers - 1}'))
  # for the remaining layers, we then sample the respective nodes values
  # applying the link function
  # we do not do this for the final node, as it is supposed to be observed
  for l in range(num_layers, 1, -1):
    if coupling_link:
      nodes = yield tfd.Independent(tfd.Normal(tf.stack(
        [coupling_link(nodes[..., i]) - coupling_link(nodes[..., i + 1]) for i
         in range(0, 2 ** l, 2)],
        -1),
        nodes_scale), 1, name=f'layer_{l - 1}')
    else:
      nodes = yield tfd.Independent(tfd.Normal(tf.stack(
        [nodes[..., i] - nodes[..., i + 1] for i in range(0, 2 ** l, 2)], -1),
        nodes_scale), 1, name=f'layer_{l - 1}')


def gaussian_binary_tree_log_likelihood_fn(values, observed_last_node,
                                           nodes_scale, coupling_link=None):
  left_node, right_node = values[-2], values[-1]
  if coupling_link:
    lps = tfd.Normal(loc=coupling_link(left_node) - coupling_link(right_node),
                     scale=nodes_scale).log_prob(observed_last_node)
  else:
    lps = tfd.Normal(loc=left_node - right_node,
                     scale=nodes_scale).log_prob(observed_last_node)
  return lps


class GaussianBinaryTree(bayesian_model.BayesianModel):
  def __init__(self,
               num_layers,
               observed_last_node,
               initial_scale,
               nodes_scale,
               coupling_link=None,
               name='gaussian_binary_tree',
               pretty_name='Gaussian Binary Tree'):
    """Construct the Gaussian Binary Tree model."""
    with tf.name_scope(name):
      self._prior_dist = tfd.JointDistributionCoroutine(functools.partial(
        gaussian_binary_tree_prior_fn,
        num_layers=num_layers,
        initial_scale=initial_scale,
        nodes_scale=nodes_scale,
        coupling_link=coupling_link
      ))

      self._log_likelihood_fn = functools.partial(
        gaussian_binary_tree_log_likelihood_fn,
        observed_last_node=observed_last_node,
        nodes_scale=nodes_scale,
        coupling_link=coupling_link
      )

      # todo: what should I use here?
      sample_transformations = {
        'identity':
          model.Model.SampleTransformation(
            fn=lambda params: params,
            pretty_name='Identity',
            dtype=self._prior_dist.dtype,
          )
      }

    super(GaussianBinaryTree, self).__init__(
      default_event_space_bijector=tfb.Identity(),
      # todo: what should I use here?
      event_shape=self._prior_dist.event_shape,
      dtype=self._prior_dist.dtype,
      name=name,
      pretty_name=pretty_name,
      sample_transformations=sample_transformations
    )

  def _prior_distribution(self):
    return self._prior_dist

  def _log_likelihood(self, value):
    return self._log_likelihood_fn(value)
