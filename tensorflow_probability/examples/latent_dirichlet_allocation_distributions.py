# Copyright 2018 The TensorFlow Probability Authors.
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
"""Trains a Latent Dirichlet Allocation (LDA) model on 20 Newsgroups.

LDA [1] is a topic model for documents represented as bag-of-words
(word counts). It attempts to find a set of topics so that every document from
the corpus is well-described by a few topics.

Suppose that there are `V` words in the vocabulary and we want to learn `K`
topics. For each document, let `w` be its `V`-dimensional vector of word counts
and `theta` be its `K`-dimensional vector of topics. Let `Beta` be a `KxN`
matrix in which each row is a discrete distribution over words in the
corresponding topic (in other words, belong to a unit simplex). Also, let
`alpha` be the `K`-dimensional vector of prior distribution parameters
(prior topic weights).

The model we consider here is obtained from the standard LDA by collapsing
the (non-reparameterizable) Categorical distribution over the topics
[1, Sec. 3.2; 3]. Then, the prior distribution is
`p(theta) = Dirichlet(theta | alpha)`, and the likelihood is
`p(w | theta, Beta) = OneHotCategorical(w | theta Beta)`. This means that we
sample the words from a Categorical distribution that is a weighted average
of topics, with the weights specified by `theta`. The number of samples (words)
in the document is assumed to be known, and the words are sampled independently.
We follow [2] and perform amortized variational inference similarly to
Variational Autoencoders. We use a neural network encoder to
parameterize a Dirichlet variational posterior distribution `q(theta | w)`.
Then, an evidence lower bound (ELBO) is maximized with respect to
`alpha`, `Beta` and the parameters of the variational posterior distribution.

We use the preprocessed version of 20 newsgroups dataset from [3].
This implementation uses the hyperparameters of [2] and reproduces the reported
results (test perplexity ~875).

Example output for the final iteration:

```none
elbo
-567.829

loss
567.883

global_step
180000

reconstruction
-562.065

topics
index=8 alpha=0.46 write article get think one like know say go make
index=21 alpha=0.29 use get thanks one write know anyone car please like
index=0 alpha=0.09 file use key program window image available information
index=43 alpha=0.08 drive use card disk system problem windows driver mac run
index=6 alpha=0.07 god one say christian jesus believe people bible think man
index=5 alpha=0.07 space year new program use research launch university nasa
index=33 alpha=0.07 government gun law people state use right weapon crime
index=36 alpha=0.05 game team play player year win season hockey league score
index=42 alpha=0.05 go say get know come one think people see tell
index=49 alpha=0.04 bike article write post get ride dod car one go

kl
5.76408

perplexity
873.206
```

#### References

[1]: David M. Blei, Andrew Y. Ng, Michael I. Jordan. Latent Dirichlet
     Allocation. In _Journal of Machine Learning Research_, 2003.
     http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
[2]: Michael Figurnov, Shakir Mohamed, Andriy Mnih. Implicit Reparameterization
     Gradients, 2018
     https://arxiv.org/abs/1805.08498
[3]: Akash Srivastava, Charles Sutton. Autoencoding Variational Inference For
     Topic Models. In _International Conference on Learning Representations_,
     2017.
     https://arxiv.org/abs/1703.01488
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
import scipy.sparse
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


flags.DEFINE_float(
    "learning_rate", default=3e-4, help="Learning rate.")
flags.DEFINE_integer(
    "max_steps", default=180000, help="Number of training steps to run.")
flags.DEFINE_integer(
    "num_topics",
    default=50,
    help="The number of topics.")
flags.DEFINE_list(
    "layer_sizes",
    default=["300", "300", "300"],
    help="Comma-separated list denoting hidden units per layer in the encoder.")
flags.DEFINE_string(
    "activation",
    default="relu",
    help="Activation function for all hidden layers.")
flags.DEFINE_integer(
    "batch_size",
    default=32,
    help="Batch size.")
flags.DEFINE_float(
    "prior_initial_value", default=0.7, help="The initial value for prior.")
flags.DEFINE_integer(
    "prior_burn_in_steps",
    default=120000,
    help="The number of training steps with fixed prior.")
flags.DEFINE_string(
    "data_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "lda/data"),
    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "lda/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer(
    "viz_steps", default=10000, help="Frequency at which save visualizations.")
flags.DEFINE_bool("fake_data", default=False, help="If true, uses fake data.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing directory.")

FLAGS = flags.FLAGS


def _clip_dirichlet_parameters(x):
  """Clips the Dirichlet parameters to the numerically stable KL region."""
  return tf.clip_by_value(x, 1e-3, 1e3)


def make_encoder(activation, num_topics, layer_sizes):
  """Create the encoder function.

  Args:
    activation: Activation function to use.
    num_topics: The number of topics.
    layer_sizes: The number of hidden units per layer in the encoder.

  Returns:
    encoder: A `callable` mapping a bag-of-words `Tensor` to a
      `tfd.Distribution` instance over topics.
  """
  encoder_net = tf.keras.Sequential()
  for num_hidden_units in layer_sizes:
    encoder_net.add(
        tf.keras.layers.Dense(
            num_hidden_units,
            activation=activation,
            kernel_initializer=tf.compat.v1.glorot_normal_initializer()))
  encoder_net.add(
      tf.keras.layers.Dense(
          num_topics,
          activation=tf.nn.softplus,
          kernel_initializer=tf.compat.v1.glorot_normal_initializer()))

  def encoder(bag_of_words):
    net = _clip_dirichlet_parameters(encoder_net(bag_of_words))
    return tfd.Dirichlet(concentration=net,
                         name="topics_posterior")

  return encoder


def make_decoder(num_topics, num_words):
  """Create the decoder function.

  Args:
    num_topics: The number of topics.
    num_words: The number of words.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over words.
  """
  topics_words_logits = tf.compat.v1.get_variable(
      "topics_words_logits",
      shape=[num_topics, num_words],
      initializer=tf.compat.v1.glorot_normal_initializer())
  topics_words = tf.nn.softmax(topics_words_logits, axis=-1)

  def decoder(topics):
    word_probs = tf.matmul(topics, topics_words)
    # The observations are bag of words and therefore not one-hot. However,
    # log_prob of OneHotCategorical computes the probability correctly in
    # this case.
    return tfd.OneHotCategorical(probs=word_probs,
                                 name="bag_of_words")

  return decoder, topics_words


def make_prior(num_topics, initial_value):
  """Create the prior distribution.

  Args:
    num_topics: Number of topics.
    initial_value: The starting value for the prior parameters.

  Returns:
    prior: A `callable` that returns a `tf.distribution.Distribution`
        instance, the prior distribution.
    prior_variables: A `list` of `Variable` objects, the trainable parameters
        of the prior.
  """
  def _softplus_inverse(x):
    return np.log(np.expm1(x))

  logit_concentration = tf.compat.v1.get_variable(
      "logit_concentration",
      shape=[1, num_topics],
      initializer=tf.compat.v1.initializers.constant(
          _softplus_inverse(initial_value)))
  concentration = _clip_dirichlet_parameters(
      tf.nn.softplus(logit_concentration))

  def prior():
    return tfd.Dirichlet(concentration=concentration,
                         name="topics_prior")

  prior_variables = [logit_concentration]

  return prior, prior_variables


def model_fn(features, labels, mode, params, config):
  """Build the model function for use in an estimator.

  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.
  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  encoder = make_encoder(params["activation"],
                         params["num_topics"],
                         params["layer_sizes"])
  decoder, topics_words = make_decoder(params["num_topics"],
                                       features.shape[1])
  prior, prior_variables = make_prior(params["num_topics"],
                                      params["prior_initial_value"])

  topics_prior = prior()
  alpha = topics_prior.concentration

  topics_posterior = encoder(features)
  topics = topics_posterior.sample()
  random_reconstruction = decoder(topics)

  reconstruction = random_reconstruction.log_prob(features)
  tf.compat.v1.summary.scalar("reconstruction",
                              tf.reduce_mean(input_tensor=reconstruction))

  # Compute the KL-divergence between two Dirichlets analytically.
  # The sampled KL does not work well for "sparse" distributions
  # (see Appendix D of [2]).
  kl = tfd.kl_divergence(topics_posterior, topics_prior)
  tf.compat.v1.summary.scalar("kl", tf.reduce_mean(input_tensor=kl))

  # Ensure that the KL is non-negative (up to a very small slack).
  # Negative KL can happen due to numerical instability.
  with tf.control_dependencies(
      [tf.compat.v1.assert_greater(kl, -1e-3, message="kl")]):
    kl = tf.identity(kl)

  elbo = reconstruction - kl
  avg_elbo = tf.reduce_mean(input_tensor=elbo)
  tf.compat.v1.summary.scalar("elbo", avg_elbo)
  loss = -avg_elbo

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.compat.v1.train.get_or_create_global_step()
  optimizer = tf.compat.v1.train.AdamOptimizer(params["learning_rate"])

  # This implements the "burn-in" for prior parameters (see Appendix D of [2]).
  # For the first prior_burn_in_steps steps they are fixed, and then trained
  # jointly with the other parameters.
  grads_and_vars = optimizer.compute_gradients(loss)
  grads_and_vars_except_prior = [
      x for x in grads_and_vars if x[1] not in prior_variables]

  def train_op_except_prior():
    return optimizer.apply_gradients(
        grads_and_vars_except_prior,
        global_step=global_step)

  def train_op_all():
    return optimizer.apply_gradients(
        grads_and_vars,
        global_step=global_step)

  train_op = tf.cond(
      pred=global_step < params["prior_burn_in_steps"],
      true_fn=train_op_except_prior,
      false_fn=train_op_all)

  # The perplexity is an exponent of the average negative ELBO per word.
  words_per_document = tf.reduce_sum(input_tensor=features, axis=1)
  log_perplexity = -elbo / words_per_document
  tf.compat.v1.summary.scalar(
      "perplexity", tf.exp(tf.reduce_mean(input_tensor=log_perplexity)))
  (log_perplexity_tensor,
   log_perplexity_update) = tf.compat.v1.metrics.mean(log_perplexity)
  perplexity_tensor = tf.exp(log_perplexity_tensor)

  # Obtain the topics summary. Implemented as a py_func for simplicity.
  topics = tf.compat.v1.py_func(
      functools.partial(get_topics_strings, vocabulary=params["vocabulary"]),
      [topics_words, alpha],
      tf.string,
      stateful=False)
  tf.compat.v1.summary.text("topics", topics)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo": tf.compat.v1.metrics.mean(elbo),
          "reconstruction": tf.compat.v1.metrics.mean(reconstruction),
          "kl": tf.compat.v1.metrics.mean(kl),
          "perplexity": (perplexity_tensor, log_perplexity_update),
          "topics": (topics, tf.no_op()),
      },
  )


def get_topics_strings(topics_words, alpha, vocabulary,
                       topics_to_print=10, words_per_topic=10):
  """Returns the summary of the learned topics.

  Arguments:
    topics_words: KxV tensor with topics as rows and words as columns.
    alpha: 1xK tensor of prior Dirichlet concentrations for the
        topics.
    vocabulary: A mapping of word's integer index to the corresponding string.
    topics_to_print: The number of topics with highest prior weight to
        summarize.
    words_per_topic: Number of wodrs per topic to return.
  Returns:
    summary: A np.array with strings.
  """
  alpha = np.squeeze(alpha, axis=0)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  highest_weight_topics = np.argsort(-alpha, kind="mergesort")
  top_words = np.argsort(-topics_words, axis=1)

  res = []
  for topic_idx in highest_weight_topics[:topics_to_print]:
    l = ["index={} alpha={:.2f}".format(topic_idx, alpha[topic_idx])]
    l += [vocabulary[word] for word in top_words[topic_idx, :words_per_topic]]
    res.append(" ".join(l))

  return np.array(res)


ROOT_PATH = "https://github.com/akashgit/autoencoding_vi_for_topic_models/raw/9db556361409ecb3a732f99b4ef207aeb8516f83/data/20news_clean"
FILE_TEMPLATE = "{split}.txt.npy"


def download(directory, filename):
  """Download a file."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  url = os.path.join(ROOT_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def newsgroups_dataset(directory, split_name, num_words, shuffle_and_repeat):
  """Return 20 newsgroups tf.data.Dataset."""
  data = np.load(download(directory, FILE_TEMPLATE.format(split=split_name)),
                 allow_pickle=True, encoding="latin1")
  # The last row is empty in both train and test.
  data = data[:-1]

  # Each row is a list of word ids in the document. We first convert this to
  # sparse COO matrix (which automatically sums the repeating words). Then,
  # we convert this COO matrix to CSR format which allows for fast querying of
  # documents.
  num_documents = data.shape[0]
  indices = np.array([(row_idx, column_idx)
                      for row_idx, row in enumerate(data)
                      for column_idx in row])
  sparse_matrix = scipy.sparse.coo_matrix(
      (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
      shape=(num_documents, num_words),
      dtype=np.float32)
  sparse_matrix = sparse_matrix.tocsr()

  dataset = tf.data.Dataset.range(num_documents)

  # For training, we shuffle each epoch and repeat the epochs.
  if shuffle_and_repeat:
    dataset = dataset.shuffle(num_documents).repeat()

  # Returns a single document as a dense TensorFlow tensor. The dataset is
  # stored as a sparse matrix outside of the graph.
  def get_row_py_func(idx):
    def get_row_python(idx_py):
      return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)

    py_func = tf.compat.v1.py_func(
        get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_words,))
    return py_func

  dataset = dataset.map(get_row_py_func)
  return dataset


def build_fake_input_fns(batch_size):
  """Build fake data for unit testing."""
  num_words = 1000
  vocabulary = [str(i) for i in range(num_words)]

  random_sample = np.random.randint(
      10, size=(batch_size, num_words)).astype(np.float32)

  def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(random_sample)
    dataset = dataset.batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset.repeat()).get_next()

  def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(random_sample)
    dataset = dataset.batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn, vocabulary


def build_input_fns(data_dir, batch_size):
  """Builds iterators for train and evaluation data.

  Each object is represented as a bag-of-words vector.

  Arguments:
    data_dir: Folder in which to store the data.
    batch_size: Batch size for both train and evaluation.
  Returns:
    train_input_fn: A function that returns an iterator over the training data.
    eval_input_fn: A function that returns an iterator over the evaluation data.
    vocabulary: A mapping of word's integer index to the corresponding string.
  """

  with open(download(data_dir, "vocab.pkl"), "rb") as f:
    words_to_idx = pickle.load(f)
  num_words = len(words_to_idx)

  vocabulary = [None] * num_words
  for word, idx in words_to_idx.items():
    vocabulary[idx] = word

  # Build an iterator over training batches.
  def train_input_fn():
    dataset = newsgroups_dataset(
        data_dir, "train", num_words, shuffle_and_repeat=True)
    # Prefetching makes training about 1.5x faster.
    dataset = dataset.batch(batch_size).prefetch(32)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    dataset = newsgroups_dataset(
        data_dir, "test", num_words, shuffle_and_repeat=False)
    dataset = dataset.batch(batch_size)
    return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

  return train_input_fn, eval_input_fn, vocabulary


def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["layer_sizes"] = [int(units) for units in params["layer_sizes"]]
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.io.gfile.exists(FLAGS.model_dir):
    tf.compat.v1.logging.warn("Deleting old log directory at {}".format(
        FLAGS.model_dir))
    tf.io.gfile.rmtree(FLAGS.model_dir)
  tf.io.gfile.makedirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    train_input_fn, eval_input_fn, vocabulary = build_fake_input_fns(
        FLAGS.batch_size)
  else:
    train_input_fn, eval_input_fn, vocabulary = build_input_fns(
        FLAGS.data_dir, FLAGS.batch_size)
  params["vocabulary"] = vocabulary

  estimator = tf.estimator.Estimator(
      model_fn,
      params=params,
      config=tf.estimator.RunConfig(
          model_dir=FLAGS.model_dir,
          save_checkpoints_steps=FLAGS.viz_steps,
      ),
  )

  for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
    estimator.train(train_input_fn, steps=FLAGS.viz_steps)
    eval_results = estimator.evaluate(eval_input_fn)
    # Print the evaluation results. The keys are strings specified in
    # eval_metric_ops, and the values are NumPy scalars/arrays.
    for key, value in eval_results.items():
      print(key)
      if key == "topics":
        # Topics description is a np.array which prints better row-by-row.
        for s in value:
          print(s)
      else:
        print(str(value))
      print("")
    print("")


if __name__ == "__main__":
  tf.compat.v1.app.run()
