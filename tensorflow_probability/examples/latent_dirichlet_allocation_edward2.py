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
     Gradients, 2018.
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
import tensorflow as tf

from tensorflow_probability import edward2 as ed


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


def _softplus_inverse(x):
  """Returns inverse of softplus function."""
  return np.log(np.expm1(x))


def latent_dirichlet_allocation(concentration, topics_words):
  """Latent Dirichlet Allocation in terms of its generative process.

  The model posits a distribution over bags of words and is parameterized by
  a concentration and the topic-word probabilities. It collapses per-word
  topic assignments.

  Args:
    concentration: A Tensor of shape [1, num_topics], which parameterizes the
      Dirichlet prior over topics.
    topics_words: A Tensor of shape [num_topics, num_words], where each row
      (topic) denotes the probability of each word being in that topic.

  Returns:
    bag_of_words: A random variable capturing a sample from the model, of shape
      [1, num_words]. It represents one generated document as a bag of words.
  """
  topics = ed.Dirichlet(concentration=concentration, name="topics")
  word_probs = tf.matmul(topics, topics_words)
  # The observations are bags of words and therefore not one-hot. However,
  # log_prob of OneHotCategorical computes the probability correctly in
  # this case.
  bag_of_words = ed.OneHotCategorical(probs=word_probs, name="bag_of_words")
  return bag_of_words


def make_lda_variational(activation, num_topics, layer_sizes):
  """Creates the variational distribution for LDA.

  Args:
    activation: Activation function to use.
    num_topics: The number of topics.
    layer_sizes: The number of hidden units per layer in the encoder.

  Returns:
    lda_variational: A function that takes a bag-of-words Tensor as
      input and returns a distribution over topics.
  """
  encoder_net = tf.keras.Sequential()
  for num_hidden_units in layer_sizes:
    encoder_net.add(tf.keras.layers.Dense(
        num_hidden_units, activation=activation,
        kernel_initializer=tf.glorot_normal_initializer()))
  encoder_net.add(tf.keras.layers.Dense(
      num_topics, activation=tf.nn.softplus,
      kernel_initializer=tf.glorot_normal_initializer()))

  def lda_variational(bag_of_words):
    concentration = _clip_dirichlet_parameters(encoder_net(bag_of_words))
    return ed.Dirichlet(concentration=concentration, name="topics_posterior")

  return lda_variational


def make_value_setter(**model_kwargs):
  """Creates a value-setting interceptor.

  Args:
    **model_kwargs: dict of str to Tensor. Keys are the names of random variable
      in the model to which this interceptor is being applied. Values are
      Tensors to set their value to.

  Returns:
    set_values: Function which sets the value of intercepted ops.
  """
  def set_values(f, *args, **kwargs):
    """Sets random variable values to its aligned value."""
    name = kwargs.get("name")
    if name in model_kwargs:
      kwargs["value"] = model_kwargs[name]
    return ed.interceptable(f)(*args, **kwargs)
  return set_values


def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an Estimator.

  Arguments:
    features: The input features for the Estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.

  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  # Set up the model's learnable parameters.
  logit_concentration = tf.get_variable(
      "logit_concentration",
      shape=[1, params["num_topics"]],
      initializer=tf.constant_initializer(
          _softplus_inverse(params["prior_initial_value"])))
  concentration = _clip_dirichlet_parameters(
      tf.nn.softplus(logit_concentration))

  num_words = features.shape[1]
  topics_words_logits = tf.get_variable(
      "topics_words_logits",
      shape=[params["num_topics"], num_words],
      initializer=tf.glorot_normal_initializer())
  topics_words = tf.nn.softmax(topics_words_logits, axis=-1)

  # Compute expected log-likelihood. First, sample from the variational
  # distribution; second, compute the log-likelihood given the sample.
  lda_variational = make_lda_variational(
      params["activation"],
      params["num_topics"],
      params["layer_sizes"])
  with ed.tape() as variational_tape:
    _ = lda_variational(features)

  with ed.tape() as model_tape:
    with ed.interception(
        make_value_setter(topics=variational_tape["topics_posterior"])):
      posterior_predictive = latent_dirichlet_allocation(concentration,
                                                         topics_words)

  log_likelihood = posterior_predictive.distribution.log_prob(features)
  tf.summary.scalar("log_likelihood", tf.reduce_mean(log_likelihood))

  # Compute the KL-divergence between two Dirichlets analytically.
  # The sampled KL does not work well for "sparse" distributions
  # (see Appendix D of [2]).
  kl = variational_tape["topics_posterior"].distribution.kl_divergence(
      model_tape["topics"].distribution)
  tf.summary.scalar("kl", tf.reduce_mean(kl))

  # Ensure that the KL is non-negative (up to a very small slack).
  # Negative KL can happen due to numerical instability.
  with tf.control_dependencies([tf.assert_greater(kl, -1e-3, message="kl")]):
    kl = tf.identity(kl)

  elbo = log_likelihood - kl
  avg_elbo = tf.reduce_mean(elbo)
  tf.summary.scalar("elbo", avg_elbo)
  loss = -avg_elbo

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.AdamOptimizer(params["learning_rate"])

  # This implements the "burn-in" for prior parameters (see Appendix D of [2]).
  # For the first prior_burn_in_steps steps they are fixed, and then trained
  # jointly with the other parameters.
  grads_and_vars = optimizer.compute_gradients(loss)
  grads_and_vars_except_prior = [
      x for x in grads_and_vars if x[1] != logit_concentration]

  def train_op_except_prior():
    return optimizer.apply_gradients(
        grads_and_vars_except_prior,
        global_step=global_step)

  def train_op_all():
    return optimizer.apply_gradients(
        grads_and_vars,
        global_step=global_step)

  train_op = tf.cond(
      global_step < params["prior_burn_in_steps"],
      true_fn=train_op_except_prior,
      false_fn=train_op_all)

  # The perplexity is an exponent of the average negative ELBO per word.
  words_per_document = tf.reduce_sum(features, axis=1)
  log_perplexity = -elbo / words_per_document
  tf.summary.scalar("perplexity", tf.exp(tf.reduce_mean(log_perplexity)))
  (log_perplexity_tensor, log_perplexity_update) = tf.metrics.mean(
      log_perplexity)
  perplexity_tensor = tf.exp(log_perplexity_tensor)

  # Obtain the topics summary. Implemented as a py_func for simplicity.
  topics = tf.py_func(
      functools.partial(get_topics_strings, vocabulary=params["vocabulary"]),
      [topics_words, concentration], tf.string, stateful=False)
  tf.summary.text("topics", topics)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo": tf.metrics.mean(elbo),
          "log_likelihood": tf.metrics.mean(log_likelihood),
          "kl": tf.metrics.mean(kl),
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
  if tf.gfile.Exists(filepath):
    return filepath
  if not tf.gfile.Exists(directory):
    tf.gfile.MakeDirs(directory)
  url = os.path.join(ROOT_PATH, filename)
  print("Downloading %s to %s" % (url, filepath))
  urllib.request.urlretrieve(url, filepath)
  return filepath


def newsgroups_dataset(directory, split_name, num_words, shuffle_and_repeat):
  """20 newsgroups as a tf.data.Dataset."""
  data = np.load(download(directory, FILE_TEMPLATE.format(split=split_name)))
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
    py_func = tf.py_func(get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_words,))
    return py_func

  dataset = dataset.map(get_row_py_func)
  return dataset


def build_fake_input_fns(batch_size):
  """Builds fake data for unit testing."""
  num_words = 1000
  vocabulary = [str(i) for i in range(num_words)]

  random_sample = np.random.randint(
      10, size=(batch_size, num_words)).astype(np.float32)

  def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(random_sample)
    dataset = dataset.batch(batch_size).repeat()
    return dataset.make_one_shot_iterator().get_next()

  def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(random_sample)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

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

  with open(download(data_dir, "vocab.pkl"), "r") as f:
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
    return dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the heldout set.
  def eval_input_fn():
    dataset = newsgroups_dataset(
        data_dir, "test", num_words, shuffle_and_repeat=False)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn, vocabulary


def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["layer_sizes"] = [int(units) for units in params["layer_sizes"]]
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

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
  tf.app.run()
