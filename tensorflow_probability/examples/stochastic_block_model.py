import time

from matplotlib import pylab as plt
import numpy as np
from observations import karate
import tensorflow as tf
from tensorflow_probability import edward2 as ed


# Data & parameters
# -----------------------------------------------------------------------------
x_data, z = karate('~/data')
V = x_data.shape[1]
K = 2
learning_rate = 1e-4
max_steps = 10000
epsilon= 0.001


def compute_loss(latent_vars, data):
    """ Compute the loss associated with MAP calculations.
    """
    dict_vals = {var: latent.value for var, latent in latent_vars.items()}
    for x, datum in data.items(): # assuming we have the values as input
        dict_vals[x] = datum
    
    log_prob = 0.0
    for z in latent_vars.keys():
        z_copy = z.distribution.copy()
        log_prob += tf.reduce_sum(z_copy.log_prob(dict_vals[z]))

    for x in data.keys():
        x_copy = x.distribution.copy()
        log_prob += tf.reduce_sum(x_copy.log_prob(dict_vals[x]))

    reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
    loss = -log_prob + reg_penalty
    
    return loss


# Model
# -----------------------------------------------------------------------------
gamma = ed.Dirichlet(tf.ones([K]))
pi = ed.Beta(tf.ones([K,K]), tf.ones([K,K]))
z = ed.Multinomial(tf.ones([V]), gamma)
x = ed.Bernoulli(tf.matmul(z, tf.matmul(pi, tf.transpose(z))))


# Define the MAP loss
# -----------------------------------------------------------------------------
qgamma = ed.VectorDeterministic(tf.nn.softmax(tf.get_variable('qgamma/params', [K]))) # Gamma must sum to one
qpi = ed.VectorDeterministic(tf.nn.sigmoid(tf.get_variable('qpi/params', [K, K]))) # Each pi must be between 0 and 1
qz = ed.VectorDeterministic(tf.nn.softmax(tf.get_variable('qz/params', [V,K]), axis=1)) # the Z_i must sum to one row-wise (axis 0)

latent_vars = {gamma: qgamma, pi: qpi, z: qz}
data = {x: x_data}
MAP_loss = compute_loss(latent_vars, data)


# Inference
# -----------------------------------------------------------------------------
optimizer = tf.train.AdagradOptimizer(learning_rate)
train_op = optimizer.minimize(MAP_loss)

loss_vals = []
with tf.Session() as session:
    start = time.time()
    session.run(tf.global_variables_initializer())

    for step in range(max_steps):
        _, loss_value = session.run([train_op, MAP_loss])
        duration = time.time() - start
        if step % 100 == 0:
            print("Step: {:>3d} Loss: {:.3f} ({:.3f} sec)".format(step, 
                                                                  loss_value,
                                                                  duration))
        if step > 0:
            if abs(loss_vals[-1]-loss_value) <  epsilon:
                break
        loss_vals.append(loss_value)


fig = plt.figure()
plt.plot(range(len(loss_vals)), loss_vals)
plt.show()
