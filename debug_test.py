import tensorflow as tf
import tensorflow_probability as tfp

def quadratic_loss_and_grad(x):
    """f(x) = (x-2)^2 + (y-3)^2, minimum at [2, 3]"""
    diff = x - tf.constant([2.0, 3.0])
    loss = tf.reduce_sum(tf.square(diff))
    grad = 2.0 * diff
    return loss, grad

print("Testing line_search_kwargs feature...")

# Test 1: Default behavior
start = tf.constant([0.0, 0.0])
print("About to call bfgs_minimize...")
try:
    results_default = tfp.optimizer.bfgs_minimize(
        quadratic_loss_and_grad, start, tolerance=1e-10)
    print("Default result:", results_default.position.numpy())
except Exception as e:
    print("Error in bfgs_minimize:", e)
    import traceback
    traceback.print_exc()