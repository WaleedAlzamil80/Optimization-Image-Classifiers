import tensorflow as tf


def LineSearch(X, loss_func, eta=0.01):
    """
    line search is a method to determine an anappropriate step length (learning rate -eta-) usually it's one-dimensional search

    Args:
        - X: tf.Variable, the variable value at which we want to the the optimal learning rate.
        - loss_func: Callable, the loss function that computes the loss given the current variable values.
        - eta: tf.Variable, the learning rate to be updated to control the size of the update steps (default is 0.01).

    Returns:
        None
    """
    with tf.GradientTape(persistent=True) as tape_0:
        current_loss = loss_func(X)
    dX = tape_0.gradient(current_loss, X)
    with tf.GradientTape(persistent=True) as tape_2:
        with tf.GradientTape(persistent=True) as tape_1:
            Q = loss_func(X - eta * dX)
        dQ = tape_1.gradient(Q, eta)
    d2Q = tape_2.gradient(dQ, eta)
    eta.assign_sub(dQ / d2Q)
    return eta