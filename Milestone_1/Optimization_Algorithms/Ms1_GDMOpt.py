import numpy as np
import tensorflow as tf

def GDM(loss_func, X_init, V_init, X_vals, loss_vals, eta=0.01, beta_1=0.9):
    """
    Gradient Descent with Momentum optimization algorithm for updating the values of a given variable.

    Args:
      - loss_func: Callable, the loss function that computes the loss given the current variable values.
      - X_init: tf.Variable, the initial variable values to be updated.
      - eta: float, the learning rate used to control the size of the update steps (default is 0.01).
      - beta_1: float, the momentum hyperparameter used to control the momentum effect (default is 0.9).
      - X_vals: list(), empty list used to save X at each iteration.
      - loss_vals: list(), empty list used to save the loss values at each iteration.


    Returns:
      None
    """

    with tf.GradientTape(persistent=True) as t:
        # Compute the current loss
        current_loss = loss_func(X_init)

    # Compute the gradient of the loss with respect to the variables
    dx = t.gradient(current_loss, X_init)

    # Compute the estimate (momentum)
    V=beta_1 * V_init + (1 - beta_1) * dx

    # Update the variables using the momentum update rule
    X_init.assign_sub(eta * V)

    # Update the moment estimate (momentum)
    V_init.assign(V)


    # calc the loss value and append it to the list
    loss_vals.append(tf.squeeze (loss_func(X_init)))

    #append X to the list
    X_vals.append(X_init.numpy())
