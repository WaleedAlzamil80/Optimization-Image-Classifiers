import tensorflow as tf
import numpy as np
from LineSearch import *

def GDM(loss_func, X_init, V_init, loss_val, X_val, eta = 0.01, beta = 0.9, bias_correction = False, line_search = False, t = 1):
    """
    Gradient Descent with Momentum optimization algorithm for updating the values of a given variable.

    Args:
      - loss_func: Callable, the loss function that computes the loss given the current variable values.
      - X_init: tf.Variable, the initial variable values to be updated.
      - eta: float, the learning rate used to control the size of the update steps (default is 0.01).
      - beta: float, the momentum hyperparameter used to control the momentum effect (default is 0.9).
      - loss_val: list(), empty list used to save the loss values at each iteration.
      - line_search: boolean, to determine if we will search for the optimal value of the learning rate at each step (default is False).
      - bias_correction: boolean, to limit the first iteration from being biased (default is False).
      - t: int, the iteration number.

    Returns:
      None
    """

    with tf.GradientTape(persistent=True) as tape:
        # Compute the current loss
        current_loss = loss_func(X_init)

    # Compute the gradient of the loss with respect to the variables
    dx = tape.gradient(current_loss, X_init)

    # Compute the estimate (momentum)
    V = beta * V_init + (1 - beta) * dx

    # bias correction
    if bias_correction:
        V_corrected = V / (1 - tf.pow(beta, t))
    else:
        V_corrected = V
    
    # line search (finding the optimal value for the learning rate)
    if line_search:
      if t == 1:
        for i in range(1000): 
          eta = LineSearch(X_init, loss_func, tf.Variable(eta)).numpy()
      else:
        for i in range(10):
          eta = LineSearch(X_init, loss_func, tf.Variable(eta)).numpy()

    # Update the variables using the momentum update rule
    X_init.assign_sub(eta * V_corrected)

    # Update the moment estimate (momentum)
    V_init.assign(V)

    # calc the loss value and append it to the list
    loss_val.append(loss_func(X_init))

    # append the new value of X to the list
    X_val.append(X_init.numpy())
