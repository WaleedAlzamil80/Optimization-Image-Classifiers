import tensorflow as tf
import numpy as np
from LineSearch import *

def Adam(loss_func, X_init, V_init, S_init, loss_val, X_val, eta = 0.01, beta_1 = 0.9, beta_2 = 0.99, eps = 1e-8, line_search = False, bias_correction = False, t = 0):
    """
    Adam optimization algorithm for updating the values of a given variable.

    Args:
        - loss_func: Callable, the loss function that computes the loss given the current variable values.
        - X_init: tf.Variable, the initial variable values to be updated.
        - V_init: tf.Variable, the initial values for the first moment estimates (momentum).
        - S_init: tf.Variable, the initial values for the second moment estimates (RMSprop).
        - eta: float, the learning rate used to control the size of the update steps (default is 0.01).
        - beta_1: float, the amount used to control how the momentum (exponentially weighted moving averages). 
                  depending on the last values (default is 0.9).
        - beta_2: float, the amount used to control how the learning rate depend on the last values (default is 0.99).
        - eps: float, Ensure S in not zero by adding this number to it (default is 1e-8).
        - line_search: boolean, to determine if we will search for the optimal value of the learning rate at each step (default is False).
        - bias_correction: boolean, to limit the first iteration from being biased (default is False).
        - t: int, iteration number for bias correction.
        - loss_val: list(), empty list used to save the values of loss function at each iteration.
        - X_val: list(), empty list used to save the values of X at each iteration.
        - t: int, the iteration number.

    Returns:
        None
    """
    
    with tf.GradientTape(persistent=True) as tape:
        # Compute the current loss
        current_loss = loss_func(X_init)

    # Compute the gradient of the loss with respect to the variables
    dx = tape.gradient(current_loss, X_init)

    # Compute the first and second moment estimates (momentum and RMSprop)
    V = (beta_1 * V_init) + (1 - beta_1) * dx      # momentum
    S = (beta_2 * S_init) + (1 - beta_2) * (dx**2) # RMSprop
    t = t + 1

    # bias correction
    if bias_correction:
        V_corrected = V / (1 - tf.pow(beta_1, t))
        S_corrected = S / (1 - tf.pow(beta_2, t))
    else:
        V_corrected = V
        S_corrected = S
    
    # line search (finding the optimal value for the learning rate)
    if line_search:
      if t == 1:
        for i in range(1000): 
          eta = LineSearch(X_init, loss_func, tf.Variable(eta)).numpy()
      else:
        for i in range(10):
          eta = LineSearch(X_init, loss_func, tf.Variable(eta)).numpy()

    # update the variables using Adam update rule
    X_init.assign_sub(eta * V_corrected / (tf.sqrt(S_corrected) + eps))

    # update the first and second moment estimates (momentum and RMSprop)
    V_init.assign(V)
    S_init.assign(S)

    # calc the loss value and append it to the list
    loss_val.append(loss_func(X_init))

    # append the new value of X to the list
    X_val.append(X_init.numpy())
