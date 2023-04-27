import tensorflow as tf
import numpy as np

def Adagrad(loss_func, X_init, S_init, loss_val, X_val, eta = 0.01, eps = 1e-8):
  """
  Adabtive gradient(Adagrad) optimization algorithm for updating the values of a given variable.

  Args:
    - loss_func: Callable, the loss function that computes the loss given the current variable values.
    - X_init: tf.Variable, the initial variable values to be updated.
    - eta: float, the learning rate used to control the size of the update steps (default is 0.01).
    - S_init: tf.Variable, the initial values for the second moment estimates (RMSprop).
    - eps: float, Ensure S in not zero by adding this number to it (default is 1e-8).
    - loss_val: list(), empty list used to save the loss values at each iteration

  Returns:
    None
  """

  with tf.GradientTape(persistent=True) as tape:
    # Compute the current loss
    current_loss = loss_func(X_init)

  # Compute the gradient of the loss with respect to the variables
  dx = tape.gradient(current_loss, X_init)

  # Compute the estimate (RMSprop)
  S = S_init + dx**2      # RMSprop


  # Update the variables using the Adagrad update rule
  X_init.assign_sub(eta * dx / (tf.sqrt(S) + eps))
    
  # update the moment estimate (RMSprop)
  S_init.assign(S)

  # calc the loss value and append it to the list
  loss_val.append(loss_func(X_init))

  # append the new value of X to the list
  X_val.append(X_init.numpy())
