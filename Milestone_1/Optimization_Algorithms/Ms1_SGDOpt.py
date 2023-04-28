import numpy as np
import tensorflow as tf
from Optimization_Algorithms.LineSearch import *
#from DecayLearningRate import *

def SGD(loss_func, X_init, loss_val, X_val, eta = 0.01, line_search = False):
  """
  Stochastic Gradient Descent (SGD) optimization algorithm for updating the values of a given variable.

  Args:
    - loss_func: Callable, the loss function that computes the loss given the current variable values.
    - X_init: tf.Variable, the initial variable values to be updated.
    - eta: float, the learning rate used to control the size of the update steps (default is 0.01).
    - loss_val: list(), empty list used to save the loss values at each iteration
    - line_search: boolean, to determine if we will search for the optimal value of the learning rate at each step (default is False).
  
  Returns:
    None
  """

  with tf.GradientTape(persistent=True) as tape:
    # Compute the current loss
    current_loss = loss_func(X_init)

  # Compute the gradient of the loss with respect to the variables
  dx = tape.gradient(current_loss, X_init)
  
  # line search (finding the optimal value for the learning rate)
  if line_search:
    for i in range(10):
      eta = LineSearch(X_init, loss_func, tf.Variable(eta)).numpy()

  # Update the variables using the SGD update rule
  X_init.assign_sub(eta * dx)

  # calc the loss value and append it to the list
  loss_val.append(loss_func(X_init.numpy()))

  # append the new value of X to the list
  X_val.append(X_init.numpy())
