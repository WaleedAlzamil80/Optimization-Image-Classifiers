import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot (loss_function, X_vals, loss_vals ):
  # Define the range of X1 and X2 values to plot the StyblinskiTang function
  rangeX1, rangeX2 =   abs(X_vals[0])

  stepX1 = rangeX1 / 500
  stepX2 = rangeX2  / 500

  offset1 = (2 * (rangeX1))*0.05
  offset2 = (2 * (rangeX2))*0.05
 

  X1_range = np.arange(-rangeX1 - offset1  ,rangeX1 + offset1 + stepX1 ,       stepX1     ) 
  X2_range = np.arange(-rangeX1 - offset2 , rangeX1 + offset2  + stepX2 ,         stepX2    )
  X1_grid, X2_grid = np.meshgrid(X1_range, X2_range)
  Z = loss_function([X1_grid, X2_grid])


  # Plot the StyblinskiTang function
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X1_grid, X2_grid, Z, cmap=cm.coolwarm, alpha=0.6)
  ax.plot_wireframe(X1_grid, X2_grid, Z, color='black', linewidth=0.1,  alpha=0.2)
  ax.set_xlabel('X1')
  ax.set_ylabel('X2')
  ax.set_zlabel('Loss Function Value')
  ax.xaxis.labelpad = 20



  # Plot the trace of X in 3D
  X1 = [x[0] for x in X_vals]
  X2 = [x[1] for x in X_vals]

  Z_trace = loss_vals



  # Compute the directions of the arrows
  X1_diff = np.diff(X1)
  X2_diff = np.diff(X2)
  Z_trace_diff = np.diff(Z_trace)

  # Plot the arrows
  ax.quiver(X1[:-1], X2[:-1], Z_trace[:-1], X1_diff, X2_diff, Z_trace_diff, 
            length=1, color='red', arrow_length_ratio=0.0001, linewidths=2)

  # Plot the last point of the X trace as a red sphere
  ax.scatter(X1[-1], X2[-1], Z_trace[-1], s=100, color='red', marker='x')


  # Get the minimum value of Z to adjust the plot limits
  min_Z = np.min(Z)
  max_Z = np.max(Z)


  offset = (max_Z - min_Z)*0.2


  ax.set_zlim(min_Z  , max_Z - offset)

  plt.show()