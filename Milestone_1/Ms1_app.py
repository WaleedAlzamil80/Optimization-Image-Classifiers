import streamlit as st
import numpy as np
import inspect
import tensorflow as tf
import matplotlib.pyplot as plt
# # importing the optimization algorithms
# from Optimization_Algorithms.Ms1_AdadeltaOpt import *
# from Optimization_Algorithms.Ms1_AdagradOpt import *
# from Optimization_Algorithms.Ms1_AdamOpt import *
from Optimization_Algorithms.Ms1_GDMOpt import *
# from Optimization_Algorithms.Ms1_SGDOpt import *
# importing test function
# from Loss_Functions.Ms1_3HumpCamel_FunctionLoss import *
# from Loss_Functions.Ms1_EggHolder_FunctionLoss import *
from Loss_Functions.Ms1_MATYAS_FunctionLoss import *
# from Loss_Functions.Ms1_Michalewicz_FunctionLoss import *
# from Loss_Functions.Ms1_StyblinskiTang_FunctionLoss import *
# from Loss_Functions.Ms1_Trid_FunctionLoss import *


# Define the optimization algorithm

# Define the test functions

def plot_optimization(X_val, Y_val):
    fig, ax = plt.subplots()
    X1, X2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    Z = MATYAS([X1, X2])
    ax.contour(X1, X2, Z, levels=np.logspace(-0.5, 3.5, 20))
    ax.plot(*tf.transpose(X_val), Y_val, '-o', color='r')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    st.pyplot(fig)

def main():
    st.title("Matyas Function Optimization with GDM Algorithm CI Project")


    st.markdown(
        """
        <style>
        body {
                background-color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Set up initial values
    X_init = tf.Variable(tf.ones((2,)))
    V_init = tf.Variable(tf.zeros((2,)))
    loss_val = []
    X_val = []

    # Set up optimization parameters
    eta = st.slider("LearningRate", 0.001, 1.0, 0.01, 0.001)
    beta = st.slider("Momentum Hyperparameter", 0.0, 1.0, 0.9, 0.01)
    bias_correction = st.checkbox("Bias Correction", value=False)
    line_search = st.checkbox("Line Search", value=False)

    # Set up learning rate decay parameters
    decay_type = st.selectbox("Learning Rate Decay Type", ["None", "Time-Based", "Step-Based", "Exponential", "Performance"])
    if decay_type != "None":
        decay_rate = st.slider("Decay Rate", 0.0, 1.0, 0.9, 0.01)
    else:
        decay_rate = None

    # Run optimization
    for i in range(500):
        GDM(MATYAS, X_init, V_init, loss_val, X_val, eta=eta, beta=beta, bias_correction=bias_correction,
            line_search=line_search, t=i, decay_type=decay_type, decay_rate=decay_rate)

    # Plot results
    st.write("Final Result:", X_val[-1])
    st.write("Final Loss:", loss_val[-1])
    plot_optimization(np.array(X_val), loss_val)

if __name__ == '__main__':
    main()