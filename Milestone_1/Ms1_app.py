import streamlit as st
import numpy as np
import inspect
# importing the optimization algorithms
from Optimization_Algorithms.Ms1_AdadeltaOpt import *
from Optimization_Algorithms.Ms1_AdagradOpt import *
from Optimization_Algorithms.Ms1_AdamOpt import *
from Optimization_Algorithms.Ms1_GDMOpt import *
from Optimization_Algorithms.Ms1_SGDOpt import *
# importing test function
from Loss_Functions.Ms1_3HumpCamel_FunctionLoss import *
from Loss_Functions.Ms1_EggHolder_FunctionLoss import *
from Loss_Functions.Ms1_MATYAS_FunctionLoss import *
from Loss_Functions.Ms1_Michalewicz_FunctionLoss import *
from Loss_Functions.Ms1_StyblinskiTang_FunctionLoss import *
from Loss_Functions.Ms1_Trid_FunctionLoss import *


# Define the optimization algorithms
optimization_algorithms = {
    'Adadelta': Adadelta,
    'Adagrad': Adagrad,
    'Adam': Adam,
    'GDM' : GDM,
    'SGD' : SGD
}

# Define the test functions
test_functions = {
    'ThreeHumpCamel': ThreeHumpCamel,
    'EggHolder': EggHolder,
    'MATYAS': MATYAS,
    'Michalewicz': Michalewicz,
    'StyblinskiTang': StyblinskiTang,
    'Trid': Trid ,
}

X_init = tf.Variable([[-130.0], [-130.0]])
V_init = tf.Variable([[0.0], [0.0]])
S_init = tf.Variable([[0.0],[ 0.0]])
loss_val = [test_function(X_init)]
X_val = [X_init.np()]

# Define the Streamlit app
def app():
    st.title('Comparing Optimization Algorithms on Test Functions')

    # Add a dropdown to select the optimization algorithm
    optimization_algorithm = st.selectbox(
        'Select Optimization Algorithm',
        list(optimization_algorithms.keys()))

    # Add a dropdown to select the test function
    test_function = st.selectbox(
        'Select Test Function',
        list(test_functions.keys()))

    num_iterations = st.slider('Number of Iterations', 10, 100)
    

    for i in range(num_iterations):
        if optimization_algorithm == "SGD":
            result = SGD(test_function, X_init, loss_val, X_val)
        elif optimization_algorithm == "GDM":
            result = GDM(test_function,  X_init, V_init, loss_val, X_val)
        elif optimization_algorithm == "Adagrad":
            result = Adagrad(test_function, X_init, S_init, loss_val, X_val)
        elif optimization_algorithm == "Adadelta":
            result = Adadelta(test_function, X_init, S_init, loss_val, X_val)
        elif optimization_algorithm == "Adam":
            result = Adam(test_function, X_init, V_init, S_init, loss_val, X_val)    

    for i in range(num_iterations):
        if test_function == "ThreeHumpCamel":
            result = ThreeHumpCamel()
        elif test_function == "EggHolder":
            result = EggHolder()
        elif test_function == "MATYAS":
            result = MATYAS()
        elif test_function == "Michalewicz":
            result = Michalewicz()
        elif test_function == "StyblinskiTang":
            result = StyblinskiTang()

        # Run the selected optimization algorithm on the selected test function with the user input parameters

        # Visualize the results using a scatter plot
        x = np.arange(num_iterations)
        y = result[[test_function,optimization_algorithm, 0.01,False,False ,X_val,loss_val]]
        st.line_chart({'x': x, 'y': y})

# Run the Streamlit app
if __name__ == '__main__':
    app()