import streamlit as st
import numpy as np
# importing the optimization algorithms
from Optimization_Algorithms.Ms1_AdadeltaOpt import Adadelta
from Optimization_Algorithms.Ms1_AdagradOpt import Adagrad
from Optimization_Algorithms.Ms1_AdamOpt import Adam
from Optimization_Algorithms.Ms1_GDMOpt import GDM
from Optimization_Algorithms.Ms1_SGDOpt import SGD
# importing test function
from Loss_Functions.Ms1_3HumpCamel_FunctionLoss import ThreeHumpCamel
from Loss_Functions.Ms1_EggHolder_FunctionLoss import EggHolder
from Loss_Functions.Ms1_MATYAS_FunctionLoss import MATYAS
from Loss_Functions.Ms1_Michalewicz_FunctionLoss import Michalewicz
from Loss_Functions.Ms1_StyblinskiTang_FunctionLoss import StyblinskiTang
from Loss_Functions.Ms1_Trid_FunctionLoss import Trid

st.write=st.write()
# Define the optimization algorithms
optimization_algorithms = {
    'AdadeltaOpt Algorithm': Adadelta,
    'AdagradOpt Algorithm': Adagrad,
    'AdamOpt Algorithm': Adam,
    'GDMOpt Algorithm' : GDM,
    'SGDOpt Algorithm' : SGD
}

# Define the test functions
test_functions = {
    'ThreeHumpCamel Function': ThreeHumpCamel,
    'EggHolder Function': EggHolder,
    'MATYAS Function': MATYAS,
    'Michalewicz Function': Michalewicz,
    'StyblinskiTang Function': StyblinskiTang,
    'Trid Function': Trid ,
}

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

    # Add a slider to adjust the number of iterations
    num_iterations = st.slider('Number of Iterations', 10, 100, 1000)

    # Run the selected optimization algorithm on the selected test function
    results = optimization_algorithms[optimization_algorithm](test_functions[test_function], num_iterations)

    # Visualize the results using a scatter plot
    x = np.arange(num_iterations)
    y = results['values']
    st.line_chart({'x': x, 'y': y})

# Run the Streamlit app
if __name__ == '__main__':
    app()