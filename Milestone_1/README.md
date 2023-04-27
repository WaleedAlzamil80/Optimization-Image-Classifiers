# Optimization Algorithms for Deep Learning Models

This repository contains code and a report for a project that compares the performance of several optimization algorithms commonly used in deep learning, including Stochastic Gradient Descent (SGD), Gradient Descent with Momentum (GDM), Adagrad, Adadelta, and Adam. The report evaluates the performance of these algorithms on various test functions with different shapes and numbers of local minima.

## Abstract

Deep learning has revolutionized the field of artificial intelligence, enabling machines to learn from data and make predictions with unprecedented accuracy. Optimization algorithms play a crucial role in improving the performance of deep learning models by minimizing the loss function and updating the model’s parameters. The field of deep learning has seen significant advancements in recent years, with optimization algorithms playing a crucial role in improving the performance of deep learning models. In this report, we compare the performance of several optimization algorithms commonly used in deep learning.

## Introduction

### Intro

Optimization is an important topic in many areas of science and engineering. There are a variety of optimization algorithms available, each with its own strengths and weaknesses. In this project, we compare the performance of five popular optimization algorithms: Stochastic Gradient Descent (SGD), Gradient Descent with Momentum (GDM), Adagrad, Adadelta, and Adam. We evaluate the performance of these algorithms on various test functions with different shapes and numbers of local minima.

Our goal is to provide insights into the strengths and weaknesses of these algorithms and to offer recommendations for future research in this area.

### Problem Statement

The optimization problem that we considered was as follows:

Minimize f(x)

where x ∈ Rn is the decision variable, f : Rn → R is the objective function.

## Test Functions

### Intro

We used test functions from the SFU optimization test function library, including the Eggholder function, Trid function, Matyas function, Three-Hump Camel function, Michalewicz function, and Styblinski-Tang function. These test functions represent different types of objective functions, including those with many local minima, bowl-shaped functions, plate-shaped functions, Valley-Shaped, Valley-Shaped, and other shapes. Additionally, we tested each algorithm on each function in a range of search spaces with different dimensions.

## How to Use

To run the code, clone the repository and install the necessary dependencies using the following command:
`pip install -r requirements.txt`
Then you can use all the functions defined in this repository.

## Conclusion

we have compared the performance of several optimization algorithms commonly used in deep learning on different kind of Loss Functions for more details, The report can be found in the `report/` directory.
