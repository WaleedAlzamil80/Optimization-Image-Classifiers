# Optimization Image Classifiers
This project is divided into two milestones. The first milestone involves comparing the performance of different optimization algorithms on different types of test functions. The second milestone involves creating and training a multilayer neural network to classify images into their corresponding categories.

 The results from the first milestone will be used to select the most effective optimization algorithm for training the neural network for the second milestone. The final goal is to achieve high accuracy in image classification using the selected algorithm and neural network architecture.

## Milestone 1:
Optimization algorithms play a crucial role in many fields of science and engineering, from machine learning and data science to operations research and control engineering. These algorithms are designed to find the optimal solution to a given problem by iteratively improving a candidate solution based on some objective function.

However, there are many different optimization algorithms, each with its own strengths and weaknesses. Some algorithms are better suited for certain types of problems than others, and some may converge faster or more reliably than others. Therefore, it is important to compare and evaluate different optimization algorithms on various types of test functions to determine which algorithm is best suited for a given problem.

In this milestone, we will address the problem of comparing different optimization algorithms on various types of test functions.
We will explore popular optimization algorithms

### Methods and Algorithms:
We will compare the performance of the following optimization algorithms:

1. Stochastic Gradient Descent
2. Gradient Descent with Momentum
3. Adagrad
4. Adadelta
5. Adam

Additionally, we have experimented with some improvements to some of the previously stated algorithms in the following:
1. Bias correction
2. Learning rate decay
3. Line search algorithm

## Milestone 2:

In Milestone 2 we're going to build 4 different deep learning architectures on a CiFar Image 100 below you can find the dataaset link.
Since our problem dataset is an image classification.
These **`CNN architectures`** have been widely used and adapted for various image classification tasks. They demonstrate the effectiveness of deep learning in computer vision and have paved the way for many advancements in the field.

AlexNet, VGG, ResNet, and LeNet are some of the popular convolutional neural network (CNN) architectures used for image classification tasks.

* **AlexNet**:
AlexNet is a CNN architecture proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in 2012. It was the winning architecture in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012, achieving a top-5 error rate of 15.3%. AlexNet introduced several key innovations, such as the use of ReLU activation functions, overlapping pooling, and dropout regularization, which made it possible to train deeper and more complex neural networks.

* **VGG**:
VGG (Visual Geometry Group) is a CNN architecture proposed by the Visual Geometry Group at the University of Oxford in 2014. VGG is known for its simplicity and uniformity in architecture, with a total of 16 convolutional and fully connected layers. VGG achieved high accuracy in the ILSVRC 2014 competition, with a top-5 error rate of 7.3%. VGG's architecture is characterized by using only 3x3 filters, which allows for a deeper network and better feature extraction.

* **ResNet**:
ResNet (Residual Network) is a CNN architecture proposed by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun in 2015. ResNet is known for its use of residualconnections to address the vanishing gradient problem that arises in very deep neural networks. ResNet achieved state-of-the-art performance in the ILSVRC 2015 competition, with a top-5 error rate of 3.57%. The residual connections in ResNet allow for the flow of information from one layer to another, even when the gradient becomes very small. This enables ResNet to train very deep networks with hundreds of layers.

* **LeNet**: is a pioneering convolutional neural network (CNN) architecture proposed by Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner in 1998. It was designed for handwritten digit recognition and was one of the first successful applications of deep learning in computer vision. The LeNet architecture consists of a series of convolutional and pooling layers, followed by fully connected layers. The input to the network is a grayscale image of size 32x32 pixels, and the output is a probability distribution over the 10 possible digit classes (0-9).

**Dataset**:

**`CIFAR-100`** is a dataset of 60,000 32x32 color images, belonging to 100 fine-grained object categories with 600 images per category. The dataset is split into 50,000 training images and 10,000 test images.

Each image in the CIFAR-100 dataset belongs to only one of the 100 categories, and the categories are organized into 20 superclasses, each containing five fine-grained categories. The fine-grained categories are more specific than those in the related CIFAR-10 dataset, which has 10 coarse-grained categories.

The CIFAR-100 dataset is commonly used as a benchmark for image classification tasks, particularly in the field of deep learning. It is often used to evaluate the performance of various convolutional neural network architectures and training techniques.

The CIFAR-100 dataset is challenging because the images are small and low resolution, making it difficult to recognize fine-grained details. Additionally, the dataset contains many visually similar classes, making it hard to distinguish between them. However, the dataset is also diverse and contains a wide range of object categories, making it a useful testbed for evaluating the generalization performance of image classification models.

https://www.cs.toronto.edu/~kriz/cifar.html


## Future work:
We plan to apply the extension of Adam that we used in Milestone 1 to train the models that we developed in Milestone 2. Then, we will compare the performance of the models and assess whether the LineSearch algorithm is worth the additional computation. We will also investigate the effect of different loss functions on the results.
