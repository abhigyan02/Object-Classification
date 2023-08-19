# CIFAR-10 Image Classification using Convolutional Neural Networks

This repository contains a Python script for building and training a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset. The goal of the project is to demonstrate how to use Keras to create a deep learning model for image classification.

## About the Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The goal is to classify these images into their respective categories using a CNN model.

## Project Overview

- **Dataset**: The CIFAR-10 dataset is loaded and preprocessed to be used for training and testing the CNN model.

- **Model Architecture**: The model is built using the Keras library with a sequential stack of Convolutional-BatchNormalization blocks, MaxPooling, and Dropout layers.

- **Training and Evaluation**: The model is compiled with an SGD optimizer and categorical cross-entropy loss. It's trained using the training data and evaluated on the test data.

- **Results**: The model's accuracy is printed after evaluation, giving an indication of its performance in classifying the object categories.


## Prerequisites

- Python 3.x
- TensorFlow and Keras libraries
- Jupyter Notebook (optional for experimentation)


