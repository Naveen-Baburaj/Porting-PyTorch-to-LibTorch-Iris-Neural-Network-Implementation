# Porting a PyTorch Neural Network from Python to C using LibTorch – Iris Classification

## Overview
This project demonstrates how a neural network implemented in **Python using PyTorch** can be **ported to C++ using LibTorch** while preserving the same machine learning architecture.

The model performs **multi-class classification on the Iris dataset**, predicting flower species using four input features.

The goal is to show the process of translating a PyTorch-based machine learning workflow into its equivalent implementation in C++ using LibTorch.

---

## Dataset

The project uses the classic **Iris dataset**, which contains:

- 150 samples  
- 4 numerical features:
  - sepal length
  - sepal width
  - petal length
  - petal width
- 3 classes:
  - Setosa
  - Versicolor
  - Virginica
