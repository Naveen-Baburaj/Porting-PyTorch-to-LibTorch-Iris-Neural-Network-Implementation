# Porting a PyTorch Neural Network from Python to C using LibTorch – Iris Classification

## Overview
This project demonstrates how a neural network implemented in **Python using PyTorch** can be **ported to C++ using LibTorch** while preserving the same machine learning architecture.

The model performs **multi-class classification on the Iris dataset**, predicting flower species using four input features.

The goal is to show the process of translating a PyTorch-based machine learning workflow into its equivalent implementation in C++ using LibTorch.

---

## Project Structure

### Python (PyTorch)
The Python implementation follows the standard PyTorch workflow:

- Loading and preprocessing the Iris dataset  
- Defining the neural network with `torch.nn`  
- Training the model  
- Making predictions  

This version serves as the **reference implementation**.

### C++ (LibTorch)
The C++ version recreates the same model using **LibTorch**, the official PyTorch C++ API.

It demonstrates how the same neural network architecture can be implemented and executed in C++, enabling PyTorch models to run in **non-Python environments or performance-critical systems**.

---

## Screencast

To see how the **Python and C++ implementations compare**, watch the screencast below.

It walks through both versions of the code and highlights how the PyTorch model was translated to LibTorch.

👉 **Watch the screencast comparing both implementations here:**  
https://screenrec.com/share/Sotua6geVf

---

## Source of the Python Implementation

The Python code used in this project is adapted from the Appsilon tutorial:

**PyTorch Neural Network Tutorial**  
https://www.appsilon.com/post/pytorch-neural-network-tutorial

In this project, the Python code is used as the **reference implementation**, which is then ported to **C++ using LibTorch**.

---

## Technologies Used

- Python  
- PyTorch  
- C++  
- LibTorch (PyTorch C++ API)  
- Iris Dataset  

---


## Author

Naveen Baburaj
