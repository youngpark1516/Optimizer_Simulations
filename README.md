# Optimizer Simulations
## Introduction

## Imports
Following are the used libraries in the project.

### Numpy
NumPy is a third-party open-source library specializing in arrays and matrices. The library was mainly used for calculations involving the polynomial class.

### Matplotlib
Matplotlib is a third-party open-source library specializing in visualization. The library was employed for the visualization of polynomials and plotting the change of prediction of optimizers.

### Random
Random is a standard Python library that creates pseudo-random numbers. The library was used in the implementation of initializing polynomial instance and randomizing starting point of simulations.

## Polynomial Class
The polynomial class was made to simulate a hypothetical 'loss function' which the optimizer will aim to minimize the function value of. While a more efficient and extensive polynomial class can be easily used from Numpy library, I aimed to use only a few functionalities. For that reason, I attempted to make a simple polynomial function as a small challenge.

### Instance attribute
self.coefficients is a Numpy array consisting of coefficients, starting from the lowest power, 0, and incrementing by 1.

### Initialization
In the process of implementing the initialization of polynomial class, I thought that it would be interesting to have the 

I implemented method overloading for the initialization of this class. Although Python doesn't support method overloading like Java or other languages, it can be done by setting the default value of parameters as None.

### Methods


## Optimizers

### Gradient Descent (GD)

### Adaptive Gradient Descent (AdaGrad)

### Root Mean Square Propagation (RMSProp)

### Adaptive Learning Rate Method (AdaDelta)

### Adaptive Moment Estimation (Adam)

## Implementation

## Conclusion
