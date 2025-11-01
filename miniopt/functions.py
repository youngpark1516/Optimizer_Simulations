import numpy as np

def quartic(x):
    return x**4 - 3*x**3 + 2

def d_quartic(x):
    return 4*x**3 - 9*x**2

def rosenbrock_1d_like(x, a=1, b=100):
    # silly 1D slice of rosenbrock
    return (a - x)**2 + b*(0 - x**2)**2

def d_rosenbrock_1d_like(x, a=1, b=100):
    return -2*(a - x) - 4*b*x**3
