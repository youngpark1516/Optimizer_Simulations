#Imports
import random
import matplotlib.pyplot as plt
import math
import numpy as np

def gen_pol(degree):
    """
    Generate random polynomial

    :param degree: The degree of a random polynomial
    :return: A list of coefficients
    """
    coefficients = []
    for i in range (degree+1):
        if(i == 0):
            coefficients.append(random.uniform(0,10))
        else:
            coefficients.append(random.uniform(-10,10))
    return coefficients
def use_pol(cof, val):
    """
    Applies polynomial on a value and returns it

    :param cof: A list of coefficients representing a polynomial
    :param arg: A value
    :return: Resulting function value
    """
    func_val = 0
    for i in range(len(cof)):
        func_val += cof[i]*val**(len(cof)-i-1)
    return func_val
def display_pol(cof):
    """
    Displays a polynomial

    :param cof:
    :return:
    """
    polynomial = ""
    for i in range(len(cof)):
        co = str(round(cof[i],2))
        var = "x^"+str(len(cof)-i-1)
        if(cof[i] == 0):
            continue
        if(cof[i] == 1 and not i == len(cof)-1):
            co = ""
        if(i == len(cof)-1):
            var = ""
        elif(i == len(cof)-2):
            var = "x"
        polynomial += co+var+ " + "
    return polynomial[:-3]
def y_range(cof, valrange, stepsize = 1.0):
    arg = valrange[0]
    arguments = []
    function_values = []
    while arg < valrange[1]:
        arguments.append(arg)
        function_values.append(use_pol(cof,arg))
        arg += stepsize
    return arguments, function_values
def draw(x,y, xlabel = "x", ylabel = "y",title = ""):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
def grad(cof, x, step = 0.00001):
    return (use_pol(cof,x+step)-use_pol(cof,x))/step
def grad_descent(cof, loc, lr = 0.01):
    newloc = loc - lr*grad(cof,loc)
    return newloc
def simulate_optimizer(optimizer, co, bounds=[-5,5], start_loc = -5):
    x, y = y_range(co, bounds, stepsize= 0.01)
    plt.plot(x,y)
    loc = start_loc
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(display_pol(co))
    for i in range(30):
        xcor1 = loc
        ycor1 = use_pol(co, loc)
        loc = optimizer(co,loc, lr = 0.04)
        xcor2 = loc
        ycor2 = use_pol(co,loc)
        plt.plot([xcor1,xcor2],[ycor1,ycor2],marker = "o",color = "b")
    #     print(loc)
        plt.pause(0.5)
    plt.show()

simulate_optimizer(grad_descent, [0.1,0.02, -1,-1,1])