# Imports
import numpy as np
import matplotlib.pyplot as plt
import random
class polynomial:
    """
    Polynomial class
    """
    def __init__(self, coefficients=None, degree=None):
        """
        Initializes polynomial instance. Requires one of the two parameters, coefficients prioritized

        :param coefficients: A list of coefficients, from the lowest power (x^0)
        :param degree: For randomized polynomial, integer indicating the degree of the polynomial
        """
        if coefficients:
            self.coefficients = np.array(coefficients)
        elif not degree == None:
            self.coefficients = np.random.randint(-10, 10, degree)
        else:
            raise TypeError('Class polynomial requires either an array of coefficients or degree of polynomial')

    def __repr__(self):
        """
        :return: Representation of polynomial variable
        """
        l = list(map(lambda x: str(x) + 'x^', self.coefficients))
        return ' + '.join([l[i] + str(i) for i in range(len(l))])

    def apply(self, value):
        """
        Maps a value through the polynomial function

        :param value: A float or an integer
        :return: The function value
        """
        powers = np.arange(0, len(self.coefficients))
        return np.sum(self.coefficients * np.power(value, powers))

    def apply_domain(self, start, end, step=1.0):
        """
        Applies the function throughout a given domain

        :param start: Number indicating the lower bound (inclusive)
        :param end: Number indicating the upper bound (exclusive)
        :param step: Step between each value (lower => smoother curve), default 1
        :return: Two numpy arrays, the domain and range
        """
        vectorized_apply = np.vectorize(self.apply)
        domain = np.arange(start, end, step)
        return domain, vectorized_apply(domain)

    def draw(self, start, end, step=1.0, x_label='x', y_label='y', title=None, show = True):
        """
        Plots the polynomial

        :param start: Number indicating the lower bound
        :param end: Number indicating the upper bound
        :param step: Number indicating step between each value (lower => smoother curve), default 1
        :param x_label: String label for x-axis, default 'x'
        :param y_label: String label for y-axis, default 'y'
        :param title: String, default is representation of the polynomial
        :param show: Boolean on showing the plot immediately
        :return:
        """
        domain, f_range = self.apply_domain(start, end, step)
        plt.plot(domain, f_range)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if title:
            plt.title(title)
        else:
            plt.title(self.__repr__())
        if show:
            plt.show()

def Gradient_Descent(function_value, value, function_prev= None, prev=None, learning_rate=0.01):
    """
    Gradient descent

    :param function_value: Result of the current guess
    :param value: Value of the current prediction
    :param function_prev: Result of the previous guess, default None
    :param prev: Value of the previous prediction, default None
    :param learning_rate: Learning rate, default 0.01
    :return: Current prediction, updated prediction
    """

    try:
        return value, value - learning_rate * ((function_value - function_prev) / (value - prev))
    except:
        return value, value - value * learning_rate

def AdaGrad(function_value, value, function_prev=None, prev=None, alpha = 0.1, epsilon = 1e-7, learning_rate = 0.001):
    """
    Adaptive Gradient Descent

    :param function_value: Result of the current guess
    :param value: Value of the current prediction
    :param function_prev: Result of the previous guess, default None
    :param prev: Value of the previous prediction, default None
    :param alpha: Cumulated gradients (for slowing down learning), default 0.1
    :param epsilon: Small number to prevent ZeroDivisionError, default 1e-7
    :param learning_rate: Learning rate, default 0.001
    :return: Current prediction, updated prediction, updated alpha
    """
    try:
        gradient = ((function_value - function_prev) / (value - prev))
        alpha += gradient**2
        learning_rate = learning_rate/np.sqrt(alpha+epsilon)
        return value, value - learning_rate*gradient, alpha
    except:
        return value, value - value * learning_rate, alpha

def RMSProp(function_value, value, function_prev=None, prev=None, alpha = 0.1, rho = 0.9, epsilon = 1e-7, learning_rate = 0.001):
    """
    Root Mean Squared Propagation

    :param function_value: Result of the current guess
    :param value: Value of the current prediction
    :param function_prev: Result of the previous guess, default None
    :param prev: Value of the previous prediction, default None
    :param alpha: Cumulated gradients (for slowing down learning), default 0.1
    :param rho: Discounting factor for old gradients, default 0.9
    :param epsilon: Small number to prevent ZeroDivisionError, default 1e-7
    :param learning_rate: Learning rate, default 0.001
    :return: Current prediction, updated prediction, updated alpha
    """
    try:
        gradient = ((function_value - function_prev) / (value - prev))
        alpha = rho*alpha + (1-rho)*gradient**2
        learning_rate = learning_rate/np.sqrt(alpha+epsilon)
        return value, value - learning_rate*gradient, alpha
    except:
        return value, value - value * learning_rate, alpha

def AdaDelta(function_value, value, function_prev=None, prev=None, alpha = 0.1, delta = 0, rho = 0.9, epsilon = 1e-7):
    """
    Adaptive Learning Rate Method

    :param function_value: Result of the current guess
    :param value: Value of the current prediction
    :param function_prev: Result of the previous guess, default None
    :param prev: Value of the previous prediction, default None
    :param alpha: Cumulated gradients (for slowing down learning), initially 0.1
    :param delta: Cumulated rescaled gradients, initially 0
    :param rho: Discounting factor for old gradients, default 0.9
    :param epsilon: Small number to prevent ZeroDivisionError, default 1e-7
    :return: Current prediction, updated prediction, updated alpha, updated delta
    """
    try:
        gradient = ((function_value - function_prev) / (value - prev))
        alpha = rho * alpha + (1 - rho) * gradient ** 2
        rescaled_gradient = gradient*(np.sqrt(delta + epsilon)/np.sqrt(alpha + epsilon))
        delta = rho * delta + (1 - rho) * rescaled_gradient ** 2
        print(alpha, delta)
        return value, value - rescaled_gradient, alpha, delta
    except:
        gradient = 1
        alpha = rho * alpha + (1 - rho) * gradient ** 2
        rescaled_gradient = np.sqrt(epsilon)/np.sqrt(alpha + epsilon)
        delta = rho * delta + (1 - rho) * rescaled_gradient ** 2
        return value, value-rescaled_gradient, alpha, delta

def Adam(function_value, value, function_prev=None, prev=None, alpha = 0, beta = 0, rho1 = 0.9, rho2 = 0.999, epsilon = 1e-7, learning_rate = 0.001):
    """
    Adaptive Moment Estimation

    :param function_value: Result of the current guess
    :param value: Value of the current prediction
    :param function_prev: Result of the previous guess, default None
    :param prev: Value of the previous prediction, default None
    :param alpha: Cumulated gradients (moment estimate), initially 0
    :param beta: Cumulated squared gradients (variance moment estimate), initially 0
    :param rho1: Discounting factor for moment estimate, default 0.9
    :param rho2: Discounting factor for variance moment estimate, default 0.999
    :param epsilon: Small number to prevent ZeroDivisionError, default 1e-7
    :param learning_rate: Learning rate, default 0.001
    :return: Current prediction, updated prediction, updated alpha, updated beta
    """
    try:
        gradient = ((function_value - function_prev) / (value - prev))
        alpha = rho1 * alpha + (1 - rho1) * gradient
        beta = rho2 * beta + (1 - rho2) * gradient ** 2
        alpha_normalized = alpha/(1 - rho1)
        beta_normalized = beta/(1 - rho2)
        rescaled_gradient = learning_rate*alpha_normalized/(np.sqrt(beta_normalized)+epsilon)
        return value, value - rescaled_gradient, alpha, beta
    except:
        gradient = 1
        alpha = rho1 * alpha + (1 - rho1) * gradient
        beta = rho2 * beta + (1 - rho2) * gradient ** 2
        alpha_normalized = alpha / (1 - rho1)
        beta_normalized = beta / (1 - rho2)
        rescaled_gradient = learning_rate*alpha_normalized/(np.sqrt(beta_normalized)+epsilon)
        return value, value-rescaled_gradient, alpha, beta

if __name__ == '__main__':
    activate = {
        'Gradient Descent':False,
        'AdaGrad':False,
        'RMSProp':False,
        'AdaDelta':False,
        'Adam':True
    }

    if activate['Gradient Descent']:
        #Gradient Descent
        p = polynomial([0,0,3]) #Initialize polynomial
        #Initialize inputs
        prev = None #No previous prediction
        func_prev = None #No previous result
        val = random.randint(-3,3) #Random initial prediction
        func_val = p.apply(val) #Result of initial prediction

        #Hyperparameters
        lr = 0.05 # Learning rate

        p.draw(-3,3,0.001, title = 'gradient descent',show=False) #Draw polynomial

        for i in range(50):
            prev, val = Gradient_Descent(func_val,val,func_prev, prev, learning_rate=lr) #New prediction using grad. desc.
            func_prev = func_val #Previous result
            func_val = p.apply(val) #New result from prediction
            plt.plot([prev,val],[func_prev, func_val],marker = "o",color = "b") #Plot change in prediction
            plt.pause(0.5)

        plt.show()

    if activate['AdaGrad']:
        # Adaptive gradient descent
        p = polynomial([0, 0, 3])
        prev = None
        func_prev = None
        val = random.randint(-3, 3)
        func_val = p.apply(val)

        #Hyperparameters
        alpha = 0.1
        eps = 1e-7
        lr = 0.5 #High learning rate as adagrad performs better in the condition

        p.draw(-3, 3, 0.001, title='AdaGrad', show=False)

        for i in range(50):
            prev, val, alpha = AdaGrad(func_val, val, function_prev=func_prev, prev=prev, alpha = alpha, epsilon = eps, learning_rate=lr)  # New prediction using AdaGrad
            func_prev = func_val  # Previous result
            func_val = p.apply(val)  # New result from prediction
            plt.plot([prev, val], [func_prev, func_val], marker="o", color="b")  # Plot change in prediction
            plt.pause(0.5)

        plt.show()

    if activate['RMSProp']:
        # Root Mean Squared Propagation
        p = polynomial([0, 0, 3])
        prev = None
        func_prev = None
        val = random.randint(-3, 3)
        func_val = p.apply(val)

        # Hyperparameters
        alpha = 0.1
        rho = 0.9
        eps = 1e-7
        lr = 0.1

        p.draw(-3, 3, 0.001, title='RMSProp', show=False)

        for i in range(50):
            prev, val, alpha = RMSProp(func_val, val, function_prev=func_prev, prev=prev, alpha=alpha, rho = rho, epsilon=eps, learning_rate=lr)  # New prediction using AdaGrad
            func_prev = func_val  # Previous result
            func_val = p.apply(val)  # New result from prediction
            plt.plot([prev, val], [func_prev, func_val], marker="o", color="b")  # Plot change in prediction
            plt.pause(0.5)

        plt.show()

    if activate['AdaDelta']:
        #Adaptive Learning Rate Method
        p = polynomial([0, 0, 3])
        prev = None
        func_prev = None
        val = random.randint(-3, 3)
        func_val = p.apply(val)

        # Hyperparameters
        alpha = 0.1
        delta = 0.5
        rho = 0.95
        eps = 1e-7

        p.draw(-3, 3, 0.001, title='AdaDelta', show=False)

        for i in range(200):
            prev, val, alpha, delta = AdaDelta(func_val, val, function_prev=func_prev, prev=prev, alpha=alpha, rho=rho, delta = delta, epsilon=eps)  # New prediction using AdaGrad
            func_prev = func_val  # Previous result
            func_val = p.apply(val)  # New result from prediction
            plt.plot([prev, val], [func_prev, func_val], marker="o", color="b")  # Plot change in prediction
            plt.pause(0.5)

        plt.show()

    if activate['Adam']:
        # Adaptive Moment Estimation
        p = polynomial([0, 0, 3])
        prev = None
        func_prev = None
        val = random.randint(-3, 3)
        func_val = p.apply(val)

        # Hyperparameters
        alpha = 0
        beta = 0
        rho1 = 0.9
        rho2 = 0.99
        eps = 1e-7
        lr = 0.05

        p.draw(-3, 3, 0.001, title='Adam', show=False)

        for i in range(200):
            prev, val, alpha, beta = Adam(func_val, val, function_prev=func_prev, prev=prev, alpha=alpha, beta = beta, rho1=rho1, rho2 = rho2, epsilon=eps, learning_rate=lr)  # New prediction using AdaGrad
            func_prev = func_val  # Previous result
            func_val = p.apply(val)  # New result from prediction
            plt.plot([prev, val], [func_prev, func_val], marker="o", color="b")  # Plot change in prediction
            plt.pause(0.5)

        plt.show()