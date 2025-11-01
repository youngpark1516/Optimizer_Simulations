import numpy as np

class Polynomial:
    def __init__(self, coefficients=None, degree=None, coef_low=-10, coef_high=10):
        if coefficients is not None:
            self.coefficients = np.array(coefficients, dtype=float)
        elif degree is not None:
            self.coefficients = np.random.randint(coef_low, coef_high, degree + 1).astype(float)
        else:
            raise TypeError("Polynomial needs coefficients or degree")

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coefficients):
            if abs(c) < 1e-12:
                continue
            if i == 0:
                terms.append(f"{c:.2f}")
            elif i == 1:
                terms.append(f"{c:.2f}x")
            else:
                terms.append(f"{c:.2f}x^{i}")
        return " + ".join(terms) if terms else "0"

    def __call__(self, x):
        x = np.asarray(x)
        powers = np.arange(len(self.coefficients))
        x_powered = x[..., None] ** powers   # broadcast
        return (x_powered * self.coefficients).sum(axis=-1)

    def derivative(self):
        if len(self.coefficients) == 1:
            return Polynomial([0.0])
        deriv_coefs = [i * self.coefficients[i] for i in range(1, len(self.coefficients))]
        return Polynomial(deriv_coefs)

    def domain(self, start, end, step=0.01):
        xs = np.arange(start, end, step)
        ys = self(xs)
        return xs, ys
