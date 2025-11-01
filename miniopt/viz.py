import matplotlib.pyplot as plt
import numpy as np

def plot_function_1d(f, x_min=-3, x_max=3, step=0.01, label="f(x)"):
    xs = np.arange(x_min, x_max, step)
    ys = f(xs)
    plt.plot(xs, ys, label=label)

def plot_path_1d(f, history, x_min=-3, x_max=3, step=0.01, title=None):
    plot_function_1d(f, x_min, x_max, step)
    xs_path = [h["x"] for h in history]
    ys_path = [h["f"] for h in history]
    plt.plot(xs_path, ys_path, marker="o", color="b")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    if title:
        plt.title(title)
    plt.legend()
    plt.show()

def plot_convergence(history, title=None):
    iters = [h["step"] for h in history]
    vals = [h["f"] for h in history]
    plt.plot(iters, vals, marker="o")
    plt.xlabel("iteration")
    plt.ylabel("f(x)")
    if title:
        plt.title(title)
    plt.show()
