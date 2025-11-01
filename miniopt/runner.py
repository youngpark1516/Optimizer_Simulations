def run_1d(f, f_prime, optimizer, x0, steps=100):
    """
    Run 1D optimization.
    :param f: callable, f(x)
    :param f_prime: callable, f'(x)
    :param optimizer: object with .step(x, grad)
    :param x0: float, initial point
    :param steps: int
    :return: list of dicts with step, x, f
    """
    history = []
    x = float(x0)
    for t in range(steps):
        fx = f(x)
        gx = f_prime(x)
        history.append({"step": t, "x": x, "f": fx, "grad": gx})
        x = optimizer.step(x, gx)
    # add final
    history.append({"step": steps, "x": x, "f": f(x), "grad": f_prime(x)})
    return history
