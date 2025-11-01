from miniopt import Polynomial, Adam, GD, run_1d, viz

if __name__ == "__main__":
    p = Polynomial([0, 0, 3])
    dp = p.derivative()

    opt = Adam(lr=0.1)

    history = run_1d(
        f=p,
        f_prime=dp,
        optimizer=opt,
        x0=2.5,
        steps=50
    )

    viz.plot_path_1d(p, history, x_min=-3, x_max=3,
                     title=f"Adam on {repr(p)}")
    viz.plot_convergence(history, title="Adam convergence")
