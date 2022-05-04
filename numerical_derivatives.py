from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')


def forward_difference(f, x, step_size):
    return (f(x + step_size) - f(x))/step_size


def central_difference(f, x, step_size):
    num = f(x + (step_size/2)) - f(x - (step_size/2))
    return num / step_size


def run_derivative(x, func):
    stepsize = 4
    steps = []
    forw = []
    centr = []
    f_errors = []
    c_errors = []
    ref = func(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.20, 10.80))
    while stepsize > 1e-10:
        f = forward_difference(np.exp, x, stepsize)
        c = central_difference(np.exp, x, stepsize)
        f_errors.append(abs(f - ref))
        c_errors.append(abs(c - ref))
        forw.append(f)
        centr.append(c)
        steps.append(stepsize)
        stepsize /= 2
    ax1.plot(steps, forw, label="Forward Difference")
    ax1.plot(steps, centr, label="Central Difference")
    ax1.set_xlim(0)
    ax1.legend()
    ax2.plot(steps, f_errors, label="Forward Difference")
    ax2.plot(steps, c_errors, label="Central Difference")
    ax2.set_ylim(0)
    ax2.set_xlim(0)
    ax2.legend()
    fig.savefig(f"plots/derivatives_cos_{x}.png")


run_derivative(0.1, np.cos)
run_derivative(1, np.cos)
run_derivative(100, np.cos)
