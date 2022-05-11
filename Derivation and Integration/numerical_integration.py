from random import uniform
import os
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')


def func(x: float) -> float:
    return 1/x


def trapezoid(A: float, B: float, N: int) -> float:
    h: float = (B - A) / (N - 1)
    s: float = (func(A) + func(B)) / 2
    for i in range(1, N - 1):
        s += func(A + i*h)
    return h * s


def simpson(A: float, B: float, N: int) -> float:
    h: float = (B - A) / (N - 1)
    s: float = (func(A) + func(B)) / 3
    for i in range(1, N - 1):
        if i % 2 == 1:
            s += (4 * func(A + i*h)) / 3
        if i % 2 == 0:
            s += (2 * func(A + i*h)) / 3
    return h * s


def monte_carlo(A: float, B: float, N: int) -> float:
    # teorema della media integrale
    s: float = 0
    for _ in range(N):
        s += func(uniform(A, B))
    return (B - A) * s / N


def VonNeumann_Rejection(A: float, B: float, f,
                         N_div: int = 1000,
                         N_pts: int = 10000) -> float:
    # metodo del rifiuto
    fig, ax = plt.subplots(figsize=(18, 10))
    x: np.ndarray = np.arange(A, B + (B/N_div), B/N_div)
    y: np.ndarray = f(x)

    y_min, y_max = min(y), max(y)

    x_rand: np.ndarray = np.random.uniform(A, B, N_pts)
    y_rand: np.ndarray = np.random.uniform(y_min, y_max, N_pts)

    xi, yi, xo, yo = [], [], [], []

    ax.plot(x, y, 'c', linewidth=4)
    ax.set_xlim((A, B))
    ax.set_ylim((y_min, y_max))
    ax.set_ylabel('$f(x)$', fontsize=20)
    ax.set_xlabel('x', fontsize=20)

    area_tot: float = (B - A) * (y_max - y_min)
    i: int = 0
    for j in range(1, N_pts):
        if y_rand[j] <= f(x_rand[j]):
            xi.append(x_rand[j])
            yi.append(y_rand[j])
            i += 1
        else:
            xo.append(x_rand[j])
            yo.append(y_rand[j])
    area = area_tot * i / (N_pts - 1)
    ax.plot(xo, yo, 'bo')
    ax.plot(xi, yi, 'ro')
    fig.savefig(f'{plots}/von_neumann_rejection_integral.png')
    plt.close()
    return area


k: int = 100
a: float = -1
b: float = 1
n: int = 1000
# print("Metodo dei Trapezi:", trapezoid(a, b, n-1))
# print("Regola di Simpson:", simpson(a, b, n-1))
# print("Monte Carlo (T. Med. Int.):",
#       np.mean([monte_carlo(a, b, n) for _ in range(k)]))
# print("Monte Carlo (Acceptance Rejection):",
#       np.mean([VonNeumann_Rejection(a, b, func) for _ in range(k)]))
# print("Val. Atteso:", np.sqrt(np.pi))

NN: np.ndarray = np.array([10, 50, 250, 1250, 6250])
dd: list[float] = []

for N in NN:
    monte_carlo_seq: list[float] = []
    s: float = 0
    s_2: float = 0
    for i in range(k):
        I: float = monte_carlo(a, b, N)
        monte_carlo_seq.append(I)
        s += I
        s_2 += I**2
    dev_st: float = ((s_2 / k) - (s / k)**2)**0.5
    dd.append(dev_st)

fig, ax = plt.subplots(figsize=(18, 10))
ax.scatter(NN**(-0.5), dd)
ax.set_ylim(0)
ax.set_xlim(0)
fig.savefig(f'{plots}/integraleallavardaci.png')
