import random
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

random.seed(1)


def linear_congruential_gen(A: float = 0, B: float = 1,
                            seed: float = 1, m: int = 2**48,
                            a: int = 25214903917, c: int = 11):
    while True:
        yield A + (seed / m) * (B - A)
        seed = (a*seed + c) % m


eps: float = 0.000001
d1: float = 1
d2: float = 1
N: int = 1
nn: list[int] = []
mm: list[float] = []
ddst: list[float] = []


while d1 >= eps and d2 >= eps:
    N += 10
    # g = linear_congruential_gen()
    seq: list = []
    s: float = 0
    s_2: float = 0

    for _ in range(N):
        x = random.random()
        seq.append(x)
        s += x
        s_2 += x**2

    mean: float = s / N
    dev_st: float = ((s_2 - N*mean**2) / (N - 1))**0.5
    M: float = max(seq)
    m: float = min(seq)

    d1 = abs(mean - 0.5)
    d2 = abs(dev_st - (1/12**0.5))

    nn.append(N)
    mm.append(mean)
    ddst.append(dev_st)


fig, axes = plt.subplots(1, 2, figsize=(19.20, 10.80))
axes[0].scatter(nn, mm, c='k', label='Mean')
axes[1].scatter(nn, ddst, c='b', label='Dev St')
fig.savefig('plots/rng_valoriattesi.png')


# Plotting
# fig, ax = plt.subplots(figsize=(19.20, 10.80))

# for i in range(N):
#     ax.scatter(i, next(lcg), c='k')

# fig.savefig('plots/numbgen.png')
