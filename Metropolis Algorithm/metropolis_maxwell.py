from random import random, uniform
import os
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-paper')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')

B = 3.
T = 1.2


def metropolis(X_0, N, delta, target):
    markov_chain = np.zeros(N)
    markov_chain[0] = X_0
    for t in range(N - 1):
        X_t = markov_chain[t]
        Y = uniform(X_t - delta, X_t + delta)  # uniform proposal function
        if Y > B:
            w = target(Y)/target(X_t)
            alpha = min(w, 1)
            U = random()
            if U <= alpha:
                markov_chain[t + 1] = Y
            else:
                markov_chain[t + 1] = X_t
        elif Y <= B:
            markov_chain[t + 1] = X_t
    return markov_chain


def maxwell(E):
    return np.sqrt(E-B) * np.exp(-E/T)


fig1, ax1 = plt.subplots(figsize=(19.20, 10.80))
EE = np.arange(B, 30, 0.005)
ax1.scatter(EE, maxwell(EE), marker='.', c='k')
ax1.set_ylim(0)
fig1.savefig(f'{plots}/maxwell.png')

fig2, ax2 = plt.subplots(figsize=(19.20, 10.80))
gg = metropolis(X_0=30, N=10000, delta=1, target=maxwell)
bins = np.arange(int(B), int(max(gg)), 0.1)
ax2.hist(gg, edgecolor='k', bins=bins, density=True)
fig2.savefig(f'{plots}/metropolis.png')

fig3, ax3 = plt.subplots(figsize=(19.20, 10.80))
ax3.plot(gg)
fig3.savefig(f'{plots}/energies.png')


def f(Z):
    if Z < 45:
        return 0.28*Z**(2/3)
    if Z >= 45:
        return Z**(1/3)


def stopping_power(E, Z_1, A_1, Z_2, A_2):
    m = 6.6e-27
    e = 1.6e-13 * E
    v = np.sqrt(2*e/m) * 10
    return 1.327 * f(Z_2) * (4.7622*f(Z_1)**(5/3)+f(Z_1)) * v / A_2


fig4, ax4 = plt.subplots(figsize=(19.20, 10.80))
sp = stopping_power(gg, 2, 4, 13, 26)
SP = gg - sp
L = []
print(gg)
print(sp)
for val in SP:
    if val > 0:
        L.append(val)
bins = np.linspace(int(min(L)), int(max(L)), 76)
ax4.hist(L, edgecolor='k', density=True, bins=bins)
fig4.savefig(f'{plots}/stoppingpower.png')
