import os
from matplotlib import pyplot as plt
import numpy as np
from metropolis import metropolis


plt.style.use('seaborn-darkgrid')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')

np.seterr('raise')


def potential(x):
    return -0.1*((x**2)+0.3)*np.exp(-(x-0.1)**2)


def boltzmann(x, y):
    T, Z = y[0], y[1]
    return np.exp(-potential(x)/(k_b*T))/Z


k_b = 1
a, b = -3, 3
bounds = [a, b]
T = 0.01
domain = np.arange(a, b, 0.01, dtype=float)
Z = np.sum(np.exp(-potential(domain)/(k_b*T)))
pot = potential(x=domain)
probability = boltzmann(x=domain, y=[T, Z])


fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.20, 10.80))

ax1.scatter(domain, pot, c='b', marker='.')
ax1.set_title('Potential')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$V(x)$')

ax2.scatter(domain, probability, c='k', marker='.')
ax2.set_title('Probability')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$p(x)$')

ax3.scatter(pot, probability, c='r', marker='.')
ax3.set_title('Probability/Potential')
ax3.set_xlabel('$V(x)$')
ax3.set_ylabel('$p(x)$')

fig1.savefig(f'{plots}/boltzmann')


positions = metropolis(X_0=0, N=10000, delta=0.5,
                       target=boltzmann, y=[T, Z], boundaries=bounds)


fig2, ax = plt.subplots(figsize=(19.20, 10.80))
bins = np.arange(a, b, 0.05)
ax.hist(positions, edgecolor='w', density=True,
        bins=bins, facecolor='k', alpha=0.8)
ax.plot(bins, boltzmann(bins, y=[T, Z])*100, color='b', linestyle='dashed', lw=4)
fig2.savefig(f'{plots}/boltzmann_hist')


temps = np.linspace(0.001, 0.05, 500)
xx = np.zeros(len(temps))
e = np.zeros(len(temps))

for i, temp in enumerate(temps):
    Z = np.sum(np.exp(-potential(domain)/(k_b*temp)))
    X = metropolis(X_0=0, N=10000, delta=0.5,
                   target=boltzmann, y=[temp, Z], boundaries=bounds)
    xx[i] = np.mean(X)
    E = potential(X)
    e[i] = np.mean(E)

fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.20, 10.80))
ax1.scatter(temps, xx, c='r', marker='o')
ax1.set_title('Posizioni Medie')
ax1.set_xlabel('T')
ax1.set_ylabel('X')

ax2.scatter(temps, e, c='r', marker='o')
ax2.set_title('Energie Medie')
ax2.set_xlabel('T')
ax2.set_ylabel('E')

fig3.savefig(f'{plots}/medie_boltzmann')
