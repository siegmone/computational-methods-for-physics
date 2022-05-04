import random
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')


def random_walk() -> float:
    x, y = [50, 0]
    n: int = 0
    while 0 < x < 100 and y < 100:
        n += 1
        p: float = random.uniform(0, 3)
        if p <= 1:  # sinistra
            x -= 1
        if p >= 2:  # destra
            x += 1
        if 1 < p <= 2:  # dritto
            y += 1
    return n


k: int = 1000
drunks: list[float] = []
for _ in range(k):
    drunks.append(random_walk())
print(np.mean(drunks))
fig, ax = plt.subplots(figsize=(19.20, 10.80))
ax.hist(drunks, edgecolor='k')
fig.savefig('plots/drunken_hist.png')