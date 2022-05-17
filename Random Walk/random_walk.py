import random
import os
from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')


def random_walk() -> float:
    x, y = [50, 0]
    x_data, y_data = [x], [y]
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
        x_data.append(x)
        y_data.append(y)
    return n, x_data, y_data
