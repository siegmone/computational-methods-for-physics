import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import pi, sin, cos

plt.style.use('seaborn-paper')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')


N: int = 1000
H: float = 0.01
T: float = N*H
w: float = 2*pi/T
sqrt2pi: float = (2*pi)**(1/2)
twopi: float = 2*pi


def signal(timerange: np.ndarray) -> np.ndarray:
    y: np.ndarray = np.zeros(N, float)
    for i, t in enumerate(timerange):
        y[i] = 5*sin(w*t) + 2*cos(3*w*t) + sin(5*w*t)
    return y


def DFT(sig: np.ndarray) -> np.ndarray:
    Y: np.ndarray = np.zeros(N, complex)
    W: np.ndarray = np.zeros(N, float)
    for n in range(N):
        z_sum: complex = complex(0.0, 0.0)
        for k in range(N):
            z_exp: complex = complex(0, twopi*n/N)
            z_sum += sig[k]*np.exp(-z_exp)
        Y[n] = z_sum/sqrt2pi
        W[n] = twopi*n/T
    return Y, W


TIMERANGE: np.ndarray = np.arange(0, T, H)
SIGNAL = signal(timerange=TIMERANGE)
transform, frequencies = DFT(SIGNAL)
complex_values = np.zeros(N, float)
for j, value in enumerate(transform):
    complex_values[j] = value.imag

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25.60, 10.80))

ax1.plot(TIMERANGE, SIGNAL, 'k')
ax1.plot(TIMERANGE, complex_values*1e12, 'b')

ax2.plot(frequencies, 'g')

fig.savefig(f'{plots}/DFT_signals')
