import numpy as np


def rk4(t: float, h: float, y: np.ndarray, f) -> np.ndarray:  # 4-th order Runge-Kutta
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2*(k2+k3) + k4) / 6

