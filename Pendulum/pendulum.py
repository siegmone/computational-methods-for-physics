import os
from matplotlib import pyplot as plt
import matplotlib.animation as manimation
import numpy as np
from ode import rk4

plt.style.use('seaborn-paper')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')


def simple_pendulum(th_0: float, v_0: float, w_0:float, T: float, h: float) -> np.ndarray:
    def evaluate(t: float, th: np.ndarray) -> np.ndarray:
        f: np.ndarray = np.zeros((2), float)
        f[0] = th[1]
        f[1] = -w_0**2 * np.sin(th[0])
        return f
    def generate_pendulum(th_0: float, v_0: float, w_0:float, h: float) -> None:
        th: np.ndarray = np.zeros((2), float)  # vettore derivate
        th[0], th[1] = th_0, v_0  # condizioni iniziali
        t: float = 0
        while True:
            th = rk4(t, h, th, evaluate)
            yield th
            t += h
    vals: np.ndarray = np.arange(0, T, h)
    TH: np.ndarray = np.zeros(vals.shape + (2,))
    generator = generate_pendulum(th_0, v_0, w_0, h)
    for i, val in enumerate(vals):
        TH[i] = next(generator)
    return TH


def get_pendulum_coordinates(pendulum: np.ndarray, length: float) -> np.ndarray:
    theta: np.ndarray = pendulum[:, 0]
    
    x: np.ndarray = length*np.sin(theta)
    y: np.ndarray = -length*np.cos(theta)
    return x, y


def get_pendulum_energy(pendulum: np.ndarray, length: float) -> np.ndarray:
    x, y = get_pendulum_coordinates(pendulum=pendulum, length=length)
    v = pendulum[:, 1]*length
    kinetic = v**2/2
    potential = 9.81 * y
    return kinetic + potential


LENGTH: float = 1
W_0: float = (9.81/LENGTH)**0.5
T: float = 15
H: float = 0.01
time = np.arange(0, T, H)
P = simple_pendulum(th_0=np.pi/8, v_0=0, w_0=W_0, T=T, h=H)
X, Y = get_pendulum_coordinates(P, LENGTH)
E = get_pendulum_energy(P, LENGTH)

# Define the meta data for the movie
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Pendulum Animation', artist='matplotlib',
                comment='Pendulum')
writer = FFMpegWriter(fps=60, metadata=metadata)

# Initialize the movie
fig, ax = plt.subplots(figsize=(19.20, 19.20))

# plot the sine wave line
line, = ax.plot(X, Y, 'r', lw=0.2)
ax.plot(0, 0, markersize=10)

red_circle, = ax.plot([], [], 'ko', markersize=40)
# rope, = ax.plot([], [], 'k.', markersize=20)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-LENGTH*1.1, LENGTH*1.1)
ax.set_ylim(-LENGTH*1.1, LENGTH*1.1)

# Update the frames for the movie
with writer.saving(fig, f"{plots}/Pendulum_Movie.mp4", 100):
    for i in range(len(time)-1):
        x0 = X[i]
        y0 = Y[i]
        red_circle.set_data(x0, y0)
        rope, = ax.plot([0, x0], [0, y0], color='b')
        writer.grab_frame()
        rope.remove()
