import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from ode import rk4

plt.style.use('seaborn-paper')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')

G = 1
M = 1


def force_field(x: float, y: float) -> np.ndarray:
    force: np.ndarray = np.zeros((2), float)
    force[0] = -G*M*x/((x**2+y**2)**(3/2))
    force[1] = -G*M*y/((x**2+y**2)**(3/2))
    return force


def orbit(x_0: float, vx_0: float, y_0: float, vy_0: float, mass: float, T: float, h: float) -> np.ndarray:

    def evaluate(t: float, Y: np.ndarray) -> np.ndarray:
        f: np.ndarray = np.zeros((4), float)
        F = force_field(x=Y[0], y=Y[2])
        f[0] = Y[1]
        f[1] = F[0]/mass
        f[2] = Y[3]
        f[3] = F[1]/mass
        return f

    def generate_orbit(x_0: float, vx_0: float, y_0: float, vy_0: float, h: float) -> None:
        Y: np.ndarray = np.zeros((4), float)  # vettore derivate
        Y[0], Y[1], Y[2], Y[3] = x_0, vx_0, y_0, vy_0  # condizioni iniziali
        t: float = 0
        while True:
            Y = rk4(t, h, Y, evaluate)
            yield Y
            t += h

    timerange: np.ndarray = np.arange(0, T, h)
    orbit_values: np.ndarray = np.zeros(timerange.shape + (4,))
    generator = generate_orbit(x_0=x_0, vx_0=vx_0, y_0=y_0, vy_0=vy_0, h=h)

    for i, t in enumerate(timerange):
        orbit_values[i] = next(generator)
    return orbit_values


H: float = 0.2
X_0: float = 0.2
Y_0: float = 0.
T: float = 2000

O = orbit(x_0=X_0, vx_0=0., y_0=Y_0, vy_0=1.63, mass=1, T=T, h=H)
print(O)

time = np.arange(0, T, H)
X = O[:, 0]
Y = O[:, 2]

fig0, ax0 = plt.subplots(figsize=(19.20, 19.20))
ax0.plot(X, Y)
ax0.plot(0, 0, 'ko')
fig0.savefig(f'{plots}/planet')
print('Picture Saved')

# ---------------------------------------------------------------------------------
# Creating an animation for the pendulum
# Define the meta data for the movie
FFMpegWriter = matplotlib.animation.writers['ffmpeg']
metadata = dict(title='Animation', artist='matplotlib', comment='animation')
writer = FFMpegWriter(fps=60, metadata=metadata)

# Initialize the movie
fig, ax = plt.subplots(figsize=(19.20, 19.20))

max_radius = max(X_0, Y_0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-max_radius*3, max_radius*3)
ax.set_ylim(-max_radius*3, max_radius*3)

# plot the trajectory
line, = ax.plot(X, Y, lw=0.2, alpha=0.5)

# create the point
ax.plot(0, 0, 'ro', markersize=20)
point, = ax.plot([], [], 'ko', markersize=10)

# Update the frames for the movie
with writer.saving(fig, f"{plots}/Movie.mp4", 100):
    for i in range(len(time)-1):
        print(i)
        x0 = X[i]
        y0 = Y[i]
        point.set_data(x0, y0)
        ax.set_title(f'$T=${time[i]:.2f}')
        writer.grab_frame()
print('Movie Saved')
# ---------------------------------------------------------------------------------
