from dataclasses import dataclass, field
import os
import numpy as np
from matplotlib import pyplot as plt
from ode import rk4

plt.style.use('seaborn-paper')

root = os.path.dirname(__file__)
plots = os.path.join(root, 'plots')


@dataclass()
class Harmonic_Oscillator:
    '''
    Creates an Harmonic Oscillator and solves it using Runge-Kutta Algorithm
    x_0: float -> posizione iniziale
    v_0: float -> velocità iniziale
    t_i: float -> tempo iniziale
    t_f: float -> tempo finale
    h: float -> time step
    m: float -> massa
    k: float -> costante elastica
    beta: float -> costante viscosa
    '''
    x_0: float
    v_0: float
    t_i: float
    t_f: float
    h: float
    m: float
    k: float
    beta: float
    times: list[float] = field(init=False, default_factory=list)
    positions: list[float] = field(init=False, default_factory=list)
    velocities: list[float] = field(init=False, default_factory=list)
    KE: list[float] = field(init=False, default_factory=list)
    PE: list[float] = field(init=False, default_factory=list)
    energy: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.w_0: float = np.sqrt(self.k/self.m)
        x: np.ndarray = np.zeros((2), float)  # vettore derivate
        x[0], x[1] = self.x_0, self.v_0  # condizioni iniziali
        for t in np.arange(self.t_i, self.t_f, self.h):  # runge-kutta
            x = rk4(t, self.h, x, self.evaluate)
            self.times.append(t)
            self.positions.append(x[0])
            self.velocities.append(x[1])
            self.KE.append((self.m * x[1]**2)/2)
            self.PE.append((self.k * x[0]**2)/2)
        self.KE = np.array(self.KE, float)
        self.PE = np.array(self.PE, float)
        self.energy = self.KE + self.PE

    def evaluate(self, t: float, x: np.ndarray) -> np.ndarray:
        f: np.ndarray = np.zeros((2), float)
        f[0] = x[1]
        f[1] = -self.beta*x[1] - self.w_0**2*x[0]
        return f

    def get_KE(self) -> float:
        return np.array([(self.m * v**2)/2 for v in self.velocities])

    def get_PE(self) -> float:
        return np.array([(self.k * x**2)/2 for x in self.positions])

    def plot_positions(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        ax.scatter(self.times, self.positions)
        ax.set_title('Harmonic Oscillator x(t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x(t)')
        ax.set_xlim(self.t_i, self.t_f)
        ax.grid()
        fig.savefig(f'{filepath}')
        plt.close()

    def plot_PhaseSpaceOrbit(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        ax.scatter(self.positions, self.velocities)
        ax.set_title('Phase Space Orbit')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        fig.savefig(f'{filepath}')
        plt.close()

    def plot_energy(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        ax.scatter(self.times, self.energy, label='Total Energy', c='y')
        ax.scatter(self.times, self.KE, label='Kinetic Energy', c='r')
        ax.scatter(self.times, self.PE, label='Potential Energy', c='b')
        ax.set_title('Energy')
        ax.set_xlabel('t')
        ax.set_ylabel('E(t)')
        ax.legend()
        fig.savefig(f'{filepath}')
        plt.close()


@dataclass()
class Anharmonic_Oscillator:
    '''
    Parent class without differential equation, used to create specific potentials
    x_0: float -> posizione iniziale
    v_0: float -> velocità iniziale
    t_i: float -> tempo iniziale
    t_f: float -> tempo finale
    h: float -> time step
    m: float -> massa
    k: float -> costante elastica
    '''
    x_0: float
    v_0: float
    t_i: float
    t_f: float
    h: float
    m: float
    k: float
    times: list[float] = field(init=False, default_factory=list)
    positions: list[float] = field(init=False, default_factory=list)
    velocities: list[float] = field(init=False, default_factory=list)
    KE: list[float] = field(init=False, default_factory=list)
    PE: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        x: np.ndarray = np.zeros((2), float)  # vettore derivate
        x[0], x[1] = self.x_0, self.v_0  # condizioni iniziali
        for t in np.arange(self.t_i, self.t_f, self.h):  # runge-kutta
            x = rk4(t, self.h, x, self.evaluate)
            self.times.append(t)
            self.positions.append(x[0])
            self.velocities.append(x[1])

    def evaluate(self, t: float, x: np.ndarray) -> np.ndarray:
        pass

    def get_KE(self) -> float:
        return np.array([(self.m * v**2)/2 for v in self.velocities])

    def plot_positions(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        ax.plot(self.times, self.positions)
        ax.set_title('Anharmonic Oscillator x(t)')
        ax.set_xlabel('t')
        ax.set_ylabel('x(t)')
        ax.set_xlim(self.t_i, self.t_f)
        fig.savefig(f'{filepath}')
        fig.close()


    def plot_PhaseSpaceOrbit(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        ax.plot(self.positions, self.velocities)
        ax.set_title('Phase Space Orbit')
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        fig.savefig(f'{filepath}')
        fig.close()


@dataclass()
class Anharmonic_Oscillator_Power(Anharmonic_Oscillator):
    '''
    Creates an Anharmonic Oscillator with potential: V(x)=(1/p)*k*x^p
    x_0: float -> posizione iniziale
    v_0: float -> velocità iniziale
    t_i: float -> tempo iniziale
    t_f: float -> tempo finale
    h: float -> time step
    m: float -> massa
    k: float -> costante elastica
    p: int -> esponente (must be even)
    '''
    p: int
    times: list[float] = field(init=False, default_factory=list)
    positions: list[float] = field(init=False, default_factory=list)
    velocities: list[float] = field(init=False, default_factory=list)

    def evaluate(self, t: float, x: np.ndarray) -> np.ndarray:
        f: np.ndarray = np.zeros((2), float)
        f[0] = x[1]
        f[1] = (-self.k*x[0]**(self.p-1)) / self.m
        return f

    def get_PE(self) -> float:
        return np.array([(self.k * x**self.p)/self.p for x in self.positions])

    def plot_energy(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        KE: float = self.get_KE()
        PE: float = self.get_PE()
        ax.scatter(self.times, KE + PE, label='Total Energy', c='y')
        ax.scatter(self.times, KE, label='Kinetic Energy', c='r')
        ax.scatter(self.times, PE, label='Potential Energy', c='b')
        ax.set_title('Energy')
        ax.set_xlabel('t')
        ax.set_ylabel('E(t)')
        ax.legend()
        fig.savefig(f'{filepath}')
        plt.close()


@dataclass()
class Anharmonic_Oscillator_Perturbation(Anharmonic_Oscillator):
    '''
    Creates an Anharmonic Oscillator with potential: V(x)=(1/2)*k*x^2 * (1 - 2*a*x/3)
    x_0: float -> posizione iniziale
    v_0: float -> velocità iniziale
    t_i: float -> tempo iniziale
    t_f: float -> tempo finale
    h: float -> time step
    m: float -> massa
    k: float -> costante elastica
    p: int -> esponente (must be even)
    '''
    a: float
    times: list[float] = field(init=False, default_factory=list)
    positions: list[float] = field(init=False, default_factory=list)
    velocities: list[float] = field(init=False, default_factory=list)

    def evaluate(self, t: float, x: np.ndarray) -> np.ndarray:
        f: np.ndarray = np.zeros((2), float)
        f[0] = x[1]
        f[1] = (-self.k*x[0] + 2*self.a*x[0]/3) / self.m
        return f

    def get_PE(self) -> float:
        return np.array([(self.k*x**2*(1 - 2*self.a*x/3))/2 for x in self.positions])

    def plot_energy(self, filepath: str) -> None:
        fig, ax = plt.subplots(figsize=(19.20, 10.80))
        KE: float = self.get_KE()
        PE: float = self.get_PE()
        ax.scatter(self.times, KE + PE, label='Total Energy', c='y')
        ax.scatter(self.times, KE, label='Kinetic Energy', c='r')
        ax.scatter(self.times, PE, label='Potential Energy', c='b')
        ax.set_title('Energy')
        ax.set_xlabel('t')
        ax.set_ylabel('E(t)')
        ax.legend()
        fig.savefig(f'{filepath}')
        plt.close()



def check_stability(oscillator) -> None:
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    stability: np.ndarray = - np.log10(abs((oscillator.energy[1:] - oscillator.energy[0])/oscillator.energy[0]))
    ax.plot(oscillator.times[1:], stability)
    fig.savefig(f'{plots}/stability.png')
    plt.close()



if __name__ == "__main__":
    a = Harmonic_Oscillator(x_0=0.25, v_0=1, t_i=0, t_f=1000, h=0.001, m=1, k=5, beta=0)
    a.plot_energy(f'{plots}/provaenergia.png')
    check_stability(a)
    # b = Anharmonic_Oscillator_Power(1, 0, 0, 10, 0.01, 2, 3, 4)
    # b.plot_energy(f'{plots}/provaenergia2.png')
    # c = Anharmonic_Oscillator_Perturbation(1, 0, 0, 10, 0.01, 2, 3, 0.2)
    # c.plot_energy(f'{plots}/provaenergia3.png')
