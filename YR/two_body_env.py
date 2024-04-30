import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D


class TwoBodyEnv:
    # https://www.youtube.com/watch?v=7JY44m6eemo

    def __init__(self):
        self.earth_radius = 6371.0  # km
        self.earth_mu = 398600.0  # km^3/s^2

        # initial conditions for orbit parameters
        r_mag = self.earth_radius + 500.0  # km
        v_mag = np.sqrt(self.earth_mu / r_mag)  # km/s

        # initial position and velocity vectors
        r0 = np.array([r_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_mag, 0.0])

        # time span
        # self.tspan = 100 * 60.0  # seconds
        self.dt = 60.0  # seconds

        # number of steps
        # self.n_steps = int(np.ceil(self.tspan / self.dt))

        # initialize arrays to store the state vector
        self.ys = np.zeros((1, 6))
        # self.ts = np.zeros((1, 1))

        # initial conditions
        x0 = np.concatenate((r0, v0))
        self.x = x0
        # self.ys[0] = np.array(x0)
        # self.step = 1

    def diff_q(self, y):
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        r_norm = np.linalg.norm(r)
        ax, ay, az = -self.earth_mu * r / r_norm ** 3
        # return the derivative of the state
        return [vx, vy, vz, ax, ay, az]

    def orbit_propogation(self, duration):
        # initialize the solver
        sol = solve_ivp(self.diff_q, (0, duration), self.x)
        self.x = sol.y[:, -1]

    def get_energy(self, state=False):
        if state is False: state = self.x
        PE = 9.81 * np.linalg.norm(state[0:3])
        KE = 0.5 * np.linalg.norm(state[3:6])
        return PE + KE

    def plot(self):
        rs = self.ys[:, 0:3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(rs[:, 0], rs[:, 1], rs[:, 2])
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        plt.show()
