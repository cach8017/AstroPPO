import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
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
        self.tspan = 100 * 60.0  # seconds
        self.dt = 60.0  # seconds

        # number of steps
        self.n_steps = int(np.ceil(self.tspan / self.dt))

        # initialize arrays to store the state vector
        self.ys = np.zeros((self.n_steps, 6))
        self.ts = np.zeros((self.n_steps, 1))

        # initial conditions
        y0 = r0 + v0
        self.ys[0] = np.array(y0)
        self.step = 1

    def diff_q(t,y,mu):
        # unpack the state vector
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        #norm of the radius vector
        r_norm = np.linalg.norm(r)
        # compute the 2 body acceleration
        ax, ay, az = -mu * r / r_norm**3
        # return the derivative of the state
        return [vx, vy, vz, ax, ay, az]
    
    def orbit_propogation(self):
        # initialize the solver
        self.solver = ode(self.diff_q)
        self.solver.set_integrator('rk4')
        self.solver.set_initial_value(self.ys[0], 0.0)
        self.solver.set_f_params(self.earth_mu) # set the mu parameter for the diff_q function

        # propogate the orbit
        while self.solver.successful() and self.step < self.n_steps:
            self.solver.integrate(self.solver.t + self.dt)
            self.ts[self.step] = self.solver.t
            self.ys[self.step] = self.solver.y
            self.step += 1
        rs = self.ys[:, 0:3] # position vector
        # plot(rs) below in plot function

    def plot(self):
        rs = self.ys[:, 0:3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(rs[:, 0], rs[:, 1], rs[:, 2])
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        plt.show()

    




