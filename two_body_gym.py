import numpy as np
import plotly.graph_objects as go
# from poliastro.bodies import Earth
# from poliastro.twobody import Orbit
# from poliastro.maneuver import Maneuver
from scipy.integrate import solve_ivp
# from astropy import units as u
from gym import Env, spaces

class SpacecraftEnv(Env):
    def __init__(self):
        super().__init__()
        # Reset the state to initial conditions
        self.earth_radius = 6371.0  # km
        self.earth_mu = 398600.0  # km^3/s^2

        # initial conditions for orbit parameters
        r_mag = self.earth_radius + 300.0  # km
        r_mag_final = self.earth_radius + 35786.0  # km
        v_mag = np.sqrt(self.earth_mu / r_mag) # km/s
        v_mag_final = np.sqrt(self.earth_mu / r_mag_final) # km/s

        # initial position and velocity vectors
        r0 = np.array([r_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_mag, 0.0])
        rf = np.array([r_mag_final, 0.0, 0.0])
        vf = np.array([0.0, v_mag_final, 0.0])
        self.initial_orbit = np.concatenate([r0, v0])
        self.final_orbit = np.concatenate([rf, vf])

        self.action_space = spaces.MultiBinary(2)   # Define thrust direction and magnitude
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Position (rx, ry, rz), velocity(vx,vy,vz), mass(m)

        # Initialize the 3D plot for the orbit
        self.fig = go.FigureWidget()
        self.fig.update_layout(scene=dict(xaxis_title='X (km)',
                                          yaxis_title='Y (km)',
                                          zaxis_title='Z (km)'),
                                          title="Spacecraft Trajectory")

        self.fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.5),
            name='Earth'
        ))

        # Set initial conditions
        self.x_hist = np.empty((6, 0))
        self.state = self.reset()
        self.add_trajectory(self.final_orbit)

    def step(self, action):
        # Apply the action to simulate the thrust maneuver
        thrust_magnitude = 1 if action[1] == 1 else -1 # Select positive or negative deltaV burn

        a = self.get_semimajor_axis()
        period = 2 * np.pi * np.sqrt(a**3 / self.earth_mu)

        if action[0] == 1: # wait 1.5 * T
            self.orbit_propogation(1.5 * period) # Wait for half orbit
        unit_v = (self.state[3:6] / np.linalg.norm(self.state[3:6]) ) * thrust_magnitude * 0.5 # Apply thrust in the direction of the current velocity vector

        self.state[3:6] += unit_v
        # print(f"state: {self.state}")

        # Calculate reward and check completion
        reward = -np.abs(thrust_magnitude).sum()  # Penalize large maneuvers
        done = self.compare_orbits()  # Terminate if desired orbit is reached
        if done: 
            reward += 1000 
            self.add_trajectory()
        return self.state, reward, done, {}

    def reset(self):
        self.state = self.initial_orbit
        print(f"Initial state: {self.state}")
        # self.add_trajectory(initial_orbit)
        return self.state
    
    def get_energy(self, state=False):
        state = self.state if state is False else state
        E = 0.5 * np.linalg.norm(state[3:6])**2 - self.earth_mu/np.linalg.norm(state[0:3])
        return E
    
    def get_semimajor_axis(self, state=False):
        a = -self.earth_mu / (2 * self.get_energy(state))
        return a

    def add_trajectory(self, state=False):
        # Plot one full orbit
        state = self.state if state is False else state
        a = self.get_semimajor_axis(state)
        period = 2 * np.pi * np.sqrt(a**3 / self.earth_mu)
        trajectory = self.orbit_propogation(period, state) # Wait for half orbit

        x = trajectory[0, :]
        y = trajectory[1, :]
        z = trajectory[2, :]

        # Add the trajectory segment to the plot
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='red')))

    def render(self, mode='human'):
        # Plot state history
        x = self.x_hist[0, :]
        y = self.x_hist[1, :]
        z = self.x_hist[2, :]

        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines'))
        self.fig.show()

    def diff_q(self,t, y):
        # ODEs
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        r_norm = np.linalg.norm(r)
        ax, ay, az = -self.earth_mu * r / r_norm ** 3
        # return the derivative of the state
        return [vx, vy, vz, ax, ay, az]

    def orbit_propogation(self, duration, state=False, add=True):
        state = self.state if state is False else state
        # initialize the solver
        sol = solve_ivp(self.diff_q, (0, duration), state, max_step=50)
        if state is self.state:
            self.state = sol.y[:, -1]
            self.x_hist = np.concatenate((self.x_hist, sol.y), axis=1)
        return sol.y

    def compare_orbits(self):
        # compare current and goal orbits 
        percent_diff = abs((self.get_energy(self.final_orbit) - self.get_energy()) / self.get_energy(self.final_orbit))
        print(f"Percent difference: {int(percent_diff*100)}%")
        return percent_diff < 0.1


# Example usage
if __name__ == "__main__":
    env = SpacecraftEnv()
    a = [[0,1], [1,1], [1,1], [0,1]]
    for i in range(10):
        a = [1, 1] # always wait and positive deltaV
        state, reward, done, _ = env.step(a)
        if done: break
    print(env.get_energy())
    env.render()

