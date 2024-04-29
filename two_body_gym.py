import numpy as np
import plotly.graph_objects as go
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
# from poliastro.maneuver import Maneuver
from scipy.integrate import solve_ivp
from astropy import units as u
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
        # self.action_space = spaces.Box(low=-.1, high=.1, shape=(3,), dtype=np.float32)  # Define thrust direction and magnitude
        # 3 actions, each in the range [-1, 1], pith, yaw and magnitude
        # self.action_space = spaces.Tuple(spaces.MultiBinary(1),spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32))   # Define thrust direction and magnitude
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
        self.state = self.reset()

    def step(self, action):
        # Apply the action to simulate the thrust maneuver
        #thrust_direction = action[2]  # Assume action defines direction in some coordinate frame (not in use)
        # thrust_magnitude = action[2]  # Last component is the magnitude of the thrust
        #delta_v = np.array([*thrust_direction, 0]) * thrust_magnitude * 100  # Scale thrust for visibility

        thrust_magnitude = 1 if action[1] == 1 else -1 # Select positive or negative deltaV burn
        # if action[0] == 1: thrust_magnitude = -thrust_magnitude # Correct for waiting condition
        unit_v = (self.state[3:6] / np.linalg.norm(self.state[3:6]) ) * thrust_magnitude * 0.25 # Apply thrust in the direction of the current velocity vector
        # print(f"deltaV: {unit_v}")

        # Perform the maneuver
        # orbit = Orbit.from_vectors(Earth, self.state[:3] * u.km, self.state[3:6] * u.km / u.s)
        a = self.get_semimajor_axis()
        period = 2 * np.pi * np.sqrt(a**3 / self.earth_mu)
        # HalfPeriod = orbit.period.to(u.s)/2 
        # dv = unit_v << (u.km / u.s)
        # maneuver = Maneuver.impulse(dv)
        if action[0] == 1:
            self.orbit_propogation(period / 2) # Wait for half orbit

        # new_orbit = orbit.apply_maneuver(maneuver) # Get orbit after maneuver
        self.state[3:6] += unit_v
        # self.state = np.concatenate([new_orbit.r.to(u.km).value, new_orbit.v.to(u.km / u.s).value]) # Update the state

        # self.add_trajectory(new_orbit) # Plot the new trajectory segment

        # Calculate reward and check completion
        reward = -np.abs(thrust_magnitude).sum()  # Penalize large maneuvers
        done = self.compare_orbits()  # Terminate if desired orbit is reached
        if done: reward += 1000 # Add reward for reaching desired orbit
        print(f"State: {self.state}")

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.initial_orbit
        print(f"Initial state: {self.state}")
        # self.add_trajectory(initial_orbit)
        return self.state
    
    def get_energy(self, state=False):
        state = self.state if state is False else state
        E = 0.5 * np.linalg.norm(self.state[3:6])**2 - self.earth_mu/np.linalg.norm(self.state[0:3])
        return E
    
    def get_semimajor_axis(self, state=False):
        a = -self.earth_mu / (2 * self.get_energy(state))
        return a

    def add_trajectory(self, state=False):
        state = self.state if state is False else state
        # Sample points along orbit for plotting
        orbit = Orbit.from_vectors(Earth, self.state[:3] * u.km, self.state[3:6] * u.km / u.s)
        r = orbit.sample(values=100)
        x, y, z = r.x.to(u.km).value, r.y.to(u.km).value, r.z.to(u.km).value

        # Add the trajectory segment to the plot
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines'))

    def render(self, mode='human'):
        # Display the plot
        self.fig.show()

    def diff_q(self,t, y):
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        r_norm = np.linalg.norm(r)
        ax, ay, az = -self.earth_mu * r / r_norm ** 3
        # return the derivative of the state
        return [vx, vy, vz, ax, ay, az]

    def orbit_propogation(self, duration):
        # initialize the solver
        sol = solve_ivp(self.diff_q, (0, duration), self.state)
        self.state = sol.y[:, -1]

    def compare_orbits(self):
        # percent_diff = abs((new_orbit.a.to(u.km).value - self.final_orbit.a.to(u.km).value) / self.final_orbit.a.to(u.km).value) # Using difference in semi-major axes
        percent_diff = abs((self.get_energy(self.final_orbit) - self.get_energy()) / self.get_energy(self.final_orbit))
        # print(f"Percent difference: {int(percent_diff*100)}%")
        return percent_diff < 0.1


# Example usage
env = SpacecraftEnv()
a = [[0,1], [1,1], [1,1], [0,1]]
for i in range(20):
    # a = env.action_space.sample()
    # print(a)
    a = [1, 1] # always wait and positive deltaV
    state, reward, done, _ = env.step(a)
    # if done: break
# env.render()
