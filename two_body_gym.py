import numpy as np
import plotly.graph_objects as go
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from astropy import units as u
from gym import Env, spaces

class SpacecraftEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-.1, high=.1, shape=(3,), dtype=np.float32)  # Define thrust direction and magnitude
        # 3 actions, each in the range [-1, 1], pith, yaw and magnitude
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # Position (rx, ry, rz), velocity(vx,vy,vz), mass(m)

        # Initialize the 3D plot for the orbit
        self.fig = go.FigureWidget()
        self.fig.update_layout(scene=dict(xaxis_title='X (km)',
                                          yaxis_title='Y (km)',
                                          zaxis_title='Z (km)'),
                               title="Spacecraft Trajectory")

        # Set initial conditions
        self.state = self.reset()

    def step(self, action):
        # Apply the action to simulate the thrust maneuver
        thrust_direction = action[:2]  # Assume action defines direction in some coordinate frame
        thrust_magnitude = action[2]  # Last component is the magnitude of the thrust
        delta_v = np.array([*thrust_direction, 0]) * thrust_magnitude * 100  # Scale thrust for visibility

        # Perform the maneuver
        maneuver = Maneuver.impulse(delta_v * u.km / u.s)
        orbit = Orbit.from_vectors(Earth, self.state[:3] * u.km, self.state[3:6] * u.km / u.s)
        new_orbit = orbit.apply_maneuver(maneuver)

        # Update the state
        self.state = np.concatenate([new_orbit.r.to(u.km).value, new_orbit.v.to(u.km / u.s).value])

        # Plot the new trajectory segment
        self.add_trajectory(new_orbit)

        # Calculate reward and check completion
        reward = -np.abs(delta_v).sum()  # Penalize large maneuvers
        done = False  # Add appropriate conditions for task completion

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state to initial conditions
        initial_orbit = Orbit.circular(Earth, alt=300 * u.km)
        self.state = np.concatenate([initial_orbit.r.to(u.km).value, initial_orbit.v.to(u.km / u.s).value])
        self.add_trajectory(initial_orbit)
        return self.state

    def add_trajectory(self, orbit):
        # Extract the orbit data for plotting
        r = orbit.sample(values=100)
        x, y, z = r.x.to(u.km).value, r.y.to(u.km).value, r.z.to(u.km).value

        # Add the trajectory segment to the plot
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines'))

    def render(self, mode='human'):
        # Display the plot
        self.fig.show()

# Example usage
env = SpacecraftEnv()
for _ in range(10):
    state, reward, done, _ = env.step(np.random.random(3)/5 - 0.1)  # Example random action
env.render()


