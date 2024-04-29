import gymnasium as gym
from gymnasium import spaces
import numpy as np
import plotly.graph_objects as go
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from astropy import units as u
from astropy.units import Quantity


class BaseSpacecraftEnv(gym.Env):
    def __init__(self, env_config=None):
        super().__init__()
        # self.action_space = spaces.Box(low=-.1, high=.1, shape=(3,), dtype=np.float32)  # Define thrust direction and magnitude
        # 3 actions, each in the range [-1, 1], pith, yaw and magnitude
        # self.action_space = spaces.Tuple(spaces.MultiBinary(1),spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32))   # Define thrust direction and magnitude
        self.action_space = spaces.MultiBinary(2)   # Define thrust direction and magnitude

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # Position (rx, ry, rz), velocity(vx,vy,vz), mass(m)

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
        self.final_orbit = Orbit.circular(Earth, alt=35786 * u.km)
        r = self.final_orbit.sample(values=100)
        x, y, z = r.x.to(u.km).value, r.y.to(u.km).value, r.z.to(u.km).value
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name='Goal Orbit', line=dict(width=5, dash='dash')))
        self.state = self.reset()

    def step(self, action):
        # Apply the action to simulate the thrust maneuver
        #thrust_direction = action[2]  # Assume action defines direction in some coordinate frame (not in use)
        # thrust_magnitude = action[2]  # Last component is the magnitude of the thrust
        #delta_v = np.array([*thrust_direction, 0]) * thrust_magnitude * 100  # Scale thrust for visibility

        thrust_magnitude = 1 if action[1] == 1 else -1 # Select positive or negative deltaV burn
        if action[0] == 1: thrust_magnitude = -thrust_magnitude # Correct for waiting condition
        unit_v = (self.state[3:6] / np.linalg.norm(self.state[3:6]) ) * thrust_magnitude * 0.25 # Apply thrust in the direction of the current velocity vector
        # print(f"deltaV: {unit_v}")

        position = Quantity(self.state[:3], u.km)
        velocity = Quantity(self.state[3:6], u.km / u.s)
        orbit = Orbit.from_vectors(Earth, position, velocity)
        HalfPeriod = orbit.period.to(u.s)/2 
        dv = unit_v << (u.km / u.s)
        maneuver = Maneuver.impulse(dv)
        if action[0] == 1: orbit = orbit.propagate(HalfPeriod) # Wait for half orbit

        new_orbit = orbit.apply_maneuver(maneuver) # Get orbit after maneuver

        self.state = np.concatenate([new_orbit.r.to(u.km).value, new_orbit.v.to(u.km / u.s).value]) # Update the state

        self.add_trajectory(new_orbit) # Plot the new trajectory segment

        # Calculate reward and check completion
        reward = -np.abs(thrust_magnitude).sum()  # Penalize large maneuvers
        done = self.compare_orbits(new_orbit)  # Terminate if desired orbit is reached
        if done: reward += 1000 # Add reward for reaching desired orbit
        truncated = False  # This should be True if there is a time limit and it's reached
        info = {}
        return self.state, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        initial_orbit = Orbit.circular(Earth, alt=300 * u.km)
        # Assuming concatenate combines position and velocity correctly into a flat array
        self.state = np.concatenate([initial_orbit.r.to(u.km).value, initial_orbit.v.to(u.km / u.s).value])

        # Debugging: Check the shape and type of self.state
        print("State shape:", self.state.shape)
        print("State type:", type(self.state))

        # info = {}  # This should be a simple dictionary
        return self.state#, info


    
    def add_trajectory(self, orbit):
        # Sample points along orbit for plotting
        r = orbit.sample(values=100)
        x, y, z = r.x.to(u.km).value, r.y.to(u.km).value, r.z.to(u.km).value

        # Add the trajectory segment to the plot
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines'))

    def render(self, mode='human'):
        # Display the plot
        self.fig.show()

    def compare_orbits(self, new_orbit):
        percent_diff = abs((new_orbit.a.to(u.km).value - self.final_orbit.a.to(u.km).value) / self.final_orbit.a.to(u.km).value) # Using difference in semi-major axes
        print(f"Percent difference: {int(percent_diff*100)}%")
        return percent_diff < 0.1


# Example usage
env = BaseSpacecraftEnv()
a = [[0,1], [1,1], [1,1], [0,1]]
for i in range(20):
    # a = env.action_space.sample()
    # print(a)
    a = [1, 1] # always wait and positive deltaV
    state, reward, done, truncated, _ = env.step(a)
    print(f"Step {i}: State={state}, Reward={reward}, Done={done}, Truncated={truncated}")
    if done: 
        print("Episode finished!")
        break
env.render()





