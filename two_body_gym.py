import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from gym import Env, spaces
from scipy.integrate import solve_ivp


class SpacecraftEnv(Env):
    def __init__(self):
        super().__init__()
        # Reset the state to initial conditions
        self.earth_radius = 6371.0  # km
        self.earth_mu = 398600.0  # km^3/s^2

        # initial conditions for orbit parameters
        r_mag = self.earth_radius + 300.0  # km
        r_mag_final = self.earth_radius + 35786.0  # km
        v_mag = np.sqrt(self.earth_mu / r_mag)  # km/s
        v_mag_final = np.sqrt(self.earth_mu / r_mag_final)  # km/s
        self.initial_orbit_radius = r_mag
        self.final_orbit_radius = r_mag_final

        # initial position and velocity vectors
        r0 = np.array([r_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_mag, 0.0])
        rf = np.array([r_mag_final, 0.0, 0.0])
        vf = np.array([0.0, v_mag_final, 0.0])
        self.initial_orbit = np.concatenate([r0, v0])
        self.final_orbit = np.concatenate([rf, vf])

        max_thrust = 1
        dt = 4
        wait_actions = np.linspace(0, 1 - 1/dt, dt)  # percent of duration to wait
        thrust_actions = np.linspace(-max_thrust/2, max_thrust, 6)  # magnitude of thrust to apply
        action_pairs = []
        for wait_action in wait_actions:
            for thrust_action in thrust_actions:
                action_pairs.append([wait_action, thrust_action])

        self.action_map = {index: action_pair for index, action_pair in enumerate(action_pairs)}
        self.action_space = spaces.Discrete(len(action_pairs))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Position (rx, ry, rz), velocity(vx,vy,vz)

        # self.action_space = spaces.Discrete(4)  # Define thrust direction and magnitude
        # self.action_map = {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]}
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # Position (rx, ry, rz), velocity(vx,vy,vz), mass(m)

        # Initialize the 3D plot for the orbit
        self.fig = go.FigureWidget()
        self.fig.update_layout(scene=dict(xaxis_title='X (km)', yaxis_title='Y (km)', zaxis_title='Z (km)'),
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
        # self.add_trajectory(self.final_orbit)

    def step(self, a_ind):
        # Apply the action to simulate the thrust maneuver
        action = self.action_map[a_ind]  # later on - sample the action index to get the action pair
        thrust_magnitude = action[1]  # Thrust magnitude

        a = self.get_semimajor_axis()
        if a < self.earth_radius:
            return self.state, -5000 - np.abs(thrust_magnitude).sum(), True, {}
        period = 2 * np.pi * np.sqrt(a ** 3 / self.earth_mu)

        if action[0] != 0:  self.orbit_propogation(period * action[0])  # Wait 1.5 * T

        unit_v = (self.state[3:6] / np.linalg.norm(self.state[3:6])) * thrust_magnitude * 0.5

        self.state[3:6] += unit_v

        # Calculate reward and check completion
        reward, done = self.compare_orbits()  # Terminate if desired orbit is reached
        reward -= np.abs(thrust_magnitude).sum() * 10  # Penalize large maneuvers
        return self.state, reward, done, {}

    def reset(self):
        self.x_hist = np.empty((6, 0))
        self.state = np.array([6671., 0., 0., 0., 7.72988756, 0.])
        # print(f"Initial Orbit: {self.state}")
        # self.add_trajectory(initial_orbit)
        return self.state

    def get_energy(self, state=False):
        state = self.state if state is False else state
        E = 0.5 * np.linalg.norm(state[3:6]) ** 2 - self.earth_mu / np.linalg.norm(state[0:3])
        return E

    def get_semimajor_axis(self, state=False):
        a = -self.earth_mu / (2 * self.get_energy(state))
        return a

    def add_trajectory(self, state=False, params=dict()):
        # Plot one full orbit
        state = self.state if state is False else state
        a = self.get_semimajor_axis(state)
        if a > 0:
            period = 2 * np.pi * np.sqrt(a ** 3 / self.earth_mu)
            trajectory = self.orbit_propogation(period, state)  # Wait for half orbit

            x = trajectory[0, :]
            y = trajectory[1, :]
            z = trajectory[2, :]

            # Add the trajectory segment to the plot
            color = params.get('color', None)
            label = params.get('label', None)
            self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color), name=label))

    def render(self, mode='human'):
        # Plot state history
        x = self.x_hist[0, :]
        y = self.x_hist[1, :]
        z = self.x_hist[2, :]

        self.add_trajectory(self.initial_orbit, params={"color": px.colors.qualitative.Plotly[2], "label": "Initial orbit"})
        self.fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=px.colors.qualitative.Plotly[3]), name="Trajectory"))
        self.add_trajectory(self.state, params={"color": "orange", "label": "Final orbit"})
        self.add_trajectory(self.final_orbit, params={"color": "red", "label": "Desired orbit"})
        self.fig.show()

    def diff_q(self, t, y):
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
        time_steps = 100
        sol = solve_ivp(self.diff_q, (0, duration), state, max_step=duration / time_steps)
        if state is self.state:
            self.state = sol.y[:, -1]
            self.x_hist = np.concatenate((self.x_hist, sol.y), axis=1)
        return sol.y

    def get_orbital_elements(self, state=False):
        state = self.state if state is False else state
        r = state[0:3]
        v = state[3:6]
        E = 0.5 * np.linalg.norm(v) ** 2 - self.earth_mu / np.linalg.norm(r)

        a = -self.earth_mu / (2 * E)
        h = np.cross(r, v)
        e = np.sqrt(1 + (2 * np.linalg.norm(h) ** 2 * E) / (self.earth_mu ** 2))
        i = np.arccos(h[2] / np.linalg.norm(h))

        return a, e, i
    
    def calculate_hohmann(self):
        r1 = self.initial_orbit_radius
        r2 = self.final_orbit_radius

        delta_v1 = np.sqrt(self.earth_mu / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)
        delta_v2 = np.sqrt(self.earth_mu / r2) * (1 - np.sqrt(2 * r1 / (r1 + r2)))

        # Maybe you can use delta v ( v at start and end of orbit transfer to check how similar it is?)

        a = (r1 + r2) / 2
        e = (r2 - r1) / (r1 + r2)

        return delta_v1, delta_v2, a, e

    def compare_orbits(self):
        # compare current and goal orbits
        # final - actual/final
        aTrue, eTrue, iTrue = self.get_orbital_elements(self.final_orbit)
        a, e, i = self.get_orbital_elements()

        aDiff = abs((aTrue - a) / aTrue)
        eDiff = max(abs(eTrue - e), 0.01)  # /max(0.1,eTrue)
        iDiff = abs(iTrue - i) / max(0.01, iTrue)
        # print(f"aDiff: {aDiff}\neDiff: {eDiff}\niDiff: {iDiff}")

        aMatch = (aDiff < 0.01)
        eMatch = (eDiff < 0.01)


        delta_v1, delta_v2, a_h, e_h = self.calculate_hohmann()
        aDiff_h = abs((a_h - a) / a_h)
        eDiff_h = max(abs(e_h - e), 0.01)  

        aMatch_h = (aDiff_h < 0.1)
        eMatch_h = (eDiff_h < 0.1)



        # reward = 10/aDiff + min(100, 1/eDiff)  # incentive
        reward = 1
        done = False

        '''
        # Hohmann reference trajectory check
        if aMatch_h and eMatch_h:
            reward += 10
            print("hohmann reached")
        '''

        if aMatch and eMatch:
            reward += 2000
            done = True
            print("aMatched and eMatched")
        # reward = (aMatch + eMatch) * 500
        return reward, done
    
    '''
    def compare_orbits(self):
        # compare current and goal orbits
        aTrue, eTrue, iTrue = self.get_orbital_elements(self.final_orbit)
        a, e, i = self.get_orbital_elements()

        # Calculate differences in orbital parameters
        aDiff = abs(aTrue - a) / aTrue
        eDiff = abs(eTrue - e)
        iDiff = abs(iTrue - i)

        # Intermediate rewards for getting closer to the target orbit
        reward = 0
        reward += max(0, 1 - aDiff) * 100  # Reward for semi-major axis
        reward += max(0, 1 - eDiff) * 100  # Reward for eccentricity
        reward += max(0, 1 - iDiff) * 100  # Reward for inclination

        done = False
        if aDiff < 0.05 and eDiff < 0.05 and iDiff < 0.05:
            reward += 2000
            done = True
            print("Target orbit reached")

        # Penalize for excessive fuel consumption
        reward -= np.abs(self.state[3:6]).sum() * 10

        return reward, done

    '''


# Example usage
if __name__ == "__main__":
    env = SpacecraftEnv()
    for i in range(50):
        a = env.action_space.sample()
        state_, reward_, done_, _ = env.step(a)
        if done_: break
    env.render()
