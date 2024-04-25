import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Constants for the Moon
moon_radius = 1737.1  # km
moon_mu = 4902.8  # km^3/s^2
distance_earth_moon = 384400.0  # km
earth_mu = 398600.0  # km^3/s^2

# Calculate the Sphere of Influence (SOI) for the Moon
r_SOI_moon = distance_earth_moon * (moon_mu / earth_mu)**(2/5)

# Define the differential equation with SOI
def diff_q(t, y, earth_mu, moon_mu, distance_earth_moon, r_SOI_moon):
    # unpack the state vector
    rx, ry, rz, vx, vy, vz = y
    r_earth = np.array([rx, ry, rz])
    r_moon = np.array([rx - distance_earth_moon, ry, rz])  # Assuming Moon is along the x-axis at this distance
    r_earth_norm = np.linalg.norm(r_earth)
    r_moon_norm = np.linalg.norm(r_moon)

    if r_moon_norm < r_SOI_moon:
        # Moon's gravity dominates
        ax, ay, az = -moon_mu * r_moon / r_moon_norm**3
    else:
        # Earth's gravity dominates
        ax, ay, az = -earth_mu * r_earth / r_earth_norm**3

    return [vx, vy, vz, ax, ay, az]

# Initial conditions remain the same as your previous example

# Solve the ODE using the updated function with the SOI logic
sol = solve_ivp(fun=diff_q, t_span=t_span, y0=y0, 
                args=(earth_mu, moon_mu, distance_earth_moon, r_SOI_moon), 
                method='RK45', rtol=1e-6, atol=1e-9)

# Extract the solution
rs = sol.y[:3].T  # Position vector history


def plot_orbit(rs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(rs[:, 0], rs[:, 1], rs[:, 2])
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.show()

# Plot the results
plot_orbit(rs)
