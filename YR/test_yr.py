from two_body_env import TwoBodyEnv
import numpy as np
import random

env = TwoBodyEnv()
A = [-2, -1, 0, 1, 2]
cost = [-10, -6, 0, -6, -10]
n = 6
m = len(A)
numStates = 1000
S = np.zeros((numStates, n))
Q = np.zeros((numStates, m))

xo = env.x
xf = 1.5 * xo
levels = 1000
S = np.linspace(env.get_energy(0.9 * xo), env.get_energy(1.1 * xf), levels)


def find_closest_state(target_state):
    # Compute Euclidean distance between target vector and each vector in the matrix
    distances = np.linalg.norm(S - target_state, axis=1)
    closest_index = np.argmin(distances)
    return closest_index


def policy(s, e):
    if random.random() < e:
        return random.randint(0, len(A) - 1)
    else:
        return np.argmax(Q[s])


k = 0
e = 0.95
y = 0.95
s = np.abs(S - env.get_energy()).argmin()
while k < 100:
    a_ind = policy(s, e)
    r = cost[a_ind]
    unit_dv = env.x[3:6] / np.linalg.norm(env.x[3:6])
    env.x += np.concatenate(([0, 0, 0], A[a_ind] * unit_dv))
    env.orbit_propogation()
    sp = np.abs(S - env.get_energy()).argmin()
    experience = (s, a_ind, r, sp)
    k += 1
