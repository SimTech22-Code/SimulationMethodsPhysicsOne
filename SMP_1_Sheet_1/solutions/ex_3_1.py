#!/usr/bin/env python3
import os
import numpy as np
import scipy.constants

def force(r_ij, m_i, m_j, g):
    return -1 * g * m_i * m_j * r_ij * (1 / (np.linalg.norm(r_ij)**3))

def step_euler(x, v, dt, mass, g, forces_function):
    # here we calculate the next position of the bodys
    # first we update all of the postions and then the velocities
    a = forces(x, mass, g) / mass

    x_new = x + v * dt
    v_new = v + a * dt

    return x_new, v_new

def forces(x, masses, g):
    # calculate the forces acting on each body
    forces_array = np.zeros_like(x)

    for i in range(0, x.shape[1]):
        for j in range(0, x.shape[1]):
            if i != j:
                # x shape is (2, 6)
                r_ij = x[:, i] - x[:, j]
                forces_array[:, i] += force(r_ij, masses[i], masses[j], g)

    return forces_array

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # read in the data file in files directory
    # file name is solar_system.npz
    data = np.load(os.path.join(os.path.dirname(__file__), "..", "files", "solar_system.npz"))

    names = data['names']
    x_init = data['x_init']
    v_init = data['v_init']
    m = data['m']
    g = data['g']
    timestep = 0.0001
    time = 1

    # create an array to store the positions of the bodys
    x = np.zeros((int(time / timestep), x_init.shape[0], x_init.shape[1]))
    x[0] = x_init

    # create an array to store the velocities of the bodys
    v = np.zeros((int(time / timestep), v_init.shape[0], v_init.shape[1]))
    v[0] = v_init

    for i in range(1, int(time / timestep)):
        x[i], v[i] = step_euler(x[i - 1], v[i - 1], timestep, m, g, forces)

    # plot the positions of the bodys, name in numpy bytes
    for i in range(0, x_init.shape[1]):
        plt.plot(x[:, 0, i], x[:, 1, i], label = names[i].decode("utf-8"))

    plt.legend()
    plt.show()


    """
    Perform the simulation for different time-steps ∆t ∈ {0.0001, 0.001} and
    plot the trajectory of the moon (particle number 2) in the rest frame of
    the earth (particle number 1). Are the trajectories satisfactory?
    """
    timesteps = [0.0001, 0.001]

    for timestep in timesteps:

        # create an array to store the positions of the bodys
        x = np.zeros((int(time / timestep), x_init.shape[0], x_init.shape[1]))
        x[0] = x_init

        # create an array to store the velocities of the bodys
        v = np.zeros((int(time / timestep), v_init.shape[0], v_init.shape[1]))
        v[0] = v_init

        for i in range(1, int(time / timestep)):
            x[i], v[i] = step_euler(x[i - 1], v[i - 1], timestep, m, g, forces)

        # plot positions of the moon in the rest frame of the earth, pos 1 and 2
        plt.plot(x[:, 0, 2] - x[:, 0, 1], x[:, 1, 2]- x[:, 1, 1], label = 'timestep: '+ str(timestep), alpha = 0.5)

    plt.legend()
    plt.show()

    



