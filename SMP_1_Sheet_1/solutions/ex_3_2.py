#!/usr/bin/env python3

import numpy as np

import ex_3_1

def step_symplectic_euler(x, v, dt, mass, g):
    # order in flip

    a = ex_3_1.forces(x, mass, g) / mass

    v_new = v + a * dt
    x_new = x + v * dt

    return x_new, v_new

def step_velocity_verlet(x, v, dt, mass, g, force_old):
    # implementing the velocity verlet method

    x_new = x + v * dt + 0.5 * (force_old/mass) * (dt**2)

    v_new_half = v + 0.5 * (force_old/mass) * dt

    force_new = ex_3_1.forces(x_new, mass, g)

    v_new = v_new_half + 0.5 * (force_new / mass) * dt

    return x_new, v_new, force_new



if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # read in the data file in files directory
    # file name is solar_system.npz
    data = np.load(os.path.join(os.path.dirname(__file__), "..", "files", "solar_system.npz"))

    names = data['names']
    x_init = data['x_init']
    v_init = data['v_init']
    m = data['m']
    g = data['g']
    time = 1
    timestep = 0.01

    # create data storage options for symplectic euler
    x_sympletic_euler = np.zeros((int(time / timestep), x_init.shape[0], x_init.shape[1]))
    x_sympletic_euler[0] = x_init
    v_sympletic_euler = np.zeros((int(time / timestep), v_init.shape[0], v_init.shape[1]))
    v_sympletic_euler[0] = v_init

    # create data storage options for velocity verlet
    x_velocity_verlet = np.zeros((int(time / timestep), x_init.shape[0], x_init.shape[1]))
    x_velocity_verlet[0] = x_init
    v_velocity_verlet = np.zeros((int(time / timestep), v_init.shape[0], v_init.shape[1]))
    v_velocity_verlet[0] = v_init
    force_old_verlet = ex_3_1.forces(x_init, m, g)

    for i in range(1, int(time / timestep)):
        x_sympletic_euler[i], v_sympletic_euler[i] = step_symplectic_euler(x_sympletic_euler[i - 1], v_sympletic_euler[i - 1], timestep, m, g)
        x_velocity_verlet[i], v_velocity_verlet[i], force_old_verlet = step_velocity_verlet(x_velocity_verlet[i - 1], v_velocity_verlet[i - 1], timestep, m, g, force_old_verlet)

    # plot positions of the moon in the rest frame of the earth, pos 1 and 2
    plt.plot(x_sympletic_euler[:, 0, 2] - x_sympletic_euler[:, 0, 1], x_sympletic_euler[:, 1, 2]- x_sympletic_euler[:, 1, 1], label = 'sympletic euler', alpha = 0.5)
    plt.plot(x_velocity_verlet[:, 0, 2] - x_velocity_verlet[:, 0, 1], x_velocity_verlet[:, 1, 2]- x_velocity_verlet[:, 1, 1], label = 'velocity verlet', alpha = 0.5)

    plt.legend()
    plt.show()

