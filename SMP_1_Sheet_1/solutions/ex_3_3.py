#!/usr/bin/env python3

import numpy as np

import ex_3_1
import ex_3_2

if __name__ == "__main__":

    """
    During the simulation, measure the distance between earth and moon in
    every time-step.
    Run the simulation with a time-step of ∆t = 0.01 for 20 years for the
    different integrators and plot the distance between earth and moon over
    time. Compare the results obtained with the different integrators!
    """

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
    time = 20
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

    # create data storage options for euler
    x_euler = np.zeros((int(time / timestep), x_init.shape[0], x_init.shape[1]))
    x_euler[0] = x_init
    v_euler = np.zeros((int(time / timestep), v_init.shape[0], v_init.shape[1]))
    v_euler[0] = v_init

    for i in range(1, int(time / timestep)):
        x_sympletic_euler[i], v_sympletic_euler[i] = ex_3_2.step_symplectic_euler(x_sympletic_euler[i - 1], v_sympletic_euler[i - 1], timestep, m, g)
        x_velocity_verlet[i], v_velocity_verlet[i], force_old_verlet = ex_3_2.step_velocity_verlet(x_velocity_verlet[i - 1], v_velocity_verlet[i - 1], timestep, m, g, force_old_verlet)
        x_euler[i], v_euler[i] = ex_3_1.step_euler(x_euler[i - 1], v_euler[i - 1], timestep, m, g, ex_3_1.forces)

    # plot the distance of the body 1 and 2 over time at index 1 and 2
    plt.plot(np.linalg.norm(x_sympletic_euler[:, :, 2] - x_sympletic_euler[:, :, 1], axis=1), label="Symplectic Euler")
    plt.plot(np.linalg.norm(x_velocity_verlet[:, :, 2] - x_velocity_verlet[:, :, 1], axis=1), label="Velocity Verlet")
    plt.plot(np.linalg.norm(x_euler[:, :, 2] - x_euler[:, :, 1], axis=1), label="Euler")
    plt.legend()
    plt.show()