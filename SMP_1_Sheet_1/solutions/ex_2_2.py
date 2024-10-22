#!/usr/bin/env python3

import numpy as np

import ex_2_1

def force(mass, gravity, v, gamma, v_0):
    # first we get the force from gravity from the previous exercise
    gravity_force = ex_2_1.force(mass, gravity)

    # then we calculate the drag force
    drag_force = -1 * gamma * (v - v_0)

    return gravity_force + drag_force

def step_euler(x, v, dt, mass, gravity, gamma, v_0):
    # first we calculate the acceleration
    f = force(mass, gravity, v, gamma, v_0) / mass
    # then we update the position
    x = x + v * dt
    # and the velocity
    v = v + f * dt

    return x, v

def run(x, v, dt, mass, gravity, gamma, v_0):
    # now given the intial conditions we can simulate the movement of the object
    # we return the positions of the trajectory at the end
    # we again use the euler method to simulate the movement (from the previous exercise)

    positions = [x]
    while x[1] >= 0:
        x, v = step_euler(x, v, dt, mass, gravity, gamma, v_0)
        positions.append(x)

    return np.array(positions)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # now we test our code with three different initial masses, and then simulate it unitil the x2 positions is equal or smaller than 0
    # we test for three different wind speeds
    # then we plot our results
    mass = 2
    start_velocity = np.array([60, 60])
    start_position = np.array([0, 0])
    dt = 0.1
    gravity = 9.81
    gamma = 0.1
    v_0 = [np.array([0, 0]), np.array([-30, 0]), np.array([-195, 0])]

    # first we plot the trajectory without wind and without friction (as in the previous exercise)
    x = start_position
    v = start_velocity
    positions = [x]
    while x[1] >= 0:
        x, v = ex_2_1.step_euler(x, v, dt, mass, gravity, force)
        positions.append(x)

    positions = np.array(positions)
    plt.plot(positions[:, 0], positions[:, 1], label="no wind, no friction", alpha=0.8)

    # then we plot the trajectory with wind
    for v_0_i in v_0:
        x = start_position
        v = start_velocity
        positions = run(x, v, dt, mass, gravity, gamma, v_0_i)
        plt.plot(positions[:, 0], positions[:, 1], label=f"wind: {v_0_i}", alpha=0.8)


    plt.xlabel(" $x$ in m ")
    plt.ylabel(" $y$ in m ")
    plt.legend()
    plt.show()

