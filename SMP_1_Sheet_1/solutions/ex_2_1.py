#!/usr/bin/env python3

import numpy as np

def force(mass, gravity):
    return np.array([0, -mass * gravity])

def step_euler(x, v, dt, mass, gravity, f):
    # first we calculate the acceleration
    force_full = force(mass, gravity) / mass
    # then we update the position
    x = x + v * dt
    # and the velocity
    v = v + force_full * dt

    return x, v


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # now we test our code with three different initial masses, and then simulate it unitil the x2 positions is equal or smaller than 0
    # then we plot our results
    masses = [2, 10, 100]
    start_velocity = np.array([60, 60])
    start_position = np.array([0, 0])

    for mass in masses:
        x = start_position
        v = start_velocity
        dt = 0.1
        gravity = 9.81
        positions = [x]
        while x[1] >= 0:
            x, v = step_euler(x, v, dt, mass, gravity, force)
            positions.append(x)
            print(x)
        positions = np.array(positions)
        plt.plot(positions[:, 0], positions[:, 1], label=f"mass: {mass}", alpha=0.2)

    plt.xlabel(" $x$ in m ")
    plt.ylabel(" $y$ in m ")
    plt.legend()
    plt.show()

