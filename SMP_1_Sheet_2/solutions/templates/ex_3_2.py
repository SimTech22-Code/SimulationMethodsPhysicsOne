#!/usr/bin/env python3

import numpy as np
import scipy.linalg


def lj_potential(r_ij: np.ndarray) -> float:
    # Lennard-Jones potential
    # V(r) = 4 * epsilon * [(sigma / r)^12 - (sigma / r)^6]
    r = np.linalg.norm(r_ij)
    return 4 * ((1/r**12) - (1/r**6))
    

def lj_force(r_ij: np.ndarray) -> np.ndarray:
    # Lennard-Jones force
    r = np.linalg.norm(r_ij)
    return 24 * ((2/r**14 - 1/r**8)) * r_ij

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    d = np.linspace(0.85, 2.5, 1000)

    # Lennard-Jones potential and force
    V = np.array([lj_potential(np.array([d_i, 0])) for d_i in d])
    F = np.array([lj_force(np.array([d_i, 0]))[0] for d_i in d])

    # Fully repulsive potential
    min_r = 2**(1/6)
    shift = lj_potential(np.array([min_r, 0]))
    V_repulsive = np.array([lj_potential(np.array([d_i, 0])) - shift if d_i <= min_r else 0 for d_i in d])

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].plot(d, V, label="Lennard-Jones potential")
    ax[0].plot(d, V_repulsive, label="Fully repulsive potential")
    ax[0].set_xlabel("d")
    ax[0].set_ylabel("V(d)")
    ax[0].legend()

    ax[1].plot(d, F, label="Lennard-Jones force")
    ax[1].set_xlabel("d")
    ax[1].set_ylabel("F(d)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
