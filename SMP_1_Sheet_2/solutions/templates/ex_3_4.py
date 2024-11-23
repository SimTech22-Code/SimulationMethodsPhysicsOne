#!/usr/bin/env python3

import numpy as np
import pathlib

import ex_3_2

plots_path = pathlib.Path(__file__).resolve().parent.parent.joinpath('plots')
plots_path.mkdir(parents=True, exist_ok=True)

def minimum_image_vector(x_i, x_j, box):
    r_ij = x_j - x_i
    #apply PBC
    r_ij[0] -= np.round(r_ij[0]/box[0]) * box[0]
    r_ij[1] -= np.round(r_ij[1]/box[1]) * box[1]

    return r_ij

def lj_force(r_ij, r_cutoff):
    if np.linalg.norm(r_ij) <= r_cutoff:
        return ex_3_2.lj_force(r_ij)
    else:
        return [0, 0]
    
def lj_potential(r_ij, r_cutoff, shift):
    if np.linalg.norm(r_ij) <= r_cutoff:
        return ex_3_2.lj_potential(r_ij) - shift
    else:
        return 0

def forces(x: np.ndarray, r_cutoff, box) -> np.ndarray:
    """Compute and return the forces acting onto the particles,
    depending on the positions x."""
    N = x.shape[1]
    f = np.zeros_like(x)
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = minimum_image_vector(x[:, i], x[:, j], box)
            #truncate LJ force (only calculate the force between particles within the cut-off radius)
            f_ij = lj_force(r_ij, r_cutoff)
            f[:, i] -= f_ij
            f[:, j] += f_ij
    return f


def total_energy(x: np.ndarray, v: np.ndarray, r_cutoff, shift, box) -> np.ndarray:
    """Compute and return the total energy of the system with the
    particles at positions x and velocities v."""
    N = x.shape[1]
    E_pot = 0.0
    E_kin = 0.0
    # sum up potential energies
    for i in range(1, N):
        for j in range(i):
            # distance vector
            r_ij = x[:, j] - x[:, i]
            E_pot += lj_potential(r_ij, r_cutoff, shift)
    # sum up kinetic energy
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])
    return E_pot + E_kin


def step_vv(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float, r_cutoff, box):
    # update positions
    x += v * dt + 0.5 * f * dt * dt
    # half update of the velocity
    v += 0.5 * f * dt

    # compute new forces
    f = forces(x, r_cutoff, box)
    # we assume that all particles have a mass of unity

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f

def apply_bounce_back(x, v, box_l=15):
    v[0, np.where(x[0,:]<=0)] = -v[0, np.where(x[0,:]<=0)]
    v[0, np.where(x[0,:]>=box_l)] = -v[0, np.where(x[0,:]>=box_l)]
    v[1, np.where(x[1,:]<=0)] = -v[1, np.where(x[1,:]<=0)]
    v[1, np.where(x[1,:]>=box_l)] = -v[1, np.where(x[1,:]>=box_l)]

    return v


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    DT = 0.01
    T_MAX = 20.0
    N_TIME_STEPS = int(T_MAX / DT)
    r_cutoff = 2.5
    box = [10.0, 10.0]


    # running variables
    time = 0.0

    # particle positions
    x = np.zeros((2, 2))
    x[:, 0] = [3.9, 3.0]
    x[:, 1] = [6.1, 5.0]

    # particle velocities
    v = np.zeros((2, 2))
    v[:, 0] = [-2.0, -2.0]
    v[:, 1] = [2.0, 2.0]

    f = forces(x, r_cutoff, box)
    shift = ex_3_2.lj_potential([r_cutoff,0])

    N_PART = x.shape[1]

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)


    # main loop (without walls)
    with open(f"{str(plots_path)}/"+'ljbillards3_pbc.vtf', 'w') as vtffile:
        # write the structure of the system into the file:
        # N particles ("atoms") with a radius of 0.5
        vtffile.write(f'atom 0:{N_PART - 1} radius 0.5\npbc 10.0 10.0 10.0\n')
        for i in range(N_TIME_STEPS):
            x, v, f = step_vv(x, v, f, DT, r_cutoff, box)
            time += DT

            positions[i, :2] = x
            energies[i] = total_energy(x, v, r_cutoff, shift, box)

            # write out that a new timestep starts
            vtffile.write('timestep\n')
            # write out the coordinates of the particles
            for p in x.T:
                vtffile.write(f"{p[0]} {p[1]} 0.\n")

    traj = np.array(positions)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(N_PART):
        ax1.plot(positions[:, 0, i], positions[:, 1, i], label='{}'.format(i))
    ax1.set_title('Trajectory')
    ax1.set_aspect('equal')
    ax1.set_xlabel('x position')
    ax1.set_ylabel('y position')
    ax1.legend(bbox_to_anchor=(1.1, 1.05))

    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Total energy")
    ax2.plot(energies)
    ax2.set_title('Total energy')
    plt.tight_layout()
    plt.savefig(f"{str(plots_path)}/ex_3_4_plot.pdf", format="pdf")

    plt.show()