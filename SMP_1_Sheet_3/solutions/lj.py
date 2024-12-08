#!/usr/bin/env python3
import argparse
import itertools
import logging
import os
import pickle

import numpy as np
import tqdm



# introduce classes to the students
class Simulation:
    def __init__(self, dt, x, v, box, r_cut, shift):
        self.dt = dt
        self.x = x.copy()
        self.v = v.copy()
        self.box = box.copy()
        self.r_cut = r_cut
        self.shift = shift

        self.n_dims = self.x.shape[0]
        self.n = self.x.shape[1]
        self.f = np.zeros_like(x)

        # both r_ij_matrix and f_ij_matrix are computed in self.forces()
        self.r_ij_matrix = np.zeros((self.n, self.n, self.n_dims))
        self.f_ij_matrix = np.zeros((self.n, self.n, self.n_dims))
        # computed in e_pot_ij_matrix
        self.e_pot_ij_matrix = np.zeros((self.n, self.n))

    def distances(self):
        self.r_ij_matrix = np.repeat([self.x.transpose()], self.n, axis=0)
        self.r_ij_matrix -= np.transpose(self.r_ij_matrix, axes=[1, 0, 2])
        # minimum image convention
        image_offsets = self.r_ij_matrix.copy()
        for nth_box_component, box_component in enumerate(self.box):
            image_offsets[:, :, nth_box_component] = \
                np.rint(image_offsets[:, :, nth_box_component] / box_component) * box_component
        self.r_ij_matrix -= image_offsets

    def energies(self):
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        with np.errstate(all='ignore'):
            self.e_pot_ij_matrix = np.where((r != 0.0) & (r < self.r_cut),
                                            4.0 * (np.power(r, -12.) - np.power(r, -6.)) + self.shift, 0.0)

    def forces(self):
        # first update the distance vector matrix, obeying minimum image convention
        self.distances()
        self.f_ij_matrix = self.r_ij_matrix.copy()
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        with np.errstate(all='ignore'):
            fac = np.where((r != 0.0) & (r < self.r_cut),
                           4.0 * (12.0 * np.power(r, -13.) - 6.0 * np.power(r, -7.)), 0.0)
        for dim in range(self.n_dims):
            with np.errstate(invalid='ignore'):
                self.f_ij_matrix[:, :, dim] *= np.where(r != 0.0, fac / r, 0.0)
        self.f = np.sum(self.f_ij_matrix, axis=0).transpose()

    def energy(self):
        """Compute and return the energy components of the system."""
        # compute energy matrix
        self.energies()
        # TODO compute interaction energy from self.e_pot_ij_matrix
        # TODO calculate kinetic energy from the velocities self.v and return both energy components

    def temperature(self):
        #TODO
        pass

    def pressure(self):
        #TODO
        pass

    def rdf(self):
        #TODO
        pass

    def propagate(self):
        # update positions
        self.x += self.v * self.dt + 0.5 * self.f * self.dt * self.dt

        # half update of the velocity
        self.v += 0.5 * self.f * self.dt

        # compute new forces
        self.forces()
        # we assume that all particles have a mass of unity

        # second half update of the velocity
        self.v += 0.5 * self.f * self.dt


def write_checkpoint(state, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise RuntimeError("Checkpoint file already exists")
    with open(path, 'wb') as fp:
        pickle.dump(state, fp)

def read_checkpoint(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        help='Number of particles per lattice side.')
    parser.add_argument(
        '--cpt',
        type=str,
        help='Path to checkpoint.')
    parser.add_argument(
        '--extend',
        type=float,
        default=0.0,
        help='Additional time to simulate.')
    args = parser.parse_args()

    np.random.seed(2)

    DT = 0.01
    T_MAX = 100.0
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    DIM = 2
    DENSITY = 0.316
    N_PER_SIDE = args.N_per_side
    N_PART = N_PER_SIDE**DIM
    VOLUME = N_PART / DENSITY
    BOX = np.ones(DIM) * VOLUME**(1. / DIM)

    SAMPLING_STRIDE = 3

    if not args.cpt or not os.path.exists(args.cpt):
        logging.info("Starting from scratch.")
        # particle positions
        x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
                                            np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T

        # random particle velocities
        v = 0.5*(2.0 * np.random.random((DIM, N_PART)) - 1.0)

        positions = []
        energies = []
        pressures = []
        temperatures = []
        rdfs = []
    elif args.cpt and os.path.exists(args.cpt):
        logging.info("Reading state from checkpoint.")
        data = read_checkpoint(args.cpt)
        x = data['x']
        v = data['v']
        f = data['f']
        positions = data['positions']
        energies = data['energies']
        pressures = data['pressures']
        temperatures = data['temperatures']
        rdfs = data['rdfs']

    sim = Simulation(DT, x, v, BOX, R_CUT, SHIFT)

    # If checkpoint is used, also the forces have to be reloaded!
    if args.cpt and os.path.exists(args.cpt):
        sim.f = f

    total_steps = N_TIME_STEPS + int(args.extend / DT)

    for i in tqdm.tqdm(range(N_TIME_STEPS, total_steps)):
        sim.propagate()

        if i % SAMPLING_STRIDE == 0:
            positions.append(sim.x.copy())
            pressures.append(sim.pressure())
            energies.append(np.sum(sim.energy()))
            temperatures.append(sim.temperature())
            rdfs.append(sim.rdf())

    state = {
        'x': sim.x,
        'v': sim.v,
        'f': sim.f,
        'positions': positions,
        'energies': energies,
        'pressures': pressures,
        'temperatures': temperatures,
        'rdfs': rdfs
    }
    write_checkpoint(state, args.cpt, overwrite=True)