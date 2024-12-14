#!/usr/bin/env python3

# introduce classes to the students
class Simulation:
    def __init__(self, dt, x, v, box, r_cut, shift, F_max):
        self.dt = dt
        self.x = x.copy()
        self.v = v.copy()
        self.box = box.copy()
        self.r_cut = r_cut
        self.shift = shift
        #force capping value, will be None if warmup is not enabled
        self.F_max = F_max

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
        pot_energy = np.sum(self.e_pot_ij_matrix) * 0.5
        # TODO calculate kinetic energy from the velocities self.v and return both energy components
        kin_energy = np.sum(np.linalg.norm(self.v, axis=0)**2) * 0.5
        return [pot_energy, kin_energy]

    def temperature(self):
        # TODO
        # Temperature computed from kinetic energy in units of k_B^-1
        kin_energy = self.energy()[1]
        return 2*kin_energy/(DIM * N_PART)

    def pressure(self):
        # TODO
        # Pressure computed from the ideal gas and interaction contribution
        P_ideal = 0.5/VOLUME * np.sum(np.linalg.norm(self.v, axis=0)**2)
        P_inter_ij_matrix = np.tril(np.multiply(self.f_ij_matrix, self.r_ij_matrix), -1)
        P_inter = 0.5/VOLUME * np.sum(P_inter_ij_matrix)
        return P_ideal + P_inter

    def rdf(self):
        # TODO
        # RDF computation
        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        hist, bin_edges = np.histogram(r, bins=100, range=(0.8,5.0))
        bin_r = 0.5*(bin_edges[:-1]+bin_edges[1:])
        norm = 1/(4*np.pi*DENSITY*bin_r**2)
        return (norm*hist, bin_r)

    def propagate(self):
        # update positions
        self.x += self.v * self.dt + 0.5 * self.f * self.dt * self.dt

        # half update of the velocity
        self.v += 0.5 * self.f * self.dt

        # compute new forces
        self.forces()
        if self.F_max:
            # resacle forces if they exceed the force capping value
            # the max force is capped at F_max, the other forces are rescaled accordingly
            capped_forces = np.where(np.linalg.norm(self.f, axis=0) > self.F_max, self.F_max * self.f / np.linalg.norm(self.f, axis=0), self.f)
            if np.array_equal(capped_forces, sim.f):
                print('\nWarmup finished, F_max = ' + str(self.F_max) + '.')
                self.F_max = None
            else:
                sim.f = capped_forces
        # we assume that all particles have a mass of unity

        # second half update of the velocity
        self.v += 0.5 * self.f * self.dt


def write_checkpoint(state, path, overwrite=False):
    if os.path.exists(path) and not overwrite:
        raise RuntimeError("Checkpoint file already exists")
    with open(path, 'wb') as fp:
        pickle.dump(state, fp)


if __name__ == "__main__":
    import argparse
    import pickle
    import itertools
    import logging

    import os.path

    import numpy as np
    import scipy.spatial  # todo: probably remove in template
    import tqdm

    import pathlib
    file_dir = pathlib.Path(__file__).parent

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'N_per_side',
        type=int,
        help='Number of particles per lattice side.')
    parser.add_argument(
        '--cpt',
        type=lambda cpt: str((file_dir / cpt) if not pathlib.Path(cpt).is_absolute() else pathlib.Path(cpt)),
        help='Path to checkpoint.')
    parser.add_argument(
        '--thermostat',
        type=float,
        help='Thermostat temperature.')
    parser.add_argument(
        '--warmup',
        type=float,
        help='Maximum warmup force.')
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
        if args.warmup: # random initial positions if warmup is enabled, otherwise initial positions on a lattice
            x = np.random.random((DIM, N_PART))
            x[0] = x[0]*BOX[0]
            x[1] = x[1]*BOX[1]
        else:
            x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
                                                np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T

        # random particle velocities
        v = 0.5*(2.0 * np.random.random((DIM, N_PART)) - 1.0)

        positions = []
        velocities = []
        forces = []
        energies = []
        pressures = []
        temperatures = []
        rdfs = []
    elif args.cpt and os.path.exists(args.cpt):
        logging.info("Reading state from checkpoint.")
        with open(args.cpt, 'rb') as fp:
            data = pickle.load(fp)
        
        positions = data['positions']
        velocities = data['velocities']
        forces = data['forces']
        energies = data['energies']
        pressures = data['pressures']
        temperatures = data['temperatures']
        rdfs = data['rdfs']

        # read particle positions and velocities from last time step of previous simulation
        x = positions[-1]
        v = velocities[-1]

    sim = Simulation(DT, x, v, BOX, R_CUT, SHIFT, args.warmup)

    # If checkpoint is used, also the forces have to be reloaded!
    if args.cpt and os.path.exists(args.cpt):
        sim.f = data['forces'][-1]

    kin_energy_0 = sim.energy()[1]

    for i in tqdm.tqdm(range(N_TIME_STEPS)):

        sim.propagate()

        if i % SAMPLING_STRIDE == 0:
            if sim.F_max: # increase force capping value and rescale velocities if warmup is enabled, otherwise save simulations results in lists
                sim.F_max = 1.1*sim.F_max
                # rescale_E = np.sqrt(np.abs(kin_energy_0/sim.energy()[1]))
                # sim.v = rescale_E*sim.v
            else:
                positions.append(sim.x.copy())
                velocities.append(sim.v.copy())
                forces.append(sim.f.copy())
                pressures.append(sim.pressure())
                energies.append(sim.energy())
                temperatures.append(sim.temperature())
                rdfs.append(sim.rdf())

                # rescale velocities for enabled thermostat
                if args.thermostat:
                    rescale_T = np.sqrt(args.thermostat/temperatures[-1])
                    sim.v = rescale_T*sim.v

    if args.cpt: # write checkpoint with positions and velocities to be used in next simulation, as well as (macroscopic) observalbles and RDFs for analysis 
        state = {'energies': energies,
                 'positions': positions,
                 'velocities': velocities,
                 'forces': forces,
                 'temperatures': temperatures,
                 'pressures': pressures,
                 'rdfs': rdfs}
        write_checkpoint(state, args.cpt, overwrite=True)
