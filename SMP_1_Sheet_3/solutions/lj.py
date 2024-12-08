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
    def __init__(self, dt, x, v, box, r_cut, shift, thermostat, warmup, f_max):
        self.dt = dt
        self.x = x.copy()
        self.v = v.copy()
        self.box = box.copy()
        self.r_cut = r_cut
        self.shift = shift
        self.thermostat = thermostat
        self.warmup = warmup
        self.f_max = f_max

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
        potential_energy = 0.5 * np.sum(self.e_pot_ij_matrix) # assuming all particles have a mass of unity
        kinetic_energy = np.sum(np.linalg.norm(self.v, axis=0)**2) * 0.5 # assuming all particles have a mass of unity

        return [potential_energy, kinetic_energy]

    def temperature(self):
        # the temperature is defined as the average kinetic energy per degree of freedom
        return 2 * self.energy()[1] / ( 2 * self.n)

    def pressure(self):
        # the is defined like this:
        # P = 1/2A * ( \sum_{i=1}^{N} m v_i^2 + \sum_{i=1}^{N} \sum_{j=i+1}^{N} r_ij f_ij )
        # where A is the area of the box
        # and r_ij is the distance vector between particle i and j
        # and f_ij is the force vector between particle i and j
        # we assume that all particles have a mass of unity
        return 1.0 / np.prod(self.box) * (np.sum(self.v**2) + np.sum(np.sum(self.r_ij_matrix * self.f_ij_matrix, axis=2)))

    def rdf(self):
        # compute the radial distribution function
        # we need to compute the radial distribution function
        # we need to compute the histogram of the distances

        r = np.linalg.norm(self.r_ij_matrix, axis=2)
        # we need to compute the histogram of the distances
        h, b = np.histogram(r.flatten(), bins=100, range=(0.8, 5.0))
        bin_r = 0.5 * (b[1:] + b[:-1])
        # we need to compute the radial distribution function
        # g(r) = 1 / (rho * 4 * pi * r^2 * dr) * sum_ij <delta(r - |r_ij|)>
        n = 1/(4*np.pi*DENSITY*bin_r**2)

        return (h *n, bin_r)



    def propagate(self):
        # update positions
        self.x += self.v * self.dt + 0.5 * self.f * self.dt * self.dt

        # half update of the velocity
        self.v += 0.5 * self.f * self.dt

        # compute new forces
        self.forces()

        if self.warmup:
            capped_f = np.where(np.abs(self.f) > self.f_max, np.sign(self.f) * self.f_max, self.f)

            # if capped froce are the same as the original forces, we are done
            if np.allclose(capped_f, self.f):
                self.f = capped_f
                self.warmup = False
            else:
                self.f = capped_f

        # we assume that all particles have a mass of unity

        # second half update of the velocity
        self.v += 0.5 * self.f * self.dt

    def rescale_velocities_to_temp(self, target_temperature, current_temperature):
        # the temperature is defined as the average kinetic energy per degree of freedom
        scaling_factor = np.sqrt(target_temperature / current_temperature)

        self.v *= scaling_factor





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
    
    # next we need a flag to tun on or off the termostate
    # we write on or off to the flag
    # the thermostat is a somple velocity scaling thermostat
    # the temperature is kept constant by scaling the velocities
    parser.add_argument(
        '--thermostat',
        type=str,
        default='off',
        help='Thermostat on or off.')
    
    # add the warm up time to the simulation
    parser.add_argument(
        '--warmup',
        type=str,
        default='off',
        help='Warmup time.')
    

    args = parser.parse_args()
        
    thermostat_bool = False
    if args.thermostat == 'on':
        thermostat_bool = True

    warmpup_bool = False
    if args.warmup != 'off':
        warmup_bool = True


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

    F_MAX = 20.0

    if not args.cpt or not os.path.exists(args.cpt):
        logging.info("Starting from scratch.")
        # particle positions
        # x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
        #                                    np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T
        # positions are now supposed to me randomly distributed
        x = np.random.random((DIM, N_PART)) * BOX[:, np.newaxis]

        # random particle velocities
        v = 0.5*(2.0 * np.random.random((DIM, N_PART)) - 1.0)

        positions = []
        energies = []
        pressures = []
        temperatures = []
        rdfs = []
        time = []
        extensions = False
    elif args.cpt and os.path.exists(args.cpt):
        logging.info("Reading state from checkpoint.")
        data = read_checkpoint(args.cpt)
        x = data['x']
        v = data['v']
        f = data['f']
        time = data['time']
        final_time = time[-1]
        positions = data['positions']
        energies = data['energies']
        pressures = data['pressures']
        temperatures = data['temperatures']
        rdfs = data['rdfs']
        extensions = True


    sim = Simulation(DT, x, v, BOX, R_CUT, SHIFT, thermostat_bool, warmup_bool, F_MAX)

    # If checkpoint is used, also the forces have to be reloaded!
    if args.cpt and os.path.exists(args.cpt):
        sim.f = f

    if extensions:
        logging.info("Extending simulation by %f" % args.extend)
        total_steps = int(args.extend / DT)
    else:
        total_steps = N_TIME_STEPS

    for i in tqdm.tqdm(range(total_steps)):

        if sim.warmup:
            logging.info("Warmup step %d" % i)
            # increase max force by 10% every step
            sim.f_max *= 1.1  

        sim.propagate()

        if i % SAMPLING_STRIDE == 0:
            temp = sim.temperature()

            if not sim.warmup:
                positions.append(sim.x.copy())
                pressures.append(sim.pressure())
                energies.append(sim.energy())
                temperatures.append(temp)
                rdfs.append(sim.rdf())
                if extensions:
                    time.append((i * DT) +  final_time)
                else:
                    time.append(i * DT)

            if thermostat_bool:
                sim.rescale_velocities_to_temp(3.0, temp)



    state = {
        'x': sim.x,
        'v': sim.v,
        'f': sim.f,
        'positions': positions,
        'energies': energies,
        'pressures': pressures,
        'temperatures': temperatures,
        'rdfs': rdfs,
        'time': time
    }
    write_checkpoint(state, args.cpt, overwrite=True)