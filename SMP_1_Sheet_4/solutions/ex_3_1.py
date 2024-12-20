import numpy as np
from tqdm import tqdm
import pickle

# Physical constants and parameters
# The user should define these parameters, for example:
N_PARTICLES = 100
T = 1.0
DT = 0.01
T_INT = 2000.0   # total simulation time
GAMMA = 1.0

def step_vv_langevin(x, v, f, dt, gamma, T):
    """
    Perform a single step of the velocity Verlet algorithm for the Langevin thermostat.
    x: positions (N,3)
    v: velocities (N,3)
    f: forces (N,3) - deterministic part
    dt: time step
    gamma: friction coefficient
    T: temperature
    """
    W = np.sqrt(2 * gamma * T / dt) * np.random.randn(*v.shape)

    # Total force G(t) = F(t) + W(t)
    G = f + W

    # First position update according to Eq. (12)
    x_new = x + v * dt + 0.5 * dt**2 * G

    # Now we need G(t+dt) for the velocity update.
    # Update forces (if any deterministic forces exist, compute them here)
    # In this simple example, we assume no deterministic forces:
    f_new = np.zeros_like(f)

    # Update velocity according to Eq. (13)
    v_new = (v + 0.5 * dt * G) / (1 + 0.5 * dt * gamma)


    return x_new, v_new, f_new

def step_vv_anderson(x, v, f, dt, nu, T):
    """
    Perform a single step of the velocity Verlet algorithm with Andersen thermostat.
    x: positions (N,3)
    v: velocities (N,3)
    f: forces (N,3)
    dt: time step
    nu: collision frequency
    T: temperature
    """
    # Assumes mass = 1 for simplicity and forces = -grad(U)
    # First half step: update positions
    x_new = x + v * dt + 0.5 * f * dt**2
    
    # Compute new forces f_new at x_new
    # For this example, we assume no external forces or a known potential.
    # Replace this with the correct force evaluation in your simulation:
    f_new = np.zeros_like(f)  # Placeholder
    
    # Second half step: update velocities
    v_new = v + 0.5 * (f + f_new) * dt
    
    # Andersen thermostat step:
    # After integration, each particle may be "collided" with a heat bath
    # with probability nu*dt.
    N = x_new.shape[0]
    random_numbers = np.random.rand(N)
    for i in range(N):
        if random_numbers[i] < nu * dt:
            # Draw new velocities from a Maxwell-Boltzmann distribution at temperature T
            # Assuming k_B = 1 and mass = 1 for simplicity:
            v_new[i] = np.sqrt(T) * np.random.randn(3)
    
    return x_new, v_new, f_new


def initialize_system(n_particles):
    """
    Initialize the system with zero positions and velocities.
    """
    x = np.zeros((n_particles, 3))
    v = np.zeros((n_particles, 3))
    f = np.zeros((n_particles, 3))
    return x, v, f

def compute_instaneous_temperature(v):
    """
    Compute the instantaneous temperature from the velocities.
    T = (m/(3N k_B)) sum_i |v_i|^2
    Assuming m = 1 and k_B = 1 for simplicity:
    T_inst = (1/(3N)) sum_i |v_i|^2
    """
    return np.sum(np.linalg.norm(v, axis=0)**2) / (3 * N_PARTICLES)

if __name__ == "__main__":
    # System parameters
    # (Make sure these match the defined constants above)
    n_particles = N_PARTICLES

    # Lists for observables
    temperature = []
    particle_speeds = []
    particle_velocities = []
    particle_positions = []

    # Initialize the system
    x, v, f = initialize_system(n_particles)

    # Perform time integration
    n_steps = int(T_INT/DT)
    for i in tqdm(range(n_steps)):
        x, v, f = step_vv_langevin(x, v, f, DT, GAMMA, T)
        # x, v, f = step_vv_anderson(x, v, f, DT, 0.1, T)
        temperature.append(compute_instaneous_temperature(v))
        particle_velocities.append(v.copy())
        particle_speeds.append(np.linalg.norm(v, axis=1))
        particle_positions.append(x.copy())

    # Write the observables to file
    data = {
        "temperature": np.array(temperature),
        "particle_speeds": np.array(particle_speeds),
        "particle_velocities": np.array(particle_velocities),
        "particle_positions": np.array(particle_positions),
        "DT": DT,
        "T": T,
    }
    pickle.dump(data, open("data_langevin.pkl", "wb"))
