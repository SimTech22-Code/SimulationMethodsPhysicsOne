import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pickle.load(open("/home/niklas/Desktop/Uni_Niklas/Semester_5/SimulationMethodsPhysicsOne/SMP_1_Sheet_4/solutions/data_anderson.pkl", "rb"))
# data = pickle.load(open("/home/niklas/Desktop/Uni_Niklas/Semester_5/SimulationMethodsPhysicsOne/SMP_1_Sheet_4/solutions/data_langevin.pkl", "rb"))

temperature = data["temperature"]         # shape: (n_steps,)
particle_speeds = data["particle_speeds"] # shape: (n_steps, N)
DT = data["DT"]
T = data["T"]
N = particle_speeds.shape[1]
n_steps = particle_speeds.shape[0]

# 1. Plot instantaneous temperature vs. time
time = np.arange(n_steps)*DT

plt.figure(figsize=(8,6))
plt.plot(time, temperature, label='Instantaneous Temperature')
plt.axhline(T, color='r', linestyle='--', label='Target Temperature (T={})'.format(T))
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Instantaneous Temperature vs. Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("temperature_vs_time.png", dpi=300)
plt.show()

# 2. Plot the distribution of the particle speeds
# We can take the speeds from the final time step or average over the last part of the simulation.
# For a stable distribution, let’s consider the last half of the data and stack them together:
speeds_final = particle_speeds[n_steps//2:].flatten()  # speeds from the last half of the simulation

# Compute histogram of the speeds
counts, bin_edges = np.histogram(speeds_final, bins=50, density=True)
bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

# Maxwell-Boltzmann distribution for speeds in 3D:
# For mass m=1 and k_B=1 and temperature T:
# P(|v|) = ( (1/(2*pi*T))^(3/2) ) * 4*pi * v^2 * exp(-v^2/(2T))
m = 1.0
k_B = 1.0
mb_prefactor = (m/(2*np.pi*k_B*T))**(3/2)
mb_distribution = mb_prefactor * 4*np.pi*(bin_centers**2)*np.exp(-bin_centers**2/(2*T))

plt.figure(figsize=(8,6))
plt.hist(speeds_final, bins=50, density=True, alpha=0.6, label='Simulated Speed Distribution')
plt.plot(bin_centers, mb_distribution, 'r--', lw=2, label='Maxwell-Boltzmann (3D)')

plt.xlabel('Speed |v|')
plt.ylabel('Probability Density')
plt.title('Speed Distribution vs. Maxwell-Boltzmann')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("speed_distribution.png", dpi=300)
plt.show()
