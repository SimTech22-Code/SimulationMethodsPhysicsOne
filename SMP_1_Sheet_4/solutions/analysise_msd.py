import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load data
data = pickle.load(open("/home/niklas/Desktop/Uni_Niklas/Semester_5/SimulationMethodsPhysicsOne/SMP_1_Sheet_4/solutions/data_langevin.pkl", "rb"))
positions = data["particle_positions"]  # shape: (time_steps, N, 3)
DT = data["DT"]
N = positions.shape[1]
time_steps = positions.shape[0]

# Define the dimensionality d (3 for 3D)
d = 3

# Compute MSD for a range of delta t values
max_lag = time_steps // 2  # for example, only go up to half the trajectory length
msd = []
msd_err = []
lags = np.arange(1, max_lag)  # lag times in units of simulation steps

for lag in tqdm(lags):
    displacements = positions[lag:] - positions[:-lag]  # shape: (time_steps - lag, N, 3)
    sq_displacements = np.sum(displacements**2, axis=2)  # sum over x,y,z => shape: (time_steps - lag, N)
    
    # Flatten over all particles and all valid time segments
    sq_disp_flat = sq_displacements.flatten()
    
    # Mean and standard error
    mean_sq_disp = np.mean(sq_disp_flat)
    std_sq_disp = np.std(sq_disp_flat, ddof=1)
    error = std_sq_disp / np.sqrt(len(sq_disp_flat))
    
    msd.append(mean_sq_disp)
    msd_err.append(error)

msd = np.array(msd)
msd_err = np.array(msd_err)

# Convert lag steps to actual time: lag_time = lag * DT
time_lags = lags * DT

# Plot MSD with error bars
plt.errorbar(time_lags, msd, yerr=msd_err, fmt='o', capsize=3)
plt.xlabel('Time lag')
plt.ylabel('MSD')
plt.title('Mean Squared Displacement (Langevin)')
plt.show()
