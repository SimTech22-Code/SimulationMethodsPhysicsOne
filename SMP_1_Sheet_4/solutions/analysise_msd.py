import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load data
data = pickle.load(open("/home/niklas/Desktop/Uni_Niklas/Semester_5/SimulationMethodsPhysicsOne/SMP_1_Sheet_4/solutions/data_langevin.pkl", "rb"))
positions = data["particle_positions"] 
DT = data["DT"]
N = positions.shape[1]
time_steps = positions.shape[0]

d = 3
r = 10000

max_lag = time_steps // 2  
msd = []
msd_err = []
lags = np.arange(1, max_lag)  
# use log data
for lag in tqdm(list(np.log10(lags).astype(int))):

    displacements = positions[lag:] - positions[:-lag]  
    sq_displacements = np.sum(displacements**2, axis=2)  
    
    sq_disp_flat = sq_displacements.flatten()
    
    mean_sq_disp = np.mean(sq_disp_flat)
    std_sq_disp = np.std(sq_disp_flat, ddof=1)
    error = std_sq_disp / np.sqrt(len(sq_disp_flat))
    
    msd.append(mean_sq_disp)
    msd_err.append(error)

msd = np.array(msd)
msd_err = np.array(msd_err)

time_lags = lags * DT
time_lags = np.array(list(range(1, max_lag, r))) * DT
# plot in log log scale
plt.figure(figsize=(8,6))

plt.loglog(time_lags, msd, 'o-', label='MSD')

plt.xlabel('Time lag')
plt.ylabel('MSD')
plt.title('Mean Squared Displacement (Langevin)')
plt.show()
