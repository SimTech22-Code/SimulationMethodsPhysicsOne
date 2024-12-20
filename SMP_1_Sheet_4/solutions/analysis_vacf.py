import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open("/home/niklas/Desktop/Uni_Niklas/Semester_5/SimulationMethodsPhysicsOne/SMP_1_Sheet_4/solutions/data_langevin.pkl", "rb"))
velocities = data["particle_velocities"]
time_steps, N, dim = velocities.shape
DT = data["DT"]
d = dim
T = data["T"]

v_sq = np.sum(velocities**2, axis=2) 
mean_v_sq = np.mean(v_sq)
print("Mean v^2: ", mean_v_sq, " should be ~ 3T = ", 3*T)

vel_flat = velocities.reshape(time_steps, N*dim)

vel_mean = np.mean(vel_flat, axis=0)
vel_zero = vel_flat - vel_mean

f = np.fft.rfft(vel_zero, axis=0)
S = f.real**2 + f.imag**2

acf_full = np.fft.irfft(S, n=time_steps, axis=0) 

lags = np.arange(time_steps)
acf_normalized = acf_full / (time_steps - lags)[:, None]

vacf_full = np.mean(acf_normalized, axis=1)

scaling = mean_v_sq / vacf_full[100]
vacf_full *= scaling

max_lag = time_steps // 2
vacf = vacf_full[:max_lag]
time_lags = lags[:max_lag] * DT

plt.figure()
plt.plot(time_lags, vacf, label='VACF (FFT-based)', c = 'g')
plt.xlabel('Time')
plt.ylabel('VACF')
plt.title('Velocity Auto-Correlation Function (VACF)')
plt.legend()
plt.show()

# intergrate over this to get the D component
D = np.trapezoid(vacf, time_lags)
print(f"Diffusion coefficient D = {D:.4f}")
