#!/usr/bin/env python3

import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

import pathlib
file_dir = pathlib.Path(__file__).parent
plots_path = pathlib.Path(__file__).resolve().parent.parent.joinpath('plots')
plots_path.mkdir(parents=True, exist_ok=True)

from matplotlib.colors import hsv_to_rgb

DT = 0.01
SAMPLING_STRIDE = 3

parser = argparse.ArgumentParser()
parser.add_argument('file',
                    type=lambda cpt: str((file_dir / cpt) if not pathlib.Path(cpt).is_absolute() else pathlib.Path(cpt)),
                    help="Path to pickle file.")
parser.add_argument(
    '--teq',
    type=float,
    help='Time after which to calculate equilibrium.')
args = parser.parse_args()

with open(args.file, 'rb') as fp:
    data = pickle.load(fp)

def running_average(O, M): # compute running average of obervable O with different window sizes M
    O_ra = np.nan * np.ones_like(O)
    for i in range(M, len(O)-M):
        O_ra[i] = np.mean(O[i-M:i+M+1])
    return O_ra

energies = np.array(data['energies'])
t = np.linspace(0, DT*SAMPLING_STRIDE*np.shape(energies)[0], np.shape(energies)[0])
fig1 = plt.figure(figsize=(8, 2.5))
ax1 = fig1.add_subplot(111)
plt.plot(t, energies[:,0], label=r"$E_\mathrm{pot}$", c=hsv_to_rgb((.0, .8, .1)))
plt.plot(t, running_average(energies[:,0], 10), label=r"$E_\mathrm{pot}$, $M = 10$", c=hsv_to_rgb((.0, .8, .5)))
plt.plot(t, running_average(energies[:,0], 100), label=r"$E_\mathrm{pot}$, $M = 100$", c=hsv_to_rgb((.0, .8, .9)))

plt.plot(t, energies[:,1], label=r"$E_\mathrm{kin}$", c=hsv_to_rgb((.3, .8, .1)))
plt.plot(t, running_average(energies[:,1], 10), label=r"$E_\mathrm{kin}$, $M = 10$", c=hsv_to_rgb((.3, .8, .5)))
plt.plot(t, running_average(energies[:,1], 100), label=r"$E_\mathrm{kin}$, $M = 100$", c=hsv_to_rgb((.3, .8, .9)))

plt.plot(t, energies[:,0]+energies[:,1], label=r"$E_\mathrm{tot}$", c=hsv_to_rgb((.6, .8, .1)))
plt.plot(t, running_average(energies[:,0]+energies[:,1], 10), label=r"$E_\mathrm{tot}$, $M = 10$", c=hsv_to_rgb((.6, .8, .5)))
plt.plot(t, running_average(energies[:,0]+energies[:,1], 100), label=r"$E_\mathrm{tot}$, $M = 100$", c=hsv_to_rgb((.6, .8, .9)))

plt.xlabel(r"Time $t$")
plt.ylabel(r"Energy $E$")
plt.legend(bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()


temperatures = np.array(data['temperatures'])
fig2 = plt.figure(figsize=(4, 2.5))
ax2 = fig2.add_subplot(111)
plt.plot(t, temperatures, label=r"$T_\mathrm{m}$", c=hsv_to_rgb((.3, .8, .1)))
plt.plot(t, running_average(temperatures, 10), label=r"$T_\mathrm{m}$, $M = 10$", c=hsv_to_rgb((.3, .8, .5)))
plt.plot(t, running_average(temperatures, 100), label=r"$T_\mathrm{m}$, $M = 100$", c=hsv_to_rgb((.3, .8, .9)))
plt.xlabel(r"Time $t$")
plt.ylabel(r"Temperature $T_\mathrm{m}$ in $k_\mathrm{B}^{-1}$")
plt.legend()
plt.tight_layout()


pressures = np.array(data['pressures'])
fig3 = plt.figure(figsize=(4, 2.5))
ax3 = fig3.add_subplot(111)
plt.plot(t, pressures, label=r"$P$", c=hsv_to_rgb((.3, .8, .1)))
plt.plot(t, running_average(pressures, 10), label=r"$P$, $M = 10$", c=hsv_to_rgb((.3, .8, .5)))
plt.plot(t, running_average(pressures, 100), label=r"$P$, $M = 100$", c=hsv_to_rgb((.3, .8, .9)))
plt.xlabel(r"Time $t$")
plt.ylabel(r"Pressure $P$")
plt.legend()
plt.tight_layout()


if args.teq:
    t_eq = args.teq/(DT*SAMPLING_STRIDE)
    if args.teq > np.max(t):
        print("Simulation shorter than given equilibrium time.")
    else:
        rdfs = np.array(data['rdfs'])
        rdf_mean = np.mean(rdfs[round(t_eq):,0,:], axis=0)
        r = rdfs[0,1,:]
        plt.figure(figsize=(8, 2.5))
        plt.plot(r, rdf_mean, label=r"$g$", c=hsv_to_rgb((.3, .8, .1)), marker='+')
        plt.xlabel(r"Distance $r$")
        plt.ylabel(r"RDF $g$")
        plt.tight_layout()
        plt.savefig(f"{str(plots_path)}/RDF_"+str(pathlib.Path(args.file).name[:-4])+".pdf", format="pdf")

        for ax in [ax1, ax2, ax3]:
            ax.axvline(x = args.teq, ymin=ax1.get_ylim()[0], ymax=ax1.get_ylim()[1], linestyle=':', color='k')

        print("Mean values after t_eq = "+str(args.teq)+
              "\n   Total energy: "+str(np.mean(energies[round(t_eq):,0]+energies[round(t_eq):,1]))+
              "\n   Potential energy: "+str(np.mean(energies[round(t_eq):,0]))+
              "\n   Kinetic energy: "+str(np.mean(energies[round(t_eq):,1]))+
              "\n   Temperature: "+str(np.mean(temperatures[round(t_eq):]))+
              "\n   Pressure: "+str(np.mean(pressures[round(t_eq):])))
  
fig1.savefig(f"{str(plots_path)}/Energies_"+str(pathlib.Path(args.file).name[:-4])+".pdf", format="pdf")
fig2.savefig(f"{str(plots_path)}/Temperature_"+str(pathlib.Path(args.file).name[:-4])+".pdf", format="pdf")
fig3.savefig(f"{str(plots_path)}/Pressure_"+str(pathlib.Path(args.file).name[:-4])+".pdf", format="pdf")

plt.show()