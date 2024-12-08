#!/usr/bin/env python3

import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

def running_average(O, M):
    # otherwise known as the simple moving average
    # O is the original data
    # M is the window size
    # the output is the running average data array

    N = len(O)
    running_average = np.zeros(N)
    for i in range(N):
        running_average[i] = np.mean(O[max(0, i - M):min(N, i + M)])
    return running_average


def plot_pressure(data):
    # this function should plot the pressure as a function of time
    # the pressure was already computed and stored in the data dictionary
    # the pressure is stored in data['pressures']
    # the pressure is stored as a list of floats
    # the time is stored in data['time']
    # the time is stored as a list of floats
    # the plot should have the title "Pressure vs Time"

    time = data['time']
    pressures = data['pressures']

    plt.plot(time, pressures)
    plt.title("Pressure vs Time")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.show()

def plot_energy(data):
    # this function should plot the energy as a function of time
    # the energy was already computed and stored in the data dictionary
    # the energy is stored in data['energies']
    # the energy is stored as a list of floats
    # the time is stored in data['time']
    # the time is stored as a list of floats
    # the plot should have the title "Energy vs Time"

    time = data['time']
    energies = data['energies']

    plt.plot(time, energies)
    plt.title("Energy vs Time")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.show()

if __name__ == "__main__":


    path = '/home/niklas/Desktop/Uni_Niklas/Semester_5/SimulationMethodsPhysicsOne/test_temp_3.pkl'

    with open(path, 'rb') as fp:
        data = pickle.load(fp)


    # plot_pressure(data)
    # plot_energy(data)

    # plot the pressure the energy the temperatur, all of them with running average of windowsizes 10 and 100

    time = data['time']
    pressures = data['pressures']
    energies = data['energies']
    temperatures = data['temperatures']

    plt.plot(time, pressures, label='Pressure')
    plt.plot(time, running_average(pressures, 10), label='Pressure Running Average 10')
    plt.plot(time, running_average(pressures, 100), label='Pressure Running Average 100')

    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.legend()
    plt.show()

    plt.plot(time, temperatures, label='Temperature')
    plt.plot(time, running_average(temperatures, 10), label='Temperature Running Average 10')
    plt.plot(time, running_average(temperatures, 100), label='Temperature Running Average 100')

    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()
