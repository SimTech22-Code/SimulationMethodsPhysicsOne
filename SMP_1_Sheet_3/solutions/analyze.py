#!/usr/bin/env python3

import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

def running_average(O, M):
    pass

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

if __name__ == "__main__":


    path = '/home/nab/Niklas/non_work/SimulationMethodsPhysicsOne/test2.pkl'

    with open(path, 'rb') as fp:
        data = pickle.load(fp)

    plot_pressure(data)

