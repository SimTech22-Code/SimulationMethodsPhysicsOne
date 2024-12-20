# this is a Linear Congruential Generator (LCG) class
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class LCG:

    def __init__(self, m = 2 ** 32, a = 1103515245, c = 12345, seed = 5):
        self.m = m
        self.a = a
        self.c = c
        self.seed = seed


    def next(self):
        self.seed = (self.a * self.seed + self.c) % self.m
        return self.seed
    
    def next_float(self):
        return self.next() / self.m
    
    def random_walk(self, n_steps):
        walk = np.zeros(n_steps)

        for i in range(n_steps):
            walk[i] = self.next_float() - 0.5

        walk = np.cumsum(walk)
        
        return walk

    def box_muller_transform(self, mean= 0, std=1):
        u1 = self.next_float()
        u2 = self.next_float()

        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        z1 = mean + z1 * std
        z2 = mean + z2 * std

        return z1, z2
    
def maxwell_boltzmann_pdf(v, sigma):
    coeff = np.sqrt(2 / np.pi) / sigma**3
    return coeff * v**2 * np.exp(-v**2 / (2 * sigma**2))
    

if __name__ == '__main__':
    lcg = LCG()

    # Generate random numbers
    n_sample = 10 ** 6

    random_numbers = np.zeros(n_sample)
    for i in tqdm(range(n_sample)):
        random_numbers[i] = lcg.next_float()

    # Plot histogram and normalized histogram
    plt.title('Normalized histogram of random numbers')
    plt.hist(random_numbers, bins=100, density=True, range=(0, 1))
    plt.show()

    lcg = LCG()

    walk = lcg.random_walk(1000)

    plt.title('Random walk')
    plt.plot(walk)
    plt.show()

    # Generate random numbers
    number_of_walks = 200
    for i in range(number_of_walks):
        lcg = LCG(seed=time.time())
        walk = lcg.random_walk(1000)
        plt.plot(walk)

    plt.title('Random walks')
    plt.show()

    # get 10 ** 5 samples from the Box-Muller transform
    mean = 1
    std = 4
    n_samples = 10 ** 5
    samples = np.zeros(n_samples)
    for i in tqdm(range(int(len(samples) / 2))):
        z1, z2 = lcg.box_muller_transform(mean, std)
        samples[i] = z1
        samples[len(samples) - i - 1] = z2

    # Plot histogram and normalized histogram
    plt.title('Normalized histogram of samples')
    plt.hist(samples, bins=80, density=True)
    # plot the expected normal distribution
    x = np.linspace(-10, 10, 100)
    y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    plt.plot(x, y, 'r')
    plt.show()

    """
    Generate N = 105 random Gaussian velocity vectors v = (vx , vy , vz )
    which have elements vx , vy and vz taken from a Gaussian distribution
    with mean µ = 0 and standard-deviation of σ = 1.0
    • Plot the distribution of the speeds v = |v| obtained from your ran-
    dom vectors and compare with the analytical three-dimensional Maxwell-
    Boltzmann distribution.
    """
    n_samples = 10 ** 5 * 3
    samples = np.zeros(n_samples)

    for i in tqdm(range(int(n_samples / 2))):
        z1, z2 = lcg.box_muller_transform(0, 1)
        samples[i] = z1
        samples[len(samples) - i - 1] = z2

    # calculate the speed
    # take 3 samples at a time
    speeds = np.zeros(n_samples // 3)

    for i in tqdm(range(0, n_samples, 3)):
        speeds[i] = np.sqrt(samples[i] ** 2 + samples[i + 1] ** 2 + samples[i + 2] ** 2)

    # Plot histogram and normalized histogram
    plt.title('Normalized histogram of speeds')
    plt.hist(speeds, bins=80, density=True)
    # plot the analytical Maxwell-Boltzmann distribution 3D Boltzman distribution
    x = np.linspace(0, 10, 100)
    y = maxwell_boltzmann_pdf(x, 1)
    plt.plot(x, y, 'r')
    plt.show()
    
    






