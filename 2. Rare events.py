#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:01:32 2024

@author: brettnakao
"""

"""The Poison distribution"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

# Plot factorial function
lam = 8

l = np.arange(0, 30, .2, dtype=float)
p = np.exp(-lam)*np.power(lam, l)/factorial(l)

plt.plot(l, p)
plt.title('Poisson distribution for 8% heads')
plt.show()

# Perform N coin flip trials
N = 1000000
rng = np.random.default_rng()
M = np.empty(N) # Number of heads for each trial
num_flips = 100

for i in range(N):
    heads = rng.random(num_flips) > .92
    num_heads = np.sum(heads)
    M[i] = num_heads

# Create histogram of the frequency of getting M heads in N trials
plt.figure()
plt.hist(M, bins=np.arange(-.5, 20.5, 1), edgecolor='black')
plt.xticks(np.arange(0,21,2))
plt.xlabel('Number of Heads')
plt.ylabel('Frequency')
plt.title('Frequency of the Number of Heads')

plt.plot(l, p*N, label='Poisson distribution model') # Add Poisson distribution
plt.show()


"""Waiting times"""

# Create an array defining the number of flips btw heads
num_flips = 1000000
rng = np.random.default_rng()
heads = rng.random(num_flips) > .92
heads_indices = np.nonzero(heads)
flips_btw_heads = np.diff(heads_indices).flatten()

# Create histogram of the frequency of wait time of getting heads
plt.figure()
plt.subplot(1,3,1)
plt.hist(flips_btw_heads, edgecolor='black')
plt.xlabel('Flips btw Heads')
plt.ylabel('Frequency')
plt.title('Frequency of Flips btw Heads')
# Create semilog of these frequencies
plt.subplot(1,3,2)
plt.hist(flips_btw_heads)
plt.semilogy()
plt.title('Semilogy Plot')
# Create loglog of these frequencies
plt.subplot(1,3,3)
plt.hist(flips_btw_heads)
plt.loglog()
plt.title('Loglog Plot')

# Organize and format plot
plt.tight_layout()
plt.show()

# Calculate average waiting time btw heads
print(f"Average waiting time btw heads: {np.mean(flips_btw_heads): .2f}")
print(f"Average expected waiting time btw heads: {100/8}")