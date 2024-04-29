#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:51:50 2024

@author: brettnakao
"""

"""Generating and plotting trajectories"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

num_steps = 1000    # set the size of the array
x_step = 2 * (rng.random(num_steps) > .5) - 1
y_step = 2 * (rng.random(num_steps) > .5) - 1
x_position = np.cumsum(x_step)
y_position = np.cumsum(y_step)

# plot random walk trajectory
plt.figure()
plt.plot(x_position, y_position)
plt.xlabel('x-position')
plt.ylabel('y-position')
plt.axis('equal')
plt.title('2D Random Walk')
plt.show()

# plot four such trajectories
plt.figure()
plt.subplot(2,2,1)
x_step = 2 * (rng.random(num_steps) > .5) - 1
y_step = 2 * (rng.random(num_steps) > .5) - 1
x_position = np.cumsum(x_step)
y_position = np.cumsum(y_step)
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.plot(x_position, y_position)

plt.subplot(2,2,2)
x_step = 2 * (rng.random(num_steps) > .5) - 1
y_step = 2 * (rng.random(num_steps) > .5) - 1
x_position = np.cumsum(x_step)
y_position = np.cumsum(y_step)
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.plot(x_position, y_position)

plt.subplot(2,2,3)
x_step = 2 * (rng.random(num_steps) > .5) - 1
y_step = 2 * (rng.random(num_steps) > .5) - 1
x_position = np.cumsum(x_step)
y_position = np.cumsum(y_step)
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.plot(x_position, y_position)


plt.subplot(2,2,4)
x_step = 2 * (rng.random(num_steps) > .5) - 1
y_step = 2 * (rng.random(num_steps) > .5) - 1
x_position = np.cumsum(x_step)
y_position = np.cumsum(y_step)
plt.xlim(-60, 60)
plt.ylim(-60, 60)
plt.plot(x_position, y_position)

# organize and show plot
plt.tight_layout()
plt.show


"""Plot the dispacement distribution"""

import numpy as np
import matplotlib.pyplot as plt

# creating empty arrays
x_final = np.empty(1000)
y_final = np.empty(1000)
displacement = np.empty(1000)

for i in range(1000):
    rng = np.random.default_rng()
    num_steps = 1000
    x_step = 2 * (rng.random(num_steps) > .5) - 1
    y_step = 2 * (rng.random(num_steps) > .5) - 1
    x_position = np.cumsum(x_step)
    y_position = np.cumsum(y_step)
    x = x_position
    y = y_position
    x_final[i] = x[-1]
    y_final[i] = y[-1]
    displacement[i] = np.sqrt(x[-1]**2 + y[-1]**2)

# plot the end points
plt.figure()
plt.scatter(x_final, y_final)
plt.axis('square')
plt.title('End Point Plot')
plt.show()

# histogram of the displacement values
plt.figure()
plt.hist(displacement, bins=50)
plt.title('Displacement Histogram')
plt.show()

# histogram of displacement squared
plt.figure()
plt.hist(displacement**2, bins=50)
plt.title('Displacement Squared Histogram')
plt.show()

# test for exponential or power law relationships
plt.figure()
plt.hist(displacement**2, bins=50)
plt.semilogy()
plt.title('Semilogy Plot')
plt.show()

plt.figure()
plt.hist(displacement**2, bins=50)
plt.loglog()
plt.title('Loglog plot')
plt.show()

# find the average value of displacement**2
print(f"Average displacement^2 for 1000-step walk: {np.mean(displacement**2): .2f}")

# find the mean-square desplacement of a 4000-step walk
x_final = np.empty(1000)
y_final = np.empty(1000)
displacement = np.empty(1000)

for i in range(1000):
    rng = np.random.default_rng()
    num_steps = 4000
    x_step = 2 * (rng.random(num_steps) > .5) - 1
    y_step = 2 * (rng.random(num_steps) > .5) - 1
    x_position = np.cumsum(x_step)
    y_position = np.cumsum(y_step)
    x = x_position
    y = y_position
    x_final[i] = x[-1]
    y_final[i] = y[-1]
    displacement[i] = np.sqrt(x[-1]**2 + y[-1]**2)

print(f"Average displacement^2 for 4000-step walk: {np.mean(displacement**2): .2f}")