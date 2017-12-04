#!/usr/bin/env python
from __future__ import print_function   # <- in case using Python 2.x
import networkx as nx
import numpy as np
import matplotlib.animation as animation

from datagen import *
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from viz import *

MARGIN_ERR = 3e-10
ALPHA_STEP = .1

END_TRANGE = 100
START_TRANGE = 0

def isSynchronized(thetas_b, alpha, K, xn, neighborhoods):
    dthetas = miyano(thetas_b, [], K, xn, neighborhoods)

    sigma_is = []
    for neighbor_i, neighborhood_i in enumerate(neighborhoods):
        d_ijs = []
        d_0 = alpha * np.linalg.norm(xn[neighbor_i, :])

        if not neighborhood_i: # nan avoidance with no neighbor issue
            d_ijs.append(0)
        else:
            for neighbor_j in neighborhood_i:
                d_ijs.append(np.linalg.norm(
                                 dthetas[neighbor_i*D:(neighbor_i*D)+D] -
                                 dthetas[neighbor_j*D:(neighbor_j*D)+D])
                             /d_0)
        sigma_is.append(np.average(d_ijs))

    print("sigma: " + str(np.average(sigma_is)) +  " for alpha: " + str(alpha))
    return np.average(sigma_is)

# To fix awkward numpy return.
def size(height=0, width=0):
    return (height, width)

def getNeighbors(xn, alpha):
    neighbors = []
    N = size(*xn.shape)[0]
    for i in range(N):
        neighborhood = []
        d_0 = alpha * np.linalg.norm(xn[i, :])
        for j in range(N): # potential neighbor check
            # Get the norm of the vector difference
            d_ij = np.linalg.norm(np.subtract(xn[i, :], xn[j, :]))
            if (d_ij <= d_0): # includes self
                neighborhood.append(j)
        neighbors.append(neighborhood)

    return neighbors

# Returns the differences of thetas (for a single degree of freedom) within a
# single neighborhood.
def diffs(thetas, neighbors, i):
    if not neighbors:
        return 0
    diff = []
    for neighbor in neighbors:
        diff.append(thetas[neighbor] - thetas[i])
    return diff

def miyano(vars, t, K, xn, neighbors):
    (N, D) = size(*xn.shape)
    dthetas = np.zeros((N,D))

    for i in range(N):
        for n in range(D):
            # Using the nth degree of freedom for datapoint i.
            dthetas[i,n] = xn[i,n] + K * np.average(
                                            np.sin(
                                                diffs(vars[n::D],
                                                neighbors[i], i)))
    return dthetas.flatten()

def simulate(trange, thetas, K, xn, neighbors):
    # Note that odeint expects a 1D array, so we flatten in order to get all
    # degrees of freedom for a specific datapoint in sequence. Odeint also
    # outputs a 1D array, so we flatten the output (traditionally) as well.
    return odeint(miyano, thetas, trange, args=(K, xn, neighbors))

def convert_to_hex(rgba_color) :
    red = int(rgba_color[0]*255)
    green = int(rgba_color[1]*255)
    blue = int(rgba_color[2]*255)
    return '#%02x%02x%02x' % (red, green, blue)

'''
(int[][]) -> void
Takes in degrees of freedom from data, finds the optimal conditions for
synchrony and plots the results.
'''
if __name__ == '__main__':
    # TODO(iamabel): Make these input
    # xn, alpha = flagData()           # Degrees of freedom
    xn, K = flagdata()
    sigmas = []

    N, D = size(*xn.shape)

    # Start small, increment, then take last K that is "synchronized" under margin of error
    alpha = .5

    thetas_b = np.random.normal(0, 1, (N*D))
    trange = np.linspace(START_TRANGE, END_TRANGE, 2000)

    neighbors = getNeighbors(xn, alpha)

    # TODO(iamabel): Graph sigma over time in isSynchronized
    while True:
        r = simulate(trange, thetas_b, K, xn, neighbors)
        # Now restart the simulation from where you left off
        thetas_b = r[-1,:]

        sigmas.append(isSynchronized(thetas_b, alpha, K, xn, neighbors))

        if sigmas[-1] > MARGIN_ERR:
            # alpha -= ALPHA_STEP
            neighbors = getNeighbors(xn, alpha)
            break

        # Increment K and reset simulation
        alpha += ALPHA_STEP
        neighbors = getNeighbors(xn, alpha)
        thetas_b = np.random.normal(0, 1, (N*D))

    r = endplot(r, trange, neighbors, D)
    flagplot(neighbors)

    print("Close the plot window to end the script")
    plt.show()
