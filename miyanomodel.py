#!/usr/bin/env python
from __future__ import print_function   # <- in case using Python 2.x
import numpy as np

from scipy.integrate import odeint
from matplotlib import pyplot as plt
from clusters import create2Clustering

# TODO(iamabel): Finalize a realisitic margin of error.
MARGIN_ERR = .1
ALPHA_STEP = .001

def isSynchronized(neighborhoods, phis, D):
    sigma_is = []
    for neighbor_i, neighborhood_i in enumerate(neighborhoods):
        d_ijs = []
        if not neighborhood_i: # nan avoidance with no neighbor issue
            d_ijs.append(0)
        else:
            for neighbor_j in neighborhood_i:
                d_ijs.append(np.linalg.norm(
                             np.remainder(np.subtract(phis[neighbor_i: -1: N], phis[neighbor_j: -1: N]),
                                          2 * np.pi)))
        sigma_is.append(np.average(d_ijs))

    return np.average(sigma_is) < MARGIN_ERR

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
            if (i != j):
                # Get the norm of the vector difference
                d_ij = np.linalg.norm(np.subtract(xn[i, :], xn[j, :]))
                if (d_ij <= d_0):
                    neighborhood.append(j)
        neighbors.append(neighborhood)
    print("Found neighborhoods:", neighbors)
    return neighbors

def diffs(phis, neighbors, i):
    if not neighbors:
        return 0
    diff = []
    for neighbor in neighbors:
        diff.append(phis[i] - phis[neighbor])
    return diff

def miyano(vars, t, K, xn, neighbors):
    (N, D) = size(*xn.shape)
    dphis = np.zeros((N,D))

    for i in range(N):
        for n in range(D):
            # Nth degree of freedom for datapoint i
            dphis[i,n] = xn[i,n] + K * np.average(
                                            np.sin(
                                                diffs(vars[(int)(i/N):(int)((i/N)+N)],
                                                neighbors[i], i)))
    return dphis.flatten('F')

def simulate(trange, phis, K, xn, neighbors, plot = False):
    # Note that odeint expects a 1D array, so we flatten by column in order to
    # get all phis of a degree in sequence.  It also outputs a 1D array, so we
    # flatten the output (traditionally) as well.
    results = odeint(miyano, phis.flatten('F'), trange, args=(K, xn, neighbors))
    (N, D) = size(*xn.shape)

    if plot:
        for neighborhood_i, neighborhood in enumerate(neighbors):
            plt.figure(neighborhood_i)
            plt.clf()
            # Our neighborhoods only contain neighbors, we need to add the node the
            # neighborhood is around, neighborhood_i.
            neighborhood += [neighborhood_i]
            for i in neighborhood:
                for n in range(D):
                    # Degree of freedom for i for all times
                    plt.plot(trange, results[:,D*n + i])
    return results

'''
(int[][]) -> void
Takes in degrees of freedom from data, finds the optimal conditions for
synchrony and plots the results.
'''
if __name__ == '__main__':
    # TODO(iamabel): Make these input
    xn = create2Clustering(4,2)           # Degrees of freedom
    K = 1                               # Input, fixed for a data set

    N, D = size(*xn.shape)

    phis_in = np.random.randint(2 * np.pi, size = (N, D))

    # Start small, increment, then take last alpha that is "synchronized" under margin of error
    alpha = 0

    phis = phis_in[:]
    trange = np.linspace(0, 100, 300)

    # TODO(iamabel): Graph sigma over time in isSynchronized
    while True:
        neighbors = getNeighbors(xn, alpha)

        r = simulate(trange, phis, K, xn, neighbors)
        # Now restart the simulation from where you left off
        phis = r[-1,:]
        r = simulate(trange, phis, K, xn, neighbors)

        if not isSynchronized(neighbors, phis, D):
            # neighbors = getNeighbors(xn, alpha - ALPHA_STEP)
            # # Resimulate to show last working alpha
            # r = simulate(trange, phis_in[:], K, xn, neighbors)
            # # Now restart the simulation from where you left off
            # phis = r[-1,:]
            r = simulate(trange, phis, K, xn, neighbors, True)
            break

        # Increment alpha and reset simulation
        alpha += ALPHA_STEP
        phis = phis_in[:]

    print("Close the plot window to end the script")
    plt.show()

    data = np.column_stack([r, trange])
    np.savetxt('miyano.dat', data)
