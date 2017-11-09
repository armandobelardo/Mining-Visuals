#!/usr/bin/env python
from __future__ import print_function   # <- in case using Python 2.x
import numpy as np

from scipy.integrate import odeint
from matplotlib import pyplot as plt
from clusters import *

# TODO(iamabel): Finalize a realisitic margin of error.
MARGIN_ERR = .01
K_STEP = 1

def isSynchronized(thetas, alpha, K, xn, neighborhoods):
    dthetas = miyano(thetas,[],K,xn,neighborhoods)

    sigma_is = []
    for neighbor_i, neighborhood_i in enumerate(neighborhoods):
        d_ijs = []
        d_0 = alpha * np.linalg.norm(xn[neighbor_i, :])

        if not neighborhood_i: # nan avoidance with no neighbor issue
            d_ijs.append(0)
        else:
            for neighbor_j in neighborhood_i:
                d_ijs.append(np.linalg.norm(dthetas[neighbor_i:: N] - dthetas[neighbor_j:: N])/d_0)
        sigma_is.append(np.average(d_ijs))

    print(np.average(sigma_is))
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

    return neighbors

def diffs(thetas, neighbors, i):
    if not neighbors:
        return 0
    diff = []
    for neighbor in neighbors:
        diff.append(thetas[i] - thetas[neighbor])
    return diff

def miyano(vars, t, K, xn, neighbors):
    (N, D) = size(*xn.shape)
    dthetas = np.zeros((N,D))

    for i in range(N):
        for n in range(D):
            # Nth degree of freedom for datapoint i
            dthetas[i,n] = xn[i,n] + K * np.average(
                                            np.sin(
                                                diffs(vars[(int)(i/N):(int)((i/N)+N)],
                                                neighbors[i], i)))
    return dthetas.flatten('F')

def simulate(trange, thetas, K, xn, neighbors, plot = False):
    # Note that odeint expects a 1D array, so we flatten by column in order to
    # get all thetas of a degree in sequence.  It also outputs a 1D array, so we
    # flatten the output (traditionally) as well.
    results = odeint(miyano, thetas.flatten('F'), trange, args=(K, xn, neighbors))
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
    # xn = create2Clustering(4,2)           # Degrees of freedom
    xn = miyanoGrouping()
    alpha = .5                              # Make input, fixed for a data set

    N, D = size(*xn.shape)

    thetas_in = np.random.normal(0, 1, (N, D))


    # Start small, increment, then take last K that is "synchronized" under margin of error
    K = .4

    thetas = thetas_in[:]
    trange = np.linspace(0, 100, 2000)
    neighbors = getNeighbors(xn, alpha)

    # TODO(iamabel): Graph sigma over time in isSynchronized
    while True:
        r = simulate(trange, thetas, K, xn, neighbors)
        # Now restart the simulation from where you left off
        thetas = r[-1,:]
        r = simulate(trange, thetas, K, xn, neighbors)

        if isSynchronized(thetas, alpha, K, xn, neighbors):
            break

        # Increment alpha and reset simulation
        K *= K_STEP
        thetas = np.random.normal(0, 1, (N, D))

    print("Close the plot window to end the script")
    plt.show()

    data = np.column_stack([r, trange])
    np.savetxt('miyano.dat', data)
