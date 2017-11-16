#!/usr/bin/env python
from __future__ import print_function   # <- in case using Python 2.x
import numpy as np

from scipy.integrate import odeint
from matplotlib import pyplot as plt
from clusters import *

# TODO(iamabel): Finalize a realisitic margin of error.
MARGIN_ERR = .1
K_STEP = 1

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
                d_ijs.append(np.linalg.norm(dthetas[neighbor_i:neighbor_i+D] - dthetas[neighbor_j:neighbor_j+D])/d_0)
        sigma_is.append(np.average(d_ijs))

    print(neighborhoods)
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

def endplot(results, trange, neighbors, D):
    # TODO(iamabel): Make a random array of colors of size D so that all
    # degrees of freedom are the same color in and across graphs
    colors = "bgrcmykw"

    done_neighbors = []

    for neighborhood_i, neighborhood in enumerate(neighbors):
        if neighborhood not in done_neighbors:
            plt.figure(neighborhood_i)
            plt.clf()
            for i in neighborhood:
                for n in range(D):
                    # Degree of freedom for i for all times
                    plt.plot(trange, results[:,n + i*D], colors[n%len(colors)])
            done_neighbors.append(neighborhood)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
'''
(int[][]) -> void
Takes in degrees of freedom from data, finds the optimal conditions for
synchrony and plots the results.
'''
if __name__ == '__main__':
    # TODO(iamabel): Make these input
    xn = miyanoGrouping()           # Degrees of freedom
    alpha = .5                      # Make input, fixed for a data set

    N, D = size(*xn.shape)

    # Start small, increment, then take last K that is "synchronized" under margin of error
    K = .4

    thetas_b = np.random.normal(0, 1, (N*D))
    trange = np.linspace(START_TRANGE, END_TRANGE, 2000)

    neighbors = getNeighbors(xn, alpha)

    # TODO(iamabel): Graph sigma over time in isSynchronized
    while True:
        r = simulate(trange, thetas_b, K, xn, neighbors)
        # Now restart the simulation from where you left off
        thetas_b = r[-1,:]

        if not isSynchronized(thetas_b, alpha, K, xn, neighbors):
            break

        # Increment K and reset simulation
        K *= K_STEP
        thetas_b = np.random.normal(0, 1, (N*D))

    r = endplot(r, trange, neighbors, D)

    print("Close the plot window to end the script")
    plt.show()

    data = np.column_stack([r, trange])
    np.savetxt('miyano.dat', data)
