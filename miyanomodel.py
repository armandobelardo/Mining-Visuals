#!/usr/bin/env python
from __future__ import print_function   # <- in case using Python 2.x
import numpy as np

from scipy.integrate import odeint
from matplotlib import pyplot as plt

# Fix to awkward numpy return
def size(height=0, width=0):
    return (height, width)

def neighbors(xn, alpha):
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

def diffs(phis, neighbors, i):
    diff = []
    for neighbor in neighbors:
        diff.append(phis[i] - phis[neighbor])
    return diff

def miyano(vars, t, K, xn, neighbors):
    (N, D) = size(*xn.shape)
    dphis = np.zeros(N*D)
    curr = 0

    for i in range(N):
        for n in range(D):
            # Nth degree of freedom for datapoint i
            dphis[curr] = xn[i,n] + K * np.average(np.sin(diffs(vars[(int)(i/N):(int)((i/N)+N)],
                                                                neighbors[i], i)))
            curr += 1
    return dphis

# TODO(iamabel): fix simulate
def simulate(trange, phis, K, xn, neighbors):
    # Note that odeint expects a 1D array, so we flatten by column. It also
    # outputs a 1D array, so we flatten the output (traditionally) as well
    results = odeint(miyano, phis.flatten('F'), trange, args=(K, xn, neighbors))
    (N, D) = size(*xn.shape)
    print("shape of results matrix:", results.shape)
    print("shape of trange matrix:", trange.shape)
    print(results)
    for i in range(N):
        plt.figure(i)
        for n in range(D):
            plt.plot(trange, results[:,D*n + i])
            
    return results

'''
int[][] -> void
Takes in degrees of freedom from data, finds the optimal conditions for
synchrony and plots the results.
'''
if __name__ == '__main__':
    # TODO(iamabel): Make these input
    phis = np.arange(8).reshape(4,2) # Initial phi measures
    xn = np.arange(8).reshape(4,2)   # Degrees of freedom

    # TODO(iamabel): Dynamically adjust K and alpha based on synchronization.
    K = 1
    alpha = 3

    trange = np.linspace(0, 100, 300)
    neighbors = neighbors(xn, alpha)

    r = simulate(trange, phis, K, xn, neighbors)
    # Now restart the simulation from where you left off
    phis = r[-1,:]
    r = simulate(trange, phis, K, xn, neighbors)

    print("Close the plot window to end the script")
    plt.show()

    data = np.column_stack([r, trange])
    np.savetxt('miyano.dat', data)
