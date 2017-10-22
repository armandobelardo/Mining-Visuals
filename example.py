#!/usr/env python
from __future__ import print_function   # <- in case using Python 2.x
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

def jjneuron(vars,t,gamma,lam,I_in,I_bias):
    phi_p, phi_c, v_p, v_c = vars
    dphi_c = v_c
    dphi_p = v_p
    dv_p = -gamma*v_p - np.sin(phi_p) - lam*(phi_c+phi_p) +0.5*(I_bias+I_in)
    dv_c = -gamma*v_c - np.sin(phi_c) - lam*(phi_c+phi_p) +0.5*(I_bias-I_in)
    return [dphi_p, dphi_c, dv_p, dv_c]

def simulate(trange, vinit, gamma=2.0, lam=0.1, I_in=1.9, I_bias=2.0):
    results = odeint(jjneuron, vinit, trange, args=(gamma, lam, I_in, I_bias))
    print("shape of results matrix:", results.shape)
    print("shape of trange matrix:", trange.shape)
    plt.figure(1)
    plt.plot(trange, results[:, 2], 'b', trange, results[:, 3], 'g')
    plt.figure(2)
    plt.plot(trange, results[:, 0], 'b', trange, results[:, 1], 'g')
    return results

if __name__ == '__main__':
    vinit = [5.122, -5.122, 0, 0]
    trange = np.linspace(0, 100, 300)
    r = simulate(trange, vinit)
    # Now restart the simulation from where you left off
    vinit = r[-1,:]
    r = simulate(trange,vinit)
    # if you are using interactive mode then you don't have to close
    # the window to end the script -- the windows stay open and control
    # returns to the prompt.
    print("Close the plot window to end the script")
    plt.show()

    data = np.column_stack([r, trange])
    np.savetxt('jj.dat', data)
