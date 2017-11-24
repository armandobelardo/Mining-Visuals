import numpy as np

def create2Clustering(N, D):
    cluster = np.empty([N,D])
    for i in range(N):
        for j in range(D):
            if i % 2 == 0:
                cluster[i,j] = 1 + .05*np.random.rand()
            else:
                cluster[i,j] = -1 - .05*np.random.rand()

    return cluster

def miyanoGrouping():
    grouping = np.random.normal(0, .1, (15,3))
    grouping[0:5, 0] += 1
    grouping[5:10, 1] += 1
    grouping[10:, 2] += 1

    return grouping
