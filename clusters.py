import numpy as np

def create2Clustering(N, D):
    cluster = np.empty([N,D])
    for i in range(N):
        for j in range(D):
            if i % 2 == 0:
                cluster[i,j] = 1 + .05*np.random.rand()
            else:
                cluster[i,j] = -1 - .05*np.random.rand()

    print("Returning cluster: ", cluster)
    return cluster
