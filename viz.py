import networkx as nx
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Rectangle

# To fix awkward numpy return.
def size(height=0, width=0):
    return (height, width)

def edgelist(neighbors):
    edges = []
    for neighborhood_i, neighborhood in enumerate(neighbors):
        for neighbor in neighborhood:
            edges.append((neighborhood_i, neighbor))

    return edges

def endplot(results, trange, neighbors, D):
    # TODO(iamabel): There needs to be a better way to do color.
    colors = "bgrcmykw"
    done_neighbors = []

    # Make a legend
    handles = [Rectangle((0,0),1,1, color=colors[n%len(colors)]) for n in range(D)]
    labels = ["Degree " + str(n) for n in range(D)]

    for neighborhood_i, neighborhood in enumerate(neighbors):
        if neighborhood not in done_neighbors:
            fig = plt.figure(neighborhood_i)
            plt.clf()
            for i in neighborhood:
                for n in range(D):
                    # Degree of freedom for i for all times
                    plt.plot(trange, results[:,n + i*D], colors[n%len(colors)])
            done_neighbors.append(neighborhood)
            fig.suptitle("Group: "+','.join(map(str, neighborhood)), fontsize=14, fontweight='bold')
            plt.legend(handles, labels)

def sigmaovertime(alpha, xn, neighborhoods, trange):
    N, D = size(*xn.shape)
    sigma_is = []
    sigmas = []

    # TODO(iamabel): r needs to be theta dot but is theta
    for dthetas in r:
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
        sigmas.append(np.average(sigma_is))
    plt.plot(trange, sigmas)
    return False

def flagplot(neighborhoods):
    G=nx.Graph(edgelist(neighborhoods))

    directory = os.fsencode("./assets")
    assets = []
    for file in os.listdir(directory):
        assets.append("./assets/" + os.fsdecode(file))
    assets.sort()

    for n in G:
        # Images from https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags
        G.node[n]['image']=mpimg.imread(assets[n])

    pos=nx.spring_layout(G)

    fig=plt.figure(figsize=(7,7))
    ax=plt.gca()
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    flagsize=0.1 # this is the image size
    f2=flagsize/2.0
    for n in G:
       xx,yy=trans(pos[n]) # figure coordinates
       xa,ya=trans2((xx,yy)) # axes coordinates
       a = plt.axes([xa-f2,ya-f2, flagsize, flagsize])
       a.set_aspect('equal')
       a.imshow(G.node[n]['image'])
       a.set_xticks([])
       a.set_yticks([])
    ax.axis('off')

    plt.show()

def netplot(neighborhoods):
    fig = plt.figure(77)
    G = nx.Graph(edgelist(neighborhoods))
    nx.draw_spring(G)
    plt.show()
