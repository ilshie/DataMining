# CS324 Data Mining (Spring 2017)
# Final Project
# Authors: Il Shan Ng, Yuan Shen Li
# Visualization program for SOM

import numpy as np
import pickle
import scipy.spatial.distance as spd
import sys

def constructSquareTopoMap(J,K,model):
    """
    Constructs a square topograpical map from SOM
    Returns J*K map with inverse distance values
    """
    topomap = np.array([[0. for i in range(K)] for j in range(J)])

    for j in range(J):
        for k in range(K):
            m = np.array([model[j*K + k]])
            up = (j-1) * K + k
            down = (j+1) * K + k
            left = j * K + (k - 1)
            right = j * K + (k + 1)
            
            if j == 0 and k == 0:
                neighbors = np.array([model[right],
                                      model[down]])
            elif j == 0 and 0 < k < K-1:
                neighbors = np.array([model[down],
                                      model[right],
                                      model[left]])
            elif j == 0 and k == K-1:
                neighbors = np.array([model[left],
                                      model[down]])
            elif j == J-1 and k == 0:
                neighbors = np.array([model[right],
                                      model[up]])
            elif j == J-1 and 0 < k < K-1:
                neighbors = np.array([model[up],
                                      model[right],
                                      model[left]])
            elif j == J-1 and k == K-1:
                neighbors = np.array([model[left],
                                      model[up]])
            elif k == 0 and 0 < j < J-1:
                neighbors = np.array([model[right],
                                      model[up],
                                      model[down]])
            elif k == K and 0 < j < J-1:
                neighbors = np.array([model[left],
                                      model[up],
                                      model[down]])
            else:
                neighbors = np.array([model[up],
                                      model[down],
                                      model[left],
                                      model[right]])
                
            # calculate distance to neighbors
            distMatrix = spd.cdist(m,neighbors,"euclidean")

            dist = np.mean(distMatrix)

            topomap[j][k] = dist

    return topomap

def main():
    model = pickle.load(open("digitmodel.p","rb"))
    #model_history = pickle.load(open("testmodelhistory.p","rb"))

    J = 40
    K = 40

    np.savetxt("testmodel.txt",model)
    #np.savetxt("testmodelhistory.txt",model_history)

    topomap = constructSquareTopoMap(J,K,model)

    #write to file
    f = open("testtopomap.txt","w")
    for row in topomap:
        for entry in row:
            f.write(str(entry))
            f.write('\t')
        f.write('\n')
        
    f.close()


main()
