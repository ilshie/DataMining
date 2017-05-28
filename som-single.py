# som.py
# Implements a self-organizing map to cluster countries based on world development
# indicators.
#---------------------------------------------------------------------------------------
# Yuan Shen Li and Il Shan Ng
# May 20, 2017

import numpy as np
import scipy.spatial.distance as sp
import argparse
import collections as cl
import pickle

class SOM_Single():
    """
    Self-organizing map, single data point version
    """
    def __init__(self, J, K, inputs, init, dist_func, neighborhood_func, iterations):
        """
	Store class variables.
	"""
        self.inputs = inputs
        self.J = J
        self.K = K
        self.init = init
        self.dist_func = dist_func
        self.neighborhood_func = neighborhood_func
        self.M = []
        self.M_history = []
        self.max_iterations = iterations

        # learning rate
        self.learningRate_init = 1
        self.learningRate_decayfactor = 1000
        self.learningRate = self.learningRate_init

        # if Gaussian neighborhood function
        self.sigma_init = max(J,K)
        self.sigma = self.sigma_init
        self.sigma_decayfactor = 1000
        self.sigma_floor = 0.5

    def initialize(self):
        """
	Takes in six parameters
	    - the initialization method 'init'
	    - input data
	and sets M to a (J*K x F) array storing the initialized models.
	"""
        F = len(self.inputs[0])
        
        # create 3D array storing initial models
        #np.random.seed(1)
        if self.init=='random':
            min_val = np.min(self.inputs)
            max_val = np.max(self.inputs)
            self.M = np.random.uniform(min_val, max_val, size=(self.J*self.K, F))
            self.M = np.array(self.M)
        if self.init=="uniform":
            min_vals = np.min(self.inputs,axis=0)
            max_vals = np.max(self.inputs,axis=0)

            # Apply PCA to find 2 principal components
            # for now just pick 2 components with max sum
            temp = np.argsort(np.sum(self.inputs,axis=0))
            index1 = temp[len(temp)-1]
            index2 = temp[len(temp)-2]
            stepJ = (max_vals-min_vals)[index1]/self.J
            stepK = (max_vals-min_vals)[index2]/self.K

            self.M = np.array([[0 for i in range(len(self.inputs[0]))] for j in range (self.J*self.K)])
            for j in range(self.J):
                for k in range(self.K):
                    self.M[j*self.K + k][index1] = min_vals[index1] + j * stepJ
                    self.M[j*self.K + k][index2] = min_vals[index2] + k * stepK

		    
    def competitive_step(self, pt):
        """
        Implements the competitive step of SOM training. Takes in 
            - a training point
        Returns index of winning neuron
        """
        pt = [pt]
        # create a distance matrix between inputs and models
        distance_matrix = sp.cdist(pt, self.M, self.dist_func)

        # find the index of the winner model
        winner = np.nanargmin(distance_matrix, axis=1)

        return winner[0]
        
    def adaptive_step(self, winner, pt):
        """
        Implements the adaptive step of SOM training. Takes in index of winner neuron and training pt
        Updates model M and returns error (change in model)
        """
        # compute topo distance to winner
        wJ = int(winner/self.K); wK = winner % self.K

        # construct a list of sq topo dist
        topoPos = [[int(i/self.K),i % self.K] for i in range(len(self.M))]
        sqTopoDist = np.square(sp.cdist([[wJ,wK]],topoPos,"euclidean")).T
        # from that construct list of neigh. f values
        tMatrix = np.exp(-1 * sqTopoDist / (2* self.sigma**2))
        # construct pt - self.M
        diffMatrix = pt - self.M
        # construct delta
        deltaMatrix = self.learningRate * diffMatrix * tMatrix
        # update M with delta
        self.M += deltaMatrix
        
        error = np.sum(deltaMatrix)

        return error

    def train(self):
        """
        Trains the SOM with input data
        Returns models M after training
        """
        for iterations in range(self.max_iterations):
            # save history
            # histFrames = [1,100,500,1000,1500,2000,2500,3000,6000,10000]
            # if (iterations+1) in histFrames:
            #     if len(self.M_history) == 0:
            #         self.M_history = self.M.copy()
            #     else:
            #         self.M_history = np.concatenate((self.M_history,self.M),axis=0)
            
            randindex = np.random.randint(low=0,high=len(self.inputs))
            pt = self.inputs[randindex]
            
            winner = self.competitive_step(pt)
            
            error = self.adaptive_step(winner, pt)

            if not iterations % 100:
                print("Iteration", iterations," with adjustment", error)
                
            # reduce sigma over iterations
            if self.neighborhood_func == "gaussian":
                self.sigma = max(self.sigma_floor, self.sigma_init * np.exp(-1 * iterations/self.sigma_decayfactor))
            # return learning rate over iterations
            self.learningRate = self.learningRate_init * np.exp(-1 * iterations/self.learningRate_decayfactor)

        return self.M, self.M_history

def read_data(filepath):
    """

    """
    data = np.array(np.genfromtxt(filepath, dtype=float, delimiter=',', max_rows=100000))
    
    #return (None, data[:,1:])
    return (None, data)

    
def normalize_features(indicators):
    """
    Normalizes the data for each indicator over all countries.
    """
    # subtract column-wise mean and divide by column-wise standard error
    return((indicators-np.nanmean(indicators, axis=0))/np.nanstd(indicators, axis=0))


def main():
    # obtain command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('J', type=int, 
                        help='Integer number representing the width of the map.')
    parser.add_argument('K', type=int, 
                        help='Integer number representing the height of the map.')
    parser.add_argument('init', type=str,
                        help='String representing the initialization method to use')
    parser.add_argument('dist_func', type=str,
                        help='String representing the distance function to use')
    parser.add_argument('neighborhood_func', type=str,
                        help='String representing the neighborhood function to use')
    parser.add_argument('iterations', type=int,
                        help='Number of iterations to run')
    flags, unparsed = parser.parse_known_args()

    # read in dataset
    countries, indicators = read_data("../../hw6/number_data.txt")
    #countries, indicators = read_data("testdata.txt")
    
    # normalize feature values
    # inputs = normalize_features(indicators)
    inputs = indicators
    # initialize SOM 
    som = SOM_Single(J=flags.J, K=flags.K, inputs=inputs, init=flags.init, dist_func=flags.dist_func,
                     neighborhood_func=flags.neighborhood_func, iterations=flags.iterations)
    som.initialize()

    # train SOM
    final_model,model_history = som.train()

    pickle.dump(final_model,open("digitmodel.p","wb"))
    #pickle.dump(model_history,open("testmodelhistory.p","wb"))


if __name__ == '__main__':
    main()
