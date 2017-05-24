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


class SOM():
    """
    Self-organizing map.
    """
    def __init__(self, J, K, init, dist_func, neighborhood_func, eps):
        """
	Store class variables.
	"""
	self.J = J; self.K = K
	self.init = init
	self.dist_func = dist_func
	self.neighborhood_func = neighborhood_func
	self.sigma = max(J,K)
        self.M = []
        self.L = cl.defaultdict(set)
        self.eps = eps
        self.error = float("inf")

    def initialize(init, inputs):
        """
	Takes in six parameters
	    - the initialization method 'init'
	    - input data
	and sets M to a (J*K x F) array storing the initialized models.
	"""
        F = len(inputs[0])
        min_val = np.min(inputs)
        max_val = np.max(inputs)
        
	np.random.seed(1)
        if init=='random':
            # create 3D array storing initial models
            self.M = np.random.uniform(min_val, max_val, size=(self.J*self.K, F))
            self.M = np.array(self.M)

		    
    def competitive_step(inputs):
        """
        Implements the competitive step of SOM training. Takes in 
            - the input data set
        Sets dictionary L of list of "won" data points for each neuron
        """
        # create a distance matrix between inputs and models
        distance_matrix = sp.cdist(inputs, self.M, self.dist_func)

        # for each input, find the index of the winner model
        winner_list = np.nanargmin(distance_matrix, axis=1)

        # for each data point, append to L the entry:
        #     key = i, index of the winning node
        #     value = x, index of "won" data point
        for x in range(len(winner_list)):
            i = winner_list[x]
            self.L[i].add(x)

    def adaptive_step():
        """
        Implements the adaptive step of SOM training. Takes in
        Updates model M
        """
        model_new = []
        # compute topological distance matrix
        indices = [i for i in range(len(self.M))]
        topological_distances = sp.cdist(indices,indices,self.neighborhood_func)

        # find length of L_j, mean(L_j)
        dictLen = [len(self.L[i]) for i in range(len(self.M))]
        meanpt = [[ [0 for j in range(len(self.M[0]))] if v == [] else np.mean(v,axis=0) for v in self.L[i]]
                  for i in range(len(self.M))]
        
        for i in range(len(M)):
            m_old = self.M[i]
            num = 0
            den = 0
            for j in range(len(M)):
                num += topological_distances[i][j] * dictLen[j] * meanpt[j]
                den += topological_distances[i][j] * dictLen[j]
                
            self.M[i] = num/den

            # check for convergence
            self.error = abs(1-np.max(m_old/self.M[i]))

    def train(inputs):
        """
        Trains the SOM with input data
        """
        iterations = 0
        while error > eps:
            iterations += 1
            self.competitive_step(inputs)
            self.adaptive_step()
            # incomplete. need to scale sigma

    def gaussian(x,y):
        """
        Gaussian neigborhood function 
            - x,y, indices of models X and Y
            - sigma, spread of the gaussian kernel
            - K, height of the SOM
        Returns the computed topological distance
        """
        x_j = int(x/K)
        x_k = x % K
        y_j = int(y/K)
        y_k = y % K

        sqdist = np.pow(x_j-y_j,2) + np.pow(x_k-y_k,2)

        return np.exp(-sqdist/(2*np.pow(self.sigma,2)))

def read_data():
    """
	Read in the data set to be clustered, along with the ground truth labels.
	Returns them stored in numpy arrays.
	"""
    data = np.genfromtxt('number_data.txt', dtype=int, delimiter=',', max_rows=1000)
	return(None, data)

    
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
	flags, unparsed = parser.parse_known_args()

	# read in dataset
	countries, indicators = read_data()
	# normalize feature values
	# inputs = normalize_features(indicators)
	inputs = indicators
	# initialize models
	initial_models = initialize(flags.J, flags.K, np.shape(inputs)[1], flags.init,
		                    np.nanmin(inputs, axis=0), np.nanmax(inputs, axis=1))
	
	L = competitive_step(inputs, initial_models, flags.dist_func)

if __name__ == '__main__':
    main()
