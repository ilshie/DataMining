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

# def read_data():
# 	"""
# 	Function that reads in the world development indicators dataset.
# 	"""
# 	# read in country names
# 	countries = np.genfromtxt("indicators_cleaned.txt", delimiter='\t', 
# 							usecols=0, dtype=str, skip_header=1)
# 	# read in indicators
# 	indicators = np.genfromtxt("indicators_cleaned.txt", delimiter='\t',
# 							usecols=range(1,101), dtype=float, skip_header=1)
# 	return(countries, indicators)

class SOM():
	"""
	Self-organizing map.
	"""
	def __init__(self, J, K, init, dist_func, neighborhood_func):
		"""
		Store class variables.
		"""
		self.J = J; self.K = K
		self.init = init
		self.dist_func = dist_func
		self.neighborhood_func = neighborhood_func
		self.sigma = max(J,K)


	def normalize_features(indicators):
		"""
		Normalizes the data for each indicator over all countries.
		"""
		# subtract column-wise mean and divide by column-wise standard error
		return((indicators-np.nanmean(indicators, axis=0))/np.nanstd(indicators, axis=0))

	def initialize(J, K, F, init, min_val, max_val):
		"""
		Takes in six parameters
			- the width of the 2D map J
			- the height of the 2D map K
			- the number of indicators i.e. features F
			- the initialization method 'init'
			- the minimum feature value
			- the maximum feature value
		and returns a (J*K x F) array storing the initialized models.
		"""
		np.random.seed(1)
		if init=='random':
			# create 3D array storing initial models
			M = np.random.uniform(0, 255, size=(J*K, F))
			return(np.array(M))
			
	def competitive_step(inputs, models, dist_func):
		"""
		Implements the competitive step of SOM training. Takes in 
			- the input data set
		    - the models associated with each neuron
		    - the user-specified distance metric
		Returns dictionary L of list of "won" data points for each neuron
		"""
		# create a distance matrix between inputs and models
		distance_matrix = sp.cdist(inputs, models, dist_func)

		# for each input, find the index of the winner model
		winner_list = np.nanargmin(distance_matrix, axis=1)

		# initialize dict L of list of "won" inputs for each neuron
		L = cl.defaultdict(set)

		# for each data point, append to L the entry:
		#     key = i, index of the winning node
		#     value = x, index of "won" data point
		for x in range(len(winner_list)):
			i = winner_list[x]
			L[i].add(x)

		return L

	def adaptive_step(L, models, topological_distances):
		"""
		Implements the adaptive step of SOM training. Takes in
			- L, the dictionary of "won" inputs for each neuron
			- the models associated with each neuron
			- matrix of topological distances between neurons
		Returns updated model
		"""
		model_new = []


	def kernel_gaussian(x,y):
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

		return np.exp(-sqdist/(2*np.pow(sigma,2)))

def read_data():
	"""
	Read in the data set to be clustered, along with the ground truth labels.
	Returns them stored in numpy arrays.
	"""
	data = np.genfromtxt('number_data.txt', dtype=int, delimiter=',', max_rows=1000)
	return(None, data)

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