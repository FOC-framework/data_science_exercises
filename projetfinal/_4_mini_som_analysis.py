import pickle
from minisom import MiniSom
import numpy as np
import pandas as pd

document_vector_array = pickle.load(open("state_files/tweets_vector.dat", "rb"))

som = pickle.load(open("state_files/som.model", "rb"))

# You can obtain the position of the winning neuron on the map for a given sample as follows:

neu1 = som.winner(document_vector_array[0])

print(neu1)


# each neuron represents a cluster
winner_coordinates = np.array([som.winner(x) for x in document_vector_array]).T
# with np.ravel_multi_index we convert the bidimensional
# coordinates to a monodimensional index
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)