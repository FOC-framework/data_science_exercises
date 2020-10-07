import SimpSOM as sps
import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

import tweet_cleaner

net = sps.load('state_files/som_weights')

# Print a map of the network nodes and colour them according to the first feature (column number 0) of the dataset
# and then according to the distance between each node and its neighbours.
net.nodes_graph(colnum=0)
net.diff_graph()

# Project the datapoints on the new 2D network map.
# net.project(document_vector_array, labels=labels)

# Cluster the datapoints according to the Quality Threshold algorithm.
# net.cluster(document_vector_array, type='qthresh')

plt.show()