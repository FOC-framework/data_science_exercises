import SimpSOM as sps
import gensim
import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

import tweet_cleaner

# This is the vector of documents / tweets
document_vector_list = []

#load the model back
model = gensim.models.Word2Vec.load("w2v.model")

data_pd = pd.read_csv('tweets.csv', sep=',', skipinitialspace=True,)

data_set = pd.DataFrame(data_pd)
col = data_pd['tweet_text']

# prepare an empty list
tweet_list = list()

tweets = data_set['tweet_text'].values.tolist()

for tweet in tweets[0:40000]:
    words = tweet_cleaner.clean_tweet(tweet)

    count = 0
    tweet_vector = []
    for w in words:
        if w in model:
            if count == 0:
                tweet_vector = model[w]
            else:
                tweet_vector = tweet_vector + model[w]
            count = count + 1

    if len(tweet_vector) > 0:
        document_vector_list.append(tweet_vector)

document_vector_array = np.array(document_vector_list)

# Build a network 20x20 with a weights format taken from the raw_data and activate Periodic Boundary Conditions.
net = sps.somNet(30, 30, document_vector_array, PBC=True)

# Train the network for 10000 epochs and with initial learning rate of 0.01.
net.train(0.01, 1000)

# Save the weights to file
net.save('state_files/som_weights')