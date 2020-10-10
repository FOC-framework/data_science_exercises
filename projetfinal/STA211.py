import pandas as pd
import numpy as np
import gensim
import pickle
from minisom import MiniSom
import matplotlib.pyplot as plt

# from pylab import bone, pcolor, colorbar, plot, show

# graphics
# from bokeh.io import output_file, show
# from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter)
# from bokeh.plotting import figure
# from bokeh.sampledata.unemployment1948 import data
# from bokeh.transform import transform

import tweet_cleaner


# Main Project class
class STA211:
    # Data Input parameters
    data_file_tweets = ''
    nbr_of_tweets = 0

    # Embedding parameter Word2Vec
    embedding_dim = 0
    embedding_min_size = 0
    model_file_w2v = ''

    # Tweets converted to vectors
    state_file_tweets_vector = ''

    # SOM parameters
    som_x_dim = 0
    som_y_dim = 0
    model_file_som = ''

    def __init__(self, data_file_tweets='data/tweets.csv', nbr_of_tweets=40000):
        self.data_file_tweets = data_file_tweets
        self.nbr_of_tweets = nbr_of_tweets

    # Word Embedding is the first step. Cleaning the arabic tweets then sending them to gensim.w2v model
    # Output is the model saved in a file
    def word_embedding(self, active, embedding_dim=100, embedding_min_size=30, model_file_w2v='state_files/w2v.model'):
        self.embedding_dim = embedding_dim
        self.embedding_min_size = embedding_min_size
        self.model_file_w2v = model_file_w2v

        if active:
            data_pd = pd.read_csv(self.data_file_tweets, sep=',', skipinitialspace=True, )

            data_set = pd.DataFrame(data_pd)
            # print(data_set.describe())
            # print(data_pd.__len__())
            # print(data_set.head())

            # prepare an empty list
            tweet_list = list()

            tweets = data_set['tweet_text'].values.tolist()
            for tweet in tweets[0:self.nbr_of_tweets]:
                print(tweet)

                words = tweet_cleaner.clean_tweet(tweet)

                tweet_list.append(words)
            len(tweet_list)

            model = gensim.models.Word2Vec(sentences=tweet_list, size=self.embedding_dim, workers=4, min_count=self.embedding_min_size)
            model.save(self.model_file_w2v)

    # The tweets need to be converted to vectors so we can send them into the SOM model
    def tweets_2_vectors(self, active, state_file_tweets_vector='state_files/tweets_vector.dat'):
        self.state_file_tweets_vector = state_file_tweets_vector
        if active:
            # This is the vector of documents / tweets
            tweets_vector_list = []

            # load the model back
            model = gensim.models.Word2Vec.load(self.model_file_w2v)

            data_pd = pd.read_csv(self.data_file_tweets, sep=',', skipinitialspace=True, )
            data_set = pd.DataFrame(data_pd)
            tweets = data_set['tweet_text'].values.tolist()

            for tweet in tweets[0:self.nbr_of_tweets]:
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
                    tweets_vector_list.append(tweet_vector)

            document_vector_array = np.array(tweets_vector_list)

            pickle.dump(document_vector_array, open(self.state_file_tweets_vector, "wb"))

    def som(self, active, model_file_som='state_files/som.model', som_x_dim=10, som_y_dim=10):
        self.som_x_dim = som_x_dim
        self.som_y_dim = som_y_dim
        self.model_file_som = model_file_som

        if active:
            document_vector_array = pickle.load(open(self.state_file_tweets_vector, "rb"))

            som = MiniSom(self.som_x_dim, self.som_y_dim, self.embedding_dim, sigma=0.3, learning_rate=0.5)  # initialization of 6x6 SOM
            som.train(document_vector_array, 1000)  # trains the SOM with 100 iterations

            pickle.dump(som, open(self.model_file_som, "wb"))

    def plotting_clusters(self, active):
        if active:
            data_pd = pd.read_csv(self.data_file_tweets, sep=',', skipinitialspace=True, )
            document_vector_array = pickle.load(open(self.state_file_tweets_vector, "rb"))
            som = pickle.load(open(self.model_file_som, "rb"))

            winner_coordinates = np.array([som.winner(x) for x in document_vector_array]).T
            # with np.ravel_multi_index we convert the bidimensional
            # coordinates to a monodimensional index
            som_shape = (self.som_x_dim, self.som_y_dim)
            cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

            # plotting the clusters using the first 2 dimentions of the data
            for c in np.unique(cluster_index):
                plt.scatter(document_vector_array[cluster_index == c, 0],
                            document_vector_array[cluster_index == c, 1], label='cluster=' + str(c), alpha=.7)

            # plotting centroids
            for centroid in som.get_weights():
                plt.scatter(centroid[:, 0], centroid[:, 1], marker='x',
                            s=80, linewidths=35, color='k', label='centroid')
            plt.legend()
            plt.show()

    def micro_analysis(self, active):
        if active:
            data_pd = pd.read_csv(self.data_file_tweets, sep=',', skipinitialspace=True, )
            document_vector_array = pickle.load(open(self.state_file_tweets_vector, "rb"))
            som = pickle.load(open(self.model_file_som, "rb"))

            data_set = pd.DataFrame(data_pd)

            index = int(input('index = '))

            while index >= 0:
                # iterating the columns
                for col in data_set.columns:
                    print(col, '  -  ', data_set[col][index])

                print(document_vector_array[index])
                index = int(input('index = '))

    def hierarchicalClustering(self, active):
        if active:
            import scipy.cluster.hierarchy as shc

            document_vector_array = pickle.load(open(self.state_file_tweets_vector, "rb"))
            som = pickle.load(open(self.model_file_som, "rb"))
            winner_coordinates = np.array([som.winner(x) for x in document_vector_array])
            weights = []
            [weights.append(som.get_weights()[i][j]) for i in range(self.som_x_dim) for j in range(self.som_y_dim)]
            weights = np.array(weights)

            # coordinates = som.get_euclidean_coordinates()

            plt.figure(figsize=(10, 7))
            plt.title("Neurones Dendrograms")
            linkage = shc.linkage(weights, method='average')
            dend = shc.dendrogram(linkage)
            plt.show()

            numclust = 5
            from scipy.cluster.hierarchy import fcluster
            fl = fcluster(linkage, numclust, criterion='maxclust')

            print(fl.shape)

            for c in range(numclust):
                x_c = []
                y_c = []
                for x in range(self.som_x_dim):
                    for y in range(self.som_y_dim):
                        cluster_index = fl[x*10+y]
                        if(cluster_index == c):
                            x_c.append(x)
                            y_c.append(y)
                plt.scatter(x_c, y_c, label='cluster=' + str(c), alpha=.7)

            plt.legend()
            plt.show()


sta211 = STA211()
sta211.word_embedding(False)
sta211.tweets_2_vectors(False)
sta211.som(False)
sta211.plotting_clusters(False)
sta211.micro_analysis(False)
sta211.hierarchicalClustering(True)
