import pandas as pd
import numpy as np
import gensim
import pickle
from minisom import MiniSom

# graphics
from bokeh.io import output_file, show
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter)
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import transform

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

    def micro_analysis(self, active):
        data_pd = pd.read_csv(self.data_file_tweets, sep=',', skipinitialspace=True, )
        document_vector_array = pickle.load(open(self.state_file_tweets_vector, "rb"))
        som = pickle.load(open(self.model_file_som, "rb"))

        if active:
            data_set = pd.DataFrame(data_pd)

            index = int(input('index = '))

            while index >= 0:
                # iterating the columns
                for col in data_set.columns:
                    print(col, '  -  ', data_set[col][index])

                print(document_vector_array[index])
                index = int(input('index = '))

    def heatmap(self):
        output_file("unemploymemt.html")

        data.Year = data.Year.astype(str)
        data = data.set_index('Year')
        data.drop('Annual', axis=1, inplace=True)
        data.columns.name = 'Month'

        # reshape to 1D array or rates with a month and year for each row.
        df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()

        source = ColumnDataSource(df)

        # this is the colormap from the original NYTimes plot
        colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
        mapper = LinearColorMapper(palette=colors, low=df.rate.min(), high=df.rate.max())

        p = figure(plot_width=800, plot_height=300, title="US Unemployment 1948—2016",
                   x_range=list(data.index), y_range=list(reversed(data.columns)),
                   toolbar_location=None, tools="", x_axis_location="above")

        p.rect(x="Year", y="Month", width=1, height=1, source=source,
               line_color=None, fill_color=transform('rate', mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="%d%%"))

        p.add_layout(color_bar, 'right')

        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "7px"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 1.0

        show(p)

sta211 = STA211()
sta211.word_embedding(False)
sta211.tweets_2_vectors(False)
sta211.som(False)
sta211.micro_analysis(True)
