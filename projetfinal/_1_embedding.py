import pandas as pd
import tweet_cleaner
import gensim

import params

data_pd = pd.read_csv('data/tweets.csv', sep=',', skipinitialspace=True,)

data_set = pd.DataFrame(data_pd)
# print(data_set.describe())
# print(data_pd.__len__())
# print(data_set.head())

col = data_pd['tweet_text']

# prepare an empty list
tweet_list = list()

tweets = data_set['tweet_text'].values.tolist()
for tweet in tweets[0:params.NBR_TWEETS]:
    print(tweet)

    words = tweet_cleaner.clean_tweet(tweet)

    tweet_list.append(words)
len(tweet_list)

model = gensim.models.Word2Vec(sentences=tweet_list, size=params.EMBEDDING_DIM, workers=4, min_count=params.EMBEDDING_MIN_SIZE)
model.save("state_files/w2v.model")

