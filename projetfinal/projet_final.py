import pandas as pd
import tweet_cleaner
import gensim

# Parameters
NBR_TWEETS = 40000

EMBEDDING_DIM = 100
EMBEDDING_MIN_SIZE = 30

SOM_X_DIM = 6
SOM_Y_DIM = 6

DATA_FILE_ORIGINAL_TWEETS = 'data/tweets.csv'
STATE_FILE_ORIGINAL_TWEETS = 'state_files/w2v.model'


def do_embedding():
    data_pd = pd.read_csv(DATA_FILE_ORIGINAL_TWEETS, sep=',', skipinitialspace=True, )

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
    model.save(STATE_FILE_ORIGINAL_TWEETS)


def tweets_2_vectors():
    # This is the vector of documents / tweets
    document_vector_list = []

    #load the model back
    model = gensim.models.Word2Vec.load("state_files/w2v.model")

    data_pd = pd.read_csv('data/tweets.csv', sep=',', skipinitialspace=True,)

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

    pickle.dump(document_vector_array, open("state_files/tweets_vector.dat", "wb"))

do_embedding()
