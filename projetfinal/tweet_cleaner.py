import arabic_cleaner
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# nltk.download('stopwords')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# get the list of stopwords
arabic_stop_word_list = set(stopwords.words('arabic'))
english_stop_word_list = set(stopwords.words('english'))

arabic_additional_stop_words = ['عم', 'وعم', 'يلي']


def clean_tweet(tweet):
    # removing urls
    tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)

    # arabic cleaning unifying hamze and other special characters
    # also removes emojies
    tweet = arabic_cleaner.clean(tweet)

    # create word tokens and remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(tweet)

    words = [w.lower() for w in tokens]

    # remove arabic stop words
    words = [w for w in words if w not in arabic_stop_word_list]

    # remove some additional stop words
    words = [w for w in words if w not in arabic_additional_stop_words]

    # remove english stop words in case part of the tweets were english
    words = [w for w in words if w not in english_stop_word_list]

    # remove the words formed of 1 characters
    words = [w for w in words if len(w) > 1]

    return words
