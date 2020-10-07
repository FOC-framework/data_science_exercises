import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

do_plot = True

import gensim

model = gensim.models.Word2Vec.load("w2v.model")
print(model.wv.most_similar('لبنان'))
print('--------------')
print(model.wv.most_similar('الثورة'))
print('--------------')
print(model.wv.most_similar('باسيل'))
