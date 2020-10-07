import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
# nltk.download('stopwords')

data_pd = pd.read_csv('data/tweets.csv', sep=',', skipinitialspace=True,)

data_set = pd.DataFrame(data_pd)
print(data_set.describe())
print(data_set.head(5))

# iterating the columns
for col in data_set.columns:
    print(col, '  -  ', data_set[col][1])


