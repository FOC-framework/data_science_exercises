import pickle
from minisom import MiniSom

document_vector_array = pickle.load(open("state_files/tweets_vector.dat", "rb"))

som = MiniSom(10, 10, 100, sigma=0.3, learning_rate=0.5)  # initialization of 6x6 SOM
som.train(document_vector_array, 1000)  # trains the SOM with 100 iterations

pickle.dump(som, open("state_files/som.model", "wb"))
