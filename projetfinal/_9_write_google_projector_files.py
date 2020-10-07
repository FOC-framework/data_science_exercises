import tnse_plot
import gensim

model = gensim.models.Word2Vec.load("state_files/w2v.model")

vector_list = []
meta_list = []

for word in model.wv.vocab:
    for w in model[word]:
        vector_line = ""
        vector_line = vector_line + '\t'
        vector_line = vector_line + str(w)
    print(vector_line)
    print(word)

