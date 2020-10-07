import tnse_plot
import gensim

model = gensim.models.Word2Vec.load("state_files/w2v.model")

print(model.wv.most_similar('لبنان'))
print('--------------')
# print(model.wv.most_similar('الثورة'))
print('--------------')
print(model.wv.most_similar('باسيل'))

tnse_plot.tsne_plot(model)
# words = list(model.wv.vocab)
# for i in range(40):
#     print(words[i], ' -> ', words[i].count)

