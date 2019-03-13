import numpy as np
from .utils import word_encode,featurize_sequence,interpret_features
from .io import make_negatives_file
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Make the negative sequences file (just do this once)
fasta = './data/yeast-upstream-1k-negative.fa'
pos = './data/rap1-lieb-positives.txt'
make_negatives_file(fasta,pos)





# test first with 8-3-8 encoder

# mlp = MLPClassifier(max_iter=5000,solver='adam',learning_rate_init=0.01,
#                 hidden_layer_sizes=(3))
# data = np.matrix([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
#         [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
#         [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
# mlp.fit(data,data)
# print("Training set score: %f" % mlp.score(data, data))
# print("Training set loss: %f" % mlp.loss_)

# okay, that works!

# try with sequences
# mlp = MLPClassifier(max_iter=5000,solver='adam',learning_rate_init=0.01,
#                 hidden_layer_sizes=(6))
# seqs = ['AAA','ATG','GGG','CCG','CCC','ACG','CAG','GAC','TGC','TAA','TTC','AAG','TTG','GGA','CAT']
# inputs = np.array([ featurize_sequence(s,word_len=2) for s in seqs])
# print(seqs)
# print([ interpret_features(i,2) for i in inputs ])
# mlp.fit(inputs,inputs)
# print("Training set score: %f" % mlp.score(inputs,inputs))
# print("Training set loss: %f" % mlp.loss_)
# print(mlp.predict(featurize_sequence('TTT').reshape(1,-1)))
# print(featurize_sequence('TTT').reshape(1,-1))
# print(interpret_features(
#         mlp.predict(featurize_sequence('TTT').reshape(1,-1))[0],2))
# print('TTT')

# okay, well, for the given inputs, it's alright!
