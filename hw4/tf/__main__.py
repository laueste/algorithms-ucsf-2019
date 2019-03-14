import numpy as np
from .utils import word_encode,featurize_sequence,interpret_features,hamming_distance
from .io import make_negatives_file,read_seqs,write_seqs
from .prediction import screen_by_distance,screen_by_distance_reverse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

# PREPARATORY FILE PARSING

# Make the negative sequences file (just do this once, and delete the last newline)
# fasta = './data/yeast-upstream-1k-negative.fa'
# pos = './data/rap1-lieb-positives.txt'
# make_negatives_file(fasta,pos)
# pos_seqs = read_seqs('./data/rap1-lieb-positives.txt')
# neg_seqs = read_seqs('./data/rap1-lieb-constructed-negatives.txt')
# screen_by_distance(pos_seqs,neg_seqs)
# close_seqs = read_seqs('./data/rap1-lieb-close-negatives.txt')
# screen_by_distance_reverse(neg_seqs,close_seqs)

# READ IN DATA

rap1_pos = read_seqs('./data/rap1-lieb-positives.txt')
rap1_far_neg = read_seqs('./data/rap1-lieb-far-negatives.txt')
rap1_close_neg = read_seqs('./data/rap1-lieb-close-negatives.txt')
rap1_test = read_seqs('./data/rap1-lieb-test.txt')


# UNDERSAMPLING

# Make Dataset

# Split dataset using Stratified K-Fold

# Train

# Validate

# Repeat for 1:2, 1:5, 1:100 ratios of pos to neg



# OVERSAMPLING


# COMBINED SAMPLING


# Strategies

# undersample - just do a, hm, 5:1 neg to pos ratio, just make 137*5 negs,
# one set randomly from all the negative seqs, and 4 sets from close

# oversample - keep same 5:1 ratio, just upsample from the 137 up to, say, 60k
# samples total, mostly the close negatives with some far negatives and the
# appropriate amount of upsampled positives to reach 60k

# combination of both - upsample positives to 500, then get 500 far negatives
# and 500*4 close negatives


# try all of the above with a 2:1 ratio as well
# measure F1 score as well as accuracy


# use k-fold stratified cross-validation to learn hyperparameters in all cases










###  BASIC PRELIMINARY TESTS ###

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

# okay, well, for the given inputs, it's alright
