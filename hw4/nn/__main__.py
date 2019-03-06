# implement command line stuff for the basic neural network tests
import numpy as np

from .utils import NeuralNetwork

# QUESTION: HOW DO YOU SET DELTA FOR the BIAS WHEN THEN THE DELTA IS AN ARRAY?


# TODO: allow epoch of multiple result-output pairs




# test_net = NeuralNetwork([8,3,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8])
# print(test_net)
# print()
sig = lambda z: 1/(1+np.exp(-z))
mini_net = NeuralNetwork([2,1],[1,0],[1])
input = np.array([1,0])
weights_1 = np.array([[0.4,0.6],[0.9,0.1]])
weights_2 = np.array([[0.45,0.55]])
mini_net.layers[0].b = 1 #change bias from zero so we can test it
mini_net.layers[0].set_weights_arr(weights_1)
mini_net.layers[1].set_weights_arr(weights_2)


test_net = NeuralNetwork([8,3,8],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0])
for i in range(1000):
    print('**ITERATION',i,"**")
    test_net.backpropagate()
    print("output:",test_net.feedforward())
    print("goal:",test_net.y)
    print()
    print()
print(test_net.feedforward(new_input=[0,1,0,0,0,0,0,0]))
## INTERESTING, THIS ONLY ALLOWS OUTPUTS BETWEEN 0 and 1...oh right, because of the sigmoid duh
