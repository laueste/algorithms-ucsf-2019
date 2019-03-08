# implement command line stuff for the basic neural network tests
import numpy as np

from .utils import NeuralNetwork

# QUESTION: HOW DO YOU SET DELTA FOR the BIAS WHEN THEN THE DELTA IS AN ARRAY?


# TODO: allow epoch of multiple result-output pairs


# Test values courtesy of:
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

#test_net =






# test_net = NeuralNetwork([8,3,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8])
# print(test_net)
# print()
sig = lambda z: 1/(1+np.exp(-z))
input = np.array([1,1,0])
mini_net = NeuralNetwork([3,2,1],input,[1])
# for l in mini_net.layers:
#     print(l.wts)
weights_0 = np.array([[0.6,0.7,0.2],[0.2,0.9,0.4],[0.7,0.5,0.8]])
weights_1 = np.array([[0.4,0.6,0.1],[0.9,0.1,0.3]])
weights_2 = np.array([[0.45,0.55]])
mini_net.layers[0].set_bias(np.array([1,1,1]).reshape(-1,1)) #change bias from zero so we can test it
mini_net.layers[0].set_weights_arr(weights_0)
mini_net.layers[1].set_weights_arr(weights_1)
mini_net.layers[2].set_weights_arr(weights_2)

encoder_inputs = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0] ]
encoder_outputs = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0] ]

test_net = NeuralNetwork([8,3,8],[1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0])
for i in range(100):
    print('**ITERATION',i,"**")
    test_net.backpropagate(new_x=encoder_inputs,new_y=encoder_outputs)
    print('propagation finished')
    results = test_net.feedforward(new_x=encoder_inputs)
    for r in results:
        print(list(map(lambda n:np.round(n,3),r)),max(r))
    print()
    print()
#print(mini_net.feedforward(new_input=[0,1,0,0,0,0,0,0]))
## INTERESTING, THIS ONLY ALLOWS OUTPUTS BETWEEN 0 and 1...oh right, because of the sigmoid duh
